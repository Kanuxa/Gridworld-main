from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from gui.current_environment.sensory_grid_env import EnvConfig, ObservationSwitches, N_ACTIONS


@dataclass
class ModelConfig:
    conv_channels_1: int = 16
    conv_channels_2: int = 32
    vision_embed_dim: int = 96
    thermal_embed_dim: int = 48
    sensing_embed_dim: int = 64
    scalar_state_embed_dim: int = 64
    obs_embed_dim: int = 256
    lstm_hidden_dim: int = 256
    head_hidden_dim: int = 128
    num_atoms: int = 51
    v_min: float = -600.0
    v_max: float = 50.0


class PatchEncoder(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, c1: int, c2: int, embed_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * patch_size * patch_size, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x))


class RecurrentFusionLSTMC51DuelingQNetwork(nn.Module):
    def __init__(
        self,
        patch_size: int,
        num_actions: int,
        cfg: ModelConfig,
        use_vision: bool = True,
        use_temperature_patch: bool = True,
        use_smell_patch: bool = True,
        use_visited_memory: bool = True,
        use_hazard_memory: bool = True,
        scalar_state_dim: int = 12,
    ):
        super().__init__()
        self.num_actions = int(num_actions)
        self.num_atoms = int(cfg.num_atoms)
        self.use_vision = bool(use_vision)
        self.use_temperature_patch = bool(use_temperature_patch)
        self.use_smell_patch = bool(use_smell_patch)
        self.use_visited_memory = bool(use_visited_memory)
        self.use_hazard_memory = bool(use_hazard_memory)

        if self.use_vision:
            self.vision_encoder = PatchEncoder(6, patch_size, cfg.conv_channels_1, cfg.conv_channels_2, cfg.vision_embed_dim)
        if self.use_temperature_patch:
            self.thermal_encoder = PatchEncoder(
                1,
                patch_size,
                max(8, cfg.conv_channels_1 // 2),
                max(16, cfg.conv_channels_2 // 2),
                cfg.thermal_embed_dim,
            )

        sensing_channels = int(self.use_smell_patch) + int(self.use_visited_memory) + int(self.use_hazard_memory)
        self.use_sensing_stack = sensing_channels > 0
        if self.use_sensing_stack:
            self.sensing_encoder = PatchEncoder(
                sensing_channels,
                patch_size,
                max(8, cfg.conv_channels_1 // 2),
                max(16, cfg.conv_channels_2 // 2),
                cfg.sensing_embed_dim,
            )

        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_state_dim, cfg.scalar_state_embed_dim),
            nn.ReLU(),
            nn.Linear(cfg.scalar_state_embed_dim, cfg.scalar_state_embed_dim),
            nn.ReLU(),
        )

        fusion_dim = cfg.scalar_state_embed_dim
        if self.use_vision:
            fusion_dim += cfg.vision_embed_dim
        if self.use_temperature_patch:
            fusion_dim += cfg.thermal_embed_dim
        if self.use_sensing_stack:
            fusion_dim += cfg.sensing_embed_dim

        self.obs_fusion = nn.Sequential(
            nn.Linear(fusion_dim, cfg.obs_embed_dim),
            nn.ReLU(),
            nn.Linear(cfg.obs_embed_dim, cfg.obs_embed_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=cfg.obs_embed_dim,
            hidden_size=cfg.lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.value_head = nn.Sequential(
            nn.Linear(cfg.lstm_hidden_dim, cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden_dim, self.num_atoms),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(cfg.lstm_hidden_dim, cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden_dim, self.num_actions * self.num_atoms),
        )
        support = torch.linspace(float(cfg.v_min), float(cfg.v_max), self.num_atoms, dtype=torch.float32)
        self.register_buffer("support", support)

    def encode_obs_sequence(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len = state["scalars"].shape[:2]

        def flatten_seq(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(batch_size * seq_len, *x.shape[2:])

        parts = []
        if self.use_vision:
            parts.append(self.vision_encoder(flatten_seq(state["vision"])))
        if self.use_temperature_patch:
            parts.append(self.thermal_encoder(flatten_seq(state["temperature_patch"])))
        if self.use_sensing_stack:
            sensing_parts = []
            if self.use_smell_patch:
                sensing_parts.append(state["smell_patch"])
            if self.use_visited_memory:
                sensing_parts.append(state["visited_patch"])
            if self.use_hazard_memory:
                sensing_parts.append(state["hazard_patch"])
            sensing = torch.cat(sensing_parts, dim=2)
            parts.append(self.sensing_encoder(flatten_seq(sensing)))
        parts.append(self.scalar_encoder(flatten_seq(state["scalars"])))
        fused = self.obs_fusion(torch.cat(parts, dim=1))
        return fused.view(batch_size, seq_len, -1)

    def q_values_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        return torch.sum(probs * self.support.view(1, 1, 1, -1), dim=-1)

    def forward_sequence(
        self,
        state: Dict[str, torch.Tensor],
        hidden: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        obs_embed = self.encode_obs_sequence(state)
        out, hidden_out = self.lstm(obs_embed, hidden)
        value_logits = self.value_head(out).unsqueeze(-2)
        advantage_logits = self.adv_head(out).view(out.shape[0], out.shape[1], self.num_actions, self.num_atoms)
        logits = value_logits + advantage_logits - advantage_logits.mean(dim=-2, keepdim=True)
        q_values = self.q_values_from_logits(logits)
        return logits, q_values, hidden_out


def choose_device(device_str: str = "cpu") -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def onehot_action(action: int) -> np.ndarray:
    vec = np.zeros(N_ACTIONS, dtype=np.float32)
    if 0 <= int(action) < N_ACTIONS:
        vec[int(action)] = 1.0
    return vec


def obs_to_state(
    obs: Dict[str, Any],
    env_cfg: EnvConfig,
    switches: ObservationSwitches,
    prev_action: np.ndarray | None = None,
    prev_reward: float = 0.0,
    episode_step: int = 0,
) -> Dict[str, np.ndarray]:
    patch_size = env_cfg.patch_size

    if switches.include_vision:
        vision_ids = np.asarray(obs["vision"], dtype=np.int64)
        vision = np.eye(6, dtype=np.float32)[vision_ids].transpose(2, 0, 1)
    else:
        vision = np.zeros((6, patch_size, patch_size), dtype=np.float32)

    if switches.include_temperature_patch:
        temperature_patch_c = np.asarray(obs["temperature_patch_c"], dtype=np.float32)
        scale = max(1.0, env_cfg.fire_temp_delta_amp, env_cfg.ice_temp_delta_amp)
        temperature_patch = np.clip((temperature_patch_c - env_cfg.ambient_temperature_c) / scale, -1.0, 1.0)
    else:
        temperature_patch = np.zeros((patch_size, patch_size), dtype=np.float32)

    if switches.include_smell_patch:
        smell_patch = np.clip(np.asarray(obs["smell_patch"], dtype=np.float32), 0.0, 1.0)
    else:
        smell_patch = np.zeros((patch_size, patch_size), dtype=np.float32)

    if switches.include_visited_memory:
        visited_patch = np.asarray(obs["visited_patch"], dtype=np.float32)
    else:
        visited_patch = np.zeros((patch_size, patch_size), dtype=np.float32)

    if switches.include_hazard_memory:
        hazard_patch = np.asarray(obs["hazard_patch"], dtype=np.float32)
    else:
        hazard_patch = np.zeros((patch_size, patch_size), dtype=np.float32)

    prev_action_vec = np.asarray(prev_action if prev_action is not None else np.zeros((N_ACTIONS,), dtype=np.float32), dtype=np.float32)
    prev_reward_scaled = float(np.tanh(float(prev_reward) / 10.0))
    step_fraction = float(np.clip(float(episode_step) / max(1.0, float(env_cfg.max_steps)), 0.0, 1.0))
    remaining_lives_flag = 1.0 if float(obs.get("health", 0.0)) > 0.0 else 0.0

    scalars = np.concatenate(
        [
            np.array(
                [
                    float(obs["health_norm"]),
                    float(obs["energy_norm"]),
                    remaining_lives_flag,
                ],
                dtype=np.float32,
            ),
            np.asarray(obs["direction_onehot"], dtype=np.float32),
            prev_action_vec,
            np.array([prev_reward_scaled, step_fraction], dtype=np.float32),
        ],
        axis=0,
    )

    return {
        "vision": vision.astype(np.float32),
        "temperature_patch": temperature_patch[None, :, :].astype(np.float32),
        "smell_patch": smell_patch[None, :, :].astype(np.float32),
        "visited_patch": visited_patch[None, :, :].astype(np.float32),
        "hazard_patch": hazard_patch[None, :, :].astype(np.float32),
        "scalars": scalars.astype(np.float32),
    }


def model_config_from_kwargs(kwargs: Dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        conv_channels_1=int(kwargs.get("conv_channels_1", 16)),
        conv_channels_2=int(kwargs.get("conv_channels_2", 32)),
        vision_embed_dim=int(kwargs.get("vision_embed_dim", 96)),
        thermal_embed_dim=int(kwargs.get("thermal_embed_dim", 48)),
        sensing_embed_dim=int(kwargs.get("sensing_embed_dim", 64)),
        scalar_state_embed_dim=int(kwargs.get("scalar_state_embed_dim", 64)),
        obs_embed_dim=int(kwargs.get("obs_embed_dim", 256)),
        lstm_hidden_dim=int(kwargs.get("lstm_hidden_dim", 256)),
        head_hidden_dim=int(kwargs.get("head_hidden_dim", 128)),
        num_atoms=int(kwargs.get("num_atoms", 51)),
        v_min=float(kwargs.get("v_min", -600.0)),
        v_max=float(kwargs.get("v_max", 50.0)),
    )


def build_model_from_checkpoint(payload: Dict[str, Any], device: str = "cpu") -> nn.Module:
    kwargs = payload.get("model_kwargs", {}) if isinstance(payload.get("model_kwargs", {}), dict) else {}
    env_cfg_payload = payload.get("env_config", {}) if isinstance(payload.get("env_config", {}), dict) else {}
    patch_size = int(kwargs.get("patch_size", env_cfg_payload.get("patch_size", 5)))
    num_actions = int(payload.get("num_actions", kwargs.get("num_actions", N_ACTIONS)))
    model_cfg = model_config_from_kwargs(kwargs)
    net = RecurrentFusionLSTMC51DuelingQNetwork(
        patch_size=patch_size,
        num_actions=num_actions,
        cfg=model_cfg,
        use_vision=bool(kwargs.get("use_vision", True)),
        use_temperature_patch=bool(kwargs.get("use_temperature_patch", True)),
        use_smell_patch=bool(kwargs.get("use_smell_patch", True)),
        use_visited_memory=bool(kwargs.get("use_visited_memory", True)),
        use_hazard_memory=bool(kwargs.get("use_hazard_memory", True)),
        scalar_state_dim=int(kwargs.get("scalar_state_dim", 12)),
    )
    net.load_state_dict(payload["model_state_dict"], strict=True)
    net.to(choose_device(device))
    net.eval()
    return net


def init_runtime_context(device: str = "cpu") -> Dict[str, Any]:
    return {
        "hidden": None,
        "prev_action": np.zeros((N_ACTIONS,), dtype=np.float32),
        "prev_reward": 0.0,
        "episode_step": 0,
        "device": str(device),
    }


def reset_runtime_context(context: Dict[str, Any] | None = None, device: str | None = None) -> Dict[str, Any]:
    base_device = str(device) if device is not None else (str(context.get("device", "cpu")) if context is not None else "cpu")
    return init_runtime_context(device=base_device)


def predict_action_for_gui(
    net: nn.Module,
    obs: Dict[str, Any],
    env_cfg: EnvConfig,
    switches: ObservationSwitches,
    runtime_context: Dict[str, Any] | None = None,
) -> Tuple[int, Dict[str, Any]]:
    ctx = runtime_context or init_runtime_context()
    device = choose_device(str(ctx.get("device", "cpu")))
    state = obs_to_state(
        obs,
        env_cfg,
        switches,
        prev_action=np.asarray(ctx.get("prev_action", np.zeros((N_ACTIONS,), dtype=np.float32)), dtype=np.float32),
        prev_reward=float(ctx.get("prev_reward", 0.0)),
        episode_step=int(ctx.get("episode_step", 0)),
    )
    state_t = {k: torch.from_numpy(v).float().unsqueeze(0).unsqueeze(0).to(device) for k, v in state.items()}
    hidden = ctx.get("hidden", None)
    with torch.no_grad():
        _, q_values, hidden_out = net.forward_sequence(state_t, hidden)
        action = int(torch.argmax(q_values[:, -1], dim=-1).item())
    next_ctx = dict(ctx)
    next_ctx["hidden"] = tuple(item.detach() for item in hidden_out)
    next_ctx["device"] = str(device)
    return action, next_ctx


def update_runtime_context_after_env_step(
    runtime_context: Dict[str, Any] | None,
    action: int,
    reward: float,
    done: bool = False,
) -> Dict[str, Any]:
    ctx = dict(runtime_context or init_runtime_context())
    if done:
        return reset_runtime_context(ctx)
    ctx["prev_action"] = onehot_action(action)
    ctx["prev_reward"] = float(reward)
    ctx["episode_step"] = int(ctx.get("episode_step", 0)) + 1
    return ctx
