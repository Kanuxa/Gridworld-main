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
    scalar_patch_embed_dim: int = 48
    scalar_state_embed_dim: int = 32
    obs_embed_dim: int = 256
    gru_hidden_dim: int = 256
    head_hidden_dim: int = 128
    rnd_hidden_dim: int = 256
    rnd_output_dim: int = 128


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


class RecurrentPPORNDNetwork(nn.Module):
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
        scalar_state_dim: int = 6,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.use_vision = use_vision
        self.use_temperature_patch = use_temperature_patch
        self.use_smell_patch = use_smell_patch
        self.use_visited_memory = use_visited_memory
        self.use_hazard_memory = use_hazard_memory
        self.scalar_state_dim = scalar_state_dim
        self.patch_size = int(patch_size)

        if use_vision:
            self.vision_encoder = PatchEncoder(6, patch_size, cfg.conv_channels_1, cfg.conv_channels_2, cfg.vision_embed_dim)
        if use_temperature_patch:
            self.temperature_encoder = PatchEncoder(1, patch_size, max(8, cfg.conv_channels_1 // 2), max(16, cfg.conv_channels_2 // 2), cfg.scalar_patch_embed_dim)
        if use_smell_patch:
            self.smell_encoder = PatchEncoder(1, patch_size, max(8, cfg.conv_channels_1 // 2), max(16, cfg.conv_channels_2 // 2), cfg.scalar_patch_embed_dim)
        if use_visited_memory:
            self.visited_encoder = PatchEncoder(1, patch_size, max(8, cfg.conv_channels_1 // 2), max(16, cfg.conv_channels_2 // 2), cfg.scalar_patch_embed_dim)
        if use_hazard_memory:
            self.hazard_encoder = PatchEncoder(1, patch_size, max(8, cfg.conv_channels_1 // 2), max(16, cfg.conv_channels_2 // 2), cfg.scalar_patch_embed_dim)

        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_state_dim, cfg.scalar_state_embed_dim),
            nn.ReLU(),
        )

        fusion_dim = cfg.scalar_state_embed_dim
        if use_vision:
            fusion_dim += cfg.vision_embed_dim
        if use_temperature_patch:
            fusion_dim += cfg.scalar_patch_embed_dim
        if use_smell_patch:
            fusion_dim += cfg.scalar_patch_embed_dim
        if use_visited_memory:
            fusion_dim += cfg.scalar_patch_embed_dim
        if use_hazard_memory:
            fusion_dim += cfg.scalar_patch_embed_dim

        self.obs_fusion = nn.Sequential(
            nn.Linear(fusion_dim, cfg.obs_embed_dim),
            nn.ReLU(),
            nn.Linear(cfg.obs_embed_dim, cfg.obs_embed_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=cfg.obs_embed_dim + num_actions + 1,
            hidden_size=cfg.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.policy_head = nn.Sequential(
            nn.Linear(cfg.gru_hidden_dim, cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden_dim, num_actions),
        )
        self.ext_value_head = nn.Sequential(
            nn.Linear(cfg.gru_hidden_dim, cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden_dim, 1),
        )
        self.int_value_head = nn.Sequential(
            nn.Linear(cfg.gru_hidden_dim, cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden_dim, 1),
        )

        rnd_input_dim = (6 + 4) * patch_size * patch_size + scalar_state_dim
        self.rnd_target = nn.Sequential(
            nn.Linear(rnd_input_dim, cfg.rnd_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.rnd_hidden_dim, cfg.rnd_output_dim),
        )
        self.rnd_predictor = nn.Sequential(
            nn.Linear(rnd_input_dim, cfg.rnd_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.rnd_hidden_dim, cfg.rnd_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.rnd_hidden_dim, cfg.rnd_output_dim),
        )
        for param in self.rnd_target.parameters():
            param.requires_grad = False

    def initial_hidden(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.zeros((1, int(batch_size), self.gru.hidden_size), dtype=torch.float32, device=device)

    def encode_obs_sequence(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len = state["scalars"].shape[:2]

        def flatten_seq(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(batch_size * seq_len, *x.shape[2:])

        parts = []
        if self.use_vision:
            parts.append(self.vision_encoder(flatten_seq(state["vision"])))
        if self.use_temperature_patch:
            parts.append(self.temperature_encoder(flatten_seq(state["temperature_patch"])))
        if self.use_smell_patch:
            parts.append(self.smell_encoder(flatten_seq(state["smell_patch"])))
        if self.use_visited_memory:
            parts.append(self.visited_encoder(flatten_seq(state["visited_patch"])))
        if self.use_hazard_memory:
            parts.append(self.hazard_encoder(flatten_seq(state["hazard_patch"])))
        parts.append(self.scalar_encoder(flatten_seq(state["scalars"])))
        fused = self.obs_fusion(torch.cat(parts, dim=1))
        return fused.view(batch_size, seq_len, -1)

    def _flatten_state_sequence(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [
            state["vision"].reshape(state["vision"].shape[0], state["vision"].shape[1], -1),
            state["temperature_patch"].reshape(state["temperature_patch"].shape[0], state["temperature_patch"].shape[1], -1),
            state["smell_patch"].reshape(state["smell_patch"].shape[0], state["smell_patch"].shape[1], -1),
            state["visited_patch"].reshape(state["visited_patch"].shape[0], state["visited_patch"].shape[1], -1),
            state["hazard_patch"].reshape(state["hazard_patch"].shape[0], state["hazard_patch"].shape[1], -1),
            state["scalars"].reshape(state["scalars"].shape[0], state["scalars"].shape[1], -1),
        ]
        return torch.cat(parts, dim=-1)

    def forward_sequence(
        self,
        state: Dict[str, torch.Tensor],
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        hidden: torch.Tensor | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_embed = self.encode_obs_sequence(state)
        gru_input = torch.cat([obs_embed, prev_action, prev_reward], dim=-1)
        if reset_mask is None:
            out, hidden_out = self.gru(gru_input, hidden)
        else:
            batch_size, seq_len = gru_input.shape[:2]
            outputs = []
            hidden_out = hidden
            for t in range(seq_len):
                reset_t = reset_mask[:, t].to(gru_input.dtype).view(1, batch_size, 1)
                if hidden_out is not None:
                    hidden_out = hidden_out * (1.0 - reset_t)
                out_t, hidden_out = self.gru(gru_input[:, t : t + 1], hidden_out)
                outputs.append(out_t)
            out = torch.cat(outputs, dim=1)
        logits = self.policy_head(out)
        ext_value = self.ext_value_head(out).squeeze(-1)
        int_value = self.int_value_head(out).squeeze(-1)
        return logits, ext_value, int_value, hidden_out

    def rnd_prediction_error(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_state = self._flatten_state_sequence(state)
        batch_size, seq_len, dim = flat_state.shape
        flat_state = flat_state.reshape(batch_size * seq_len, dim)
        pred = self.rnd_predictor(flat_state)
        with torch.no_grad():
            target = self.rnd_target(flat_state)
        error = (pred - target).pow(2).mean(dim=-1).view(batch_size, seq_len)
        return error, pred, target


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


def obs_to_state(obs: Dict[str, Any], env_cfg: EnvConfig, switches: ObservationSwitches) -> Dict[str, np.ndarray]:
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

    scalars = np.concatenate(
        [
            np.array([float(obs["health_norm"]), float(obs["energy_norm"])], dtype=np.float32),
            np.asarray(obs["direction_onehot"], dtype=np.float32),
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
        scalar_patch_embed_dim=int(kwargs.get("scalar_patch_embed_dim", 48)),
        scalar_state_embed_dim=int(kwargs.get("scalar_state_embed_dim", 32)),
        obs_embed_dim=int(kwargs.get("obs_embed_dim", 256)),
        gru_hidden_dim=int(kwargs.get("gru_hidden_dim", 256)),
        head_hidden_dim=int(kwargs.get("head_hidden_dim", 128)),
        rnd_hidden_dim=int(kwargs.get("rnd_hidden_dim", 256)),
        rnd_output_dim=int(kwargs.get("rnd_output_dim", 128)),
    )


def build_model_from_checkpoint(payload: Dict[str, Any], device: str = "cpu") -> nn.Module:
    kwargs = payload.get("model_kwargs", {}) if isinstance(payload.get("model_kwargs", {}), dict) else {}
    env_cfg_payload = payload.get("env_config", {}) if isinstance(payload.get("env_config", {}), dict) else {}
    patch_size = int(kwargs.get("patch_size", env_cfg_payload.get("patch_size", 5)))
    num_actions = int(payload.get("num_actions", kwargs.get("num_actions", N_ACTIONS)))
    model_cfg = model_config_from_kwargs(kwargs)
    net = RecurrentPPORNDNetwork(
        patch_size=patch_size,
        num_actions=num_actions,
        cfg=model_cfg,
        use_vision=bool(kwargs.get("use_vision", True)),
        use_temperature_patch=bool(kwargs.get("use_temperature_patch", True)),
        use_smell_patch=bool(kwargs.get("use_smell_patch", True)),
        use_visited_memory=bool(kwargs.get("use_visited_memory", True)),
        use_hazard_memory=bool(kwargs.get("use_hazard_memory", True)),
        scalar_state_dim=int(kwargs.get("scalar_state_dim", 6)),
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
    state = obs_to_state(obs, env_cfg, switches)
    state_t = {k: torch.from_numpy(v).float().unsqueeze(0).unsqueeze(0).to(device) for k, v in state.items()}
    prev_action_t = torch.from_numpy(np.asarray(ctx.get("prev_action", np.zeros((N_ACTIONS,), dtype=np.float32)), dtype=np.float32)).view(1, 1, -1).to(device)
    prev_reward_t = torch.tensor([[[float(ctx.get("prev_reward", 0.0))]]], dtype=torch.float32, device=device)
    hidden = ctx.get("hidden", None)
    with torch.no_grad():
        logits, _, _, hidden_out = net.forward_sequence(state_t, prev_action_t, prev_reward_t, hidden)
        action = int(torch.argmax(logits[:, -1], dim=-1).item())
    next_ctx = dict(ctx)
    next_ctx["hidden"] = hidden_out.detach()
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
    return ctx
