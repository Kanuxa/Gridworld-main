"""Streamlit adapter for spatial-target PPO-with-planner checkpoint."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict, Tuple

import numpy as np
import torch

from gui.current_environment.sensory_grid_env import (
    ACTION_FORWARD,
    ACTION_LEFT,
    ACTION_RIGHT,
    EnvConfig,
    N_ACTIONS,
    ObservationSwitches,
    SensoryGridEnv,
)
from models.model_based.ppo.spatial_target_ppo_with_planner.memory import CoverageMemory, StaticFrontierPlanner
from models.model_based.ppo.spatial_target_ppo_with_planner.model import TargetMapActorCritic
from models.model_based.ppo.spatial_target_ppo_with_planner.train import (
    LEGACY_MODEL_TYPE,
    MODEL_TYPE,
    TargetTrainConfig,
    build_env,
    candidate_cells,
    flat_for_world,
    target_mask_and_lookup,
    target_policy_distribution,
    target_tensors,
)


TRAINER_GUI_INTERFACE_VERSION = "spatial-target-ppo-with-planner"
TRAINER_DISPLAY_NAME = "Spatial target PPO with Dijkstra planner executor"


def build_training_env(cfg: TargetTrainConfig | None = None) -> Tuple[SensoryGridEnv, ObservationSwitches]:
    del cfg
    return build_env()


def _config_from_payload(payload: Dict[str, Any]) -> TargetTrainConfig:
    saved = payload.get("train_config", {})
    if not isinstance(saved, dict):
        return TargetTrainConfig()
    allowed = {field.name for field in fields(TargetTrainConfig)}
    return TargetTrainConfig(**{key: value for key, value in saved.items() if key in allowed})


def build_model_from_checkpoint(payload: Dict[str, Any]) -> TargetMapActorCritic:
    if payload.get("model_type") not in {MODEL_TYPE, LEGACY_MODEL_TYPE}:
        raise ValueError("Please load a spatial-target PPO checkpoint from runs/15x15/spatial_target_ppo_with_planner.")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint has no model_state_dict.")
    net = TargetMapActorCritic()
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    net.spatial_target_ppo_config = _config_from_payload(payload)
    return net


def init_runtime_context() -> Dict[str, object]:
    return {"memory": None, "planner": None, "steps": 0}


def reset_runtime_context(_context: object | None = None) -> Dict[str, object]:
    return init_runtime_context()


def _context_for_observation(
    context: object | None,
    observation: Dict[str, object],
    env_config: EnvConfig,
) -> Dict[str, object]:
    active = context if isinstance(context, dict) else init_runtime_context()
    memory = active.get("memory")
    if not isinstance(memory, CoverageMemory) or memory.grid_size != env_config.grid_size:
        memory = CoverageMemory(env_config.grid_size, env_config.patch_size, env_config.ambient_temperature_c)
        memory.reset(int(observation["direction"]))
        active = {"memory": memory, "planner": StaticFrontierPlanner(), "steps": 0}
    memory.update(observation)
    if not isinstance(active.get("planner"), StaticFrontierPlanner):
        active["planner"] = StaticFrontierPlanner()
    return active


@torch.no_grad()
def predict_action_for_gui(
    net: TargetMapActorCritic,
    observation: Dict[str, object],
    env_config: EnvConfig,
    switches: ObservationSwitches,
    runtime_context: object | None,
) -> Tuple[int, Dict[str, object]]:
    del switches
    context = _context_for_observation(runtime_context, observation, env_config)
    memory = context["memory"]
    planner = context["planner"]
    assert isinstance(memory, CoverageMemory)
    assert isinstance(planner, StaticFrontierPlanner)
    cfg = getattr(net, "spatial_target_ppo_config", TargetTrainConfig())
    if not isinstance(cfg, TargetTrainConfig):
        cfg = TargetTrainConfig()

    _, local_sensor, scalars = memory.state(observation, env_config.max_steps - int(context.get("steps", 0)))
    planner_prior, teacher_world_flat, meat_mode = planner.action_prior(
        memory, float(observation["health_norm"]), float(observation["energy_norm"])
    )
    teacher_row, teacher_col = divmod(int(teacher_world_flat), memory.grid_size)
    target_mask, lookup = target_mask_and_lookup(memory, candidate_cells(memory, meat_mode))
    if (teacher_row, teacher_col) not in lookup.values():
        teacher_row, teacher_col = min(
            lookup.values(),
            key=lambda cell: abs(cell[0] - teacher_row) + abs(cell[1] - teacher_col),
        )
    teacher_target = flat_for_world(memory, teacher_row, teacher_col)
    device = next(net.parameters()).device
    map_t, local_t, scalar_t = target_tensors(memory.agent_centred_map(), local_sensor, scalars, device)
    mask_t = torch.from_numpy(target_mask).unsqueeze(0).to(device)
    teacher_t = torch.tensor([teacher_target], dtype=torch.long, device=device)
    distribution, _, _ = target_policy_distribution(
        net, map_t, local_t, scalar_t, mask_t, teacher_t, 0.0, cfg
    )
    selected_target = int(torch.argmax(distribution.logits[0]).item())
    target_row, target_col = lookup[selected_target]
    action, _, _ = planner.action_toward(memory, target_row, target_col)
    if action is None:
        action = int(np.argmax(planner_prior))
    return int(action), context


def update_runtime_context_after_env_step(
    runtime_context: object | None,
    action: int,
    _reward: float,
    done: bool,
) -> Dict[str, object]:
    if done:
        return reset_runtime_context(runtime_context)
    context = runtime_context if isinstance(runtime_context, dict) else init_runtime_context()
    memory = context.get("memory")
    if isinstance(memory, CoverageMemory):
        memory.advance(int(action))
    context["steps"] = int(context.get("steps", 0)) + 1
    return context


def get_gui_interface_spec() -> Dict[str, object]:
    return {
        "version": TRAINER_GUI_INTERFACE_VERSION,
        "policy": "agent-centred spatial target selection",
        "executor": "thermal/revisit-aware Dijkstra; one action then replan",
        "checkpoint_format": "runs/15x15/spatial_target_ppo_with_planner/*.pt",
    }
