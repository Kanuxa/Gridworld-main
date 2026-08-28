"""Streamlit adapter for the planner-residual spatial-target PPO checkpoint format."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict, Tuple

import numpy as np
import torch

from gui.current_environment.sensory_grid_env import ACTION_FORWARD, EnvConfig, ObservationSwitches, SensoryGridEnv
from models.model_based.ppo.spatial_target_ppo_with_planner.memory import CoverageMemory
from models.non_model_based.partial_observation_frontier_planner.core import StaticFrontierPlanner
from models.model_based.ppo.planner_residual_target_ppo.model import PlannerResidualTargetActorCritic
from models.model_based.ppo.planner_residual_target_ppo.train import (
    LEGACY_MODEL_TYPE,
    MODEL_TYPE,
    PlannerResidualTrainConfig,
    build_env,
    candidate_cells,
    flat_for_world,
    planner_logits,
    policy_distribution,
    residual_limit,
    target_mask_and_lookup,
    target_tensors,
)


TRAINER_GUI_INTERFACE_VERSION = "planner-residual-spatial-target-ppo"
TRAINER_DISPLAY_NAME = "Planner-anchored residual spatial-target PPO"


def build_training_env(cfg: PlannerResidualTrainConfig | None = None) -> Tuple[SensoryGridEnv, ObservationSwitches]:
    return build_env((cfg or PlannerResidualTrainConfig()).environment_preset)


def _config_from_payload(payload: Dict[str, Any]) -> PlannerResidualTrainConfig:
    saved = payload.get("train_config", {})
    if not isinstance(saved, dict):
        return PlannerResidualTrainConfig()
    allowed = {field.name for field in fields(PlannerResidualTrainConfig)}
    return PlannerResidualTrainConfig(**{key: value for key, value in saved.items() if key in allowed})


def build_model_from_checkpoint(payload: Dict[str, Any]) -> PlannerResidualTargetActorCritic:
    if payload.get("model_type") not in {MODEL_TYPE, LEGACY_MODEL_TYPE}:
        raise ValueError("Please load a planner-residual target PPO checkpoint from runs/{15x15,31x31,45x45}/planner_residual_target_ppo.")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint has no model_state_dict.")
    net = PlannerResidualTargetActorCritic()
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    net.planner_residual_target_ppo_config = _config_from_payload(payload)
    return net


def init_runtime_context() -> Dict[str, object]:
    return {"memory": None, "planner": None, "steps": 0}


def reset_runtime_context(_context: object | None = None) -> Dict[str, object]:
    return init_runtime_context()


def _context_for_observation(context: object | None, observation: Dict[str, object], env_config: EnvConfig) -> Dict[str, object]:
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
    net: PlannerResidualTargetActorCritic,
    observation: Dict[str, object],
    env_config: EnvConfig,
    switches: ObservationSwitches,
    runtime_context: object | None,
) -> Tuple[int, Dict[str, object]]:
    del switches
    context = _context_for_observation(runtime_context, observation, env_config)
    memory, planner = context["memory"], context["planner"]
    assert isinstance(memory, CoverageMemory)
    assert isinstance(planner, StaticFrontierPlanner)
    cfg = getattr(net, "planner_residual_target_ppo_config", PlannerResidualTrainConfig())
    if not isinstance(cfg, PlannerResidualTrainConfig):
        cfg = PlannerResidualTrainConfig()
    _, local_sensor, scalars = memory.state(observation, env_config.max_steps - int(context.get("steps", 0)))
    prior, teacher_world_flat, meat_mode = planner.action_prior(memory, float(observation["health_norm"]), float(observation["energy_norm"]))
    teacher_row, teacher_col = divmod(int(teacher_world_flat), memory.grid_size)
    mask, lookup = target_mask_and_lookup(memory, candidate_cells(memory, meat_mode))
    if (teacher_row, teacher_col) not in lookup.values():
        teacher_row, teacher_col = min(lookup.values(), key=lambda cell: abs(cell[0] - teacher_row) + abs(cell[1] - teacher_col))
    teacher_target = flat_for_world(memory, teacher_row, teacher_col)
    device = next(net.parameters()).device
    map_t, local_t, scalar_t = target_tensors(memory.agent_centred_map(), local_sensor, scalars, device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
    scores_t = torch.from_numpy(planner_logits(memory, planner, lookup, meat_mode, teacher_target, cfg)).unsqueeze(0).to(device)
    distribution, _, _ = policy_distribution(net, map_t, local_t, scalar_t, mask_t, scores_t, residual_limit(0, cfg))
    selected = int(torch.argmax(distribution.logits[0]).item())
    action, _, _ = planner.action_toward(memory, *lookup[selected])
    return int(np.argmax(prior)) if action is None else int(action), context


def update_runtime_context_after_env_step(runtime_context: object | None, action: int, _reward: float, done: bool) -> Dict[str, object]:
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
        "policy": "planner scores plus bounded spatial neural residual",
        "executor": "thermal/revisit-aware Dijkstra; one action then replan",
        "checkpoint_format": "runs/{15x15,31x31,45x45}/planner_residual_target_ppo/*.pt",
    }
