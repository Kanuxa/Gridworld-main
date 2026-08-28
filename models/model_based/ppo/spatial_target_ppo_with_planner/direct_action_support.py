"""Shared spatial-target PPO training utilities retained for the selected target hierarchy.

The legacy direct-action PPO entry point and its runs were removed. The target
hierarchy imports environment, device, and local safety helpers from here.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from gui.current_environment.sensory_grid_env import ACTION_FORWARD, EnvConfig, ObservationSwitches, SensoryGridEnv
from models.model_based.ppo.spatial_target_ppo_with_planner.memory import CoverageMemory, HAZARD, HEADING_NAMES, MEAT, StaticFrontierPlanner, user_coordinate
from models.model_based.ppo.spatial_target_ppo_with_planner.model import NeuralMapActorCritic
from models.model_based.ppo.spatial_target_ppo_with_planner.trace import EpisodeTrace


@dataclass
class TrainConfig:
    episodes: int = 3000
    rollout_episodes: int = 16
    ppo_epochs: int = 4
    minibatch_size: int = 256
    learning_rate: float = 2.0e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_ratio: float = 0.15
    target_kl: float = 0.012
    entropy_weight: float = 0.010
    value_weight: float = 0.5
    planner_imitation_start: float = 0.12
    planner_imitation_end_episode: int = 1200
    # The planner starts as a strong teacher, then becomes only a weak
    # tie-breaker.  The residual grows at the same time, so a trained policy
    # can genuinely select a better safe action during deterministic eval.
    planner_weight: float = 1.40
    planner_weight_final: float = 0.10
    planner_weight_decay_end_episode: int = 1200
    residual_logit_limit: float = 0.35
    residual_logit_limit_final: float = 1.00
    residual_logit_ramp_end_episode: int = 1200
    planner_warmup_episodes: int = 200
    planner_release_end_episode: int = 800
    max_grad_norm: float = 0.6
    max_consecutive_turns: int = 2
    revisit_penalty: float = 0.24
    repeat_visit_penalty: float = 0.06
    excess_turn_penalty: float = 0.35
    opposite_turn_penalty: float = 0.50
    seed: int = 7
    device: str = "auto"
    avoidable_revisit_penalty: float = 0.50
    avoidable_hazard_penalty: float = 0.80
    wall_bump_penalty: float = 0.50
    meat_conserve_health_norm: float = 0.80
    # The direct-action trainer is disabled; this value is retained only for
    # code compatibility and cannot create an old-style run directory.
    save_dir: str = "runs/15x15/spatial_target_ppo_with_planner/direct_action_experiment_removed"
    trace_every: int = 1
    eval_every: int = 100
    eval_episodes: int = 50


def choose_device(name: str) -> torch.device:
    requested = name.lower()
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            print("CUDA unavailable; using Apple MPS.", flush=True)
            return torch.device("mps")
        print("CUDA unavailable; using CPU.", flush=True)
        return torch.device("cpu")
    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("MPS unavailable; using CPU.", flush=True)
        return torch.device("cpu")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def build_env() -> Tuple[SensoryGridEnv, ObservationSwitches]:
    # New instance only. The recurrent DQN survival bonus is disabled for training because
    # it otherwise rewards turning in place rather than covering new cells.
    env = SensoryGridEnv(EnvConfig(survival_bonus=0.0))
    switches = ObservationSwitches(
        include_vision=True,
        include_temperature=False,
        include_smell=False,
        include_temperature_patch=True,
        include_smell_patch=True,
        include_visited_memory=True,
        include_hazard_memory=True,
    )
    return env, switches


def tensors(world_map: np.ndarray, local_sensor: np.ndarray, scalars: np.ndarray, device: torch.device):
    return (
        torch.from_numpy(world_map).unsqueeze(0).to(device),
        torch.from_numpy(local_sensor).unsqueeze(0).to(device),
        torch.from_numpy(scalars).unsqueeze(0).to(device),
    )


def shaped_reward(
    action: int,
    was_new: bool,
    prior_visits: int,
    new_known: int,
    info: Dict[str, object],
    terminated: bool,
    previous_action: int | None,
    decision_context: Dict[str, object],
    cfg: TrainConfig,
) -> float:
    """Incremental coverage objective; no high-variance terminal bonus."""
    energy_spent = (
        float(info.get("time_base_cost", 0.0))
        + float(info.get("forward_extra_cost", 0.0))
        + float(info.get("turn_extra_cost", 0.0))
        + float(info.get("thermal_extra_this_tick", 0.0))
    )
    reward = 1.0 if was_new else 0.0
    if action == ACTION_FORWARD and not was_new:
        # Re-entering an already visited tile is permissible only when it is a
        # bridge to an unvisited frontier. Repeating the same bridge is worse.
        reward -= cfg.revisit_penalty + cfg.repeat_visit_penalty * min(4, max(0, prior_visits - 1))
    if action != ACTION_FORWARD:
        reward -= 0.012
    excess_turns = max(0, int(info.get("consecutive_turn_steps", 0)) - cfg.max_consecutive_turns)
    reward -= cfg.excess_turn_penalty * excess_turns
    if previous_action in (1, 2) and action in (1, 2) and action != previous_action:
        reward -= cfg.opposite_turn_penalty
    if action == ACTION_FORWARD:
        forward_status = str(decision_context["forward_status"])
        has_safe_alternative = bool(decision_context["safe_unvisited_directions"])
        if forward_status == "wall":
            reward -= cfg.wall_bump_penalty
        elif forward_status == "visited" and has_safe_alternative:
            reward -= cfg.avoidable_revisit_penalty
        elif forward_status == "hazard" and has_safe_alternative:
            reward -= cfg.avoidable_hazard_penalty
    reward -= 0.010 * energy_spent
    reward += 0.018 * min(8, new_known)
    reward -= 0.30 * max(0.0, -float(info.get("health_delta", 0.0)))
    if terminated:
        reward -= 1.2
    return float(reward)


def policy_distribution(
    net: NeuralMapActorCritic,
    world_map: torch.Tensor,
    local_sensor: torch.Tensor,
    scalars: torch.Tensor,
    prior: torch.Tensor,
    action_mask: torch.Tensor,
    cfg: TrainConfig,
    planner_weight: float | torch.Tensor | None = None,
    residual_logit_limit: float | torch.Tensor | None = None,
):
    raw_logits, value = net(world_map, local_sensor, scalars)
    residual_limit = cfg.residual_logit_limit if residual_logit_limit is None else residual_logit_limit
    soft_planner_weight = cfg.planner_weight if planner_weight is None else planner_weight

    def action_scale(value: float | torch.Tensor) -> torch.Tensor:
        scale = torch.as_tensor(value, dtype=raw_logits.dtype, device=raw_logits.device)
        # A per-transition schedule arrives as [batch]; broadcast it across
        # the three action logits without accidentally broadcasting [batch]
        # against the action dimension.
        return scale.reshape(-1, 1) if scale.ndim > 0 else scale

    residual = action_scale(residual_limit) * torch.tanh(raw_logits)
    logits = residual + action_scale(soft_planner_weight) * prior + action_mask
    return Categorical(logits=logits), value, residual


def visible_safe_unvisited_directions(memory: CoverageMemory) -> List[str]:
    """Cardinal cells visible in the current 5x5 patch that a careful player can use."""
    directions = []
    for direction, name in enumerate(HEADING_NAMES):
        dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[direction]
        row, col = memory.pos[0] + dr, memory.pos[1] + dc
        if (
            0 <= row < memory.grid_size
            and 0 <= col < memory.grid_size
            and memory.visit_count[row, col] == 0
            and memory.map[HAZARD, row, col] < 0.5
        ):
            directions.append(name)
    return directions


def directed_turn_action(
    memory: CoverageMemory,
    planner_target: int,
    planner_action: int,
    mask: np.ndarray,
) -> int | None:
    """Choose the turn that most directly faces a visible frontier/planner goal."""
    target_row, target_col = divmod(int(planner_target), memory.grid_size)
    current_distance = abs(target_row - memory.pos[0]) + abs(target_col - memory.pos[1])
    candidates = []
    for action, direction_delta in ((1, -1), (2, 1)):
        if mask[action] < -1e8:
            continue
        direction = (memory.direction + direction_delta) % 4
        dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[direction]
        row, col = memory.pos[0] + dr, memory.pos[1] + dc
        safe_new = (
            0 <= row < memory.grid_size
            and 0 <= col < memory.grid_size
            and memory.visit_count[row, col] == 0
            and memory.map[HAZARD, row, col] < 0.5
        )
        next_distance = abs(target_row - row) + abs(target_col - col)
        # A safe adjacent frontier dominates; the global planner breaks ties.
        score = 10.0 * float(safe_new) + float(current_distance - next_distance) + 0.25 * float(action == planner_action)
        candidates.append((score, -action, action))
    return max(candidates)[2] if candidates else None


def planner_commit_probability(episode: int, cfg: TrainConfig) -> float:
    """Start with planner demonstrations, then gradually release the policy."""
    if episode <= 0 or episode > cfg.planner_release_end_episode:
        return 0.0
    if episode <= cfg.planner_warmup_episodes:
        return 1.0
    span = max(1, cfg.planner_release_end_episode - cfg.planner_warmup_episodes)
    return max(0.0, 1.0 - (episode - cfg.planner_warmup_episodes) / span)


def policy_guidance(episode: int, cfg: TrainConfig) -> Tuple[float, float]:
    """Return soft planner bias and neural residual range for one episode.

    Warm-up remains teacher-led, but the *logit* prior now fades after warm-up
    too.  This fixes the old invariant where a +1.4 planner logit could never
    be overturned by residual logits confined to [-0.35, +0.35].  Negative
    episode numbers are deterministic evaluation rollouts and intentionally
    use the released (final) guidance values.
    """
    if episode <= 0:
        return float(cfg.planner_weight_final), float(cfg.residual_logit_limit_final)

    def interpolate(start: float, end: float, end_episode: int) -> float:
        start_episode = int(cfg.planner_warmup_episodes)
        if episode <= start_episode:
            return float(start)
        if end_episode <= start_episode or episode >= end_episode:
            return float(end)
        fraction = (episode - start_episode) / float(end_episode - start_episode)
        return float(start + fraction * (end - start))

    planner_weight = interpolate(
        cfg.planner_weight,
        cfg.planner_weight_final,
        cfg.planner_weight_decay_end_episode,
    )
    residual_limit = interpolate(
        cfg.residual_logit_limit,
        cfg.residual_logit_limit_final,
        cfg.residual_logit_ramp_end_episode,
    )
    return planner_weight, residual_limit


def local_action_mask(
    memory: CoverageMemory,
    health_norm: float,
    turn_streak: int,
    previous_action: int | None,
    planner_target: int,
    planner_action: int,
    meat_mode: bool,
    planner_committed: bool,
    cfg: TrainConfig,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Mask actions that a player can reject from the current 5x5 observation.

    This uses only the agent's own map and current local patch: no hidden-grid
    access. It gives resource recovery priority over generic revisit avoidance:
    a planner-approved, known-safe bridge is valid when it leads to meat.
    """
    mask = np.zeros(3, dtype=np.float32)
    row, col = memory.forward_target()
    if not (0 <= row < memory.grid_size and 0 <= col < memory.grid_size):
        forward_status = "wall"
    elif memory.map[HAZARD, row, col] > 0.5:
        forward_status = "hazard"
    elif memory.visit_count[row, col] > 0:
        forward_status = "visited"
    else:
        forward_status = "unvisited"
    safe_directions = visible_safe_unvisited_directions(memory)
    current_heading = HEADING_NAMES[memory.direction]
    safe_nonforward_directions = [name for name in safe_directions if name != current_heading]
    forward_is_meat = (
        0 <= row < memory.grid_size
        and 0 <= col < memory.grid_size
        and memory.map[MEAT, row, col] > 0.5
    )
    conserve_meat = (
        not meat_mode
        and forward_is_meat
        and health_norm > cfg.meat_conserve_health_norm
        and bool(safe_nonforward_directions)
    )
    # A low-health agent may have to cross a visited corridor to reach meat.
    # The planner can only select this action from the agent's own map, and it
    # never selects a wall; hazards remain prohibited below.
    resource_bridge = meat_mode and planner_action == ACTION_FORWARD and forward_status in {"visited", "unvisited"}
    # If two prior turns leave the agent facing a known repeat while a local
    # frontier is visible, forcing forward creates a two-cell loop. Permit one
    # directed escape turn in this exceptional state instead.
    turn_escape = (
        turn_streak >= cfg.max_consecutive_turns
        and forward_status in {"visited", "hazard"}
        and bool(safe_nonforward_directions)
        and not resource_bridge
    )

    if turn_streak >= cfg.max_consecutive_turns and not turn_escape:
        mask[1:] = -1e9
    elif previous_action == 1 and not turn_escape:
        mask[2] = -1e9
    elif previous_action == 2 and not turn_escape:
        mask[1] = -1e9
    # After one turn, do not permit an immediate U-turn when the new heading
    # already faces a safe new tile. Meat recovery may legitimately need to
    # turn toward a known resource, so it takes priority over this shortcut.
    if (
        turn_streak == 1
        and forward_status == "unvisited"
        and not (meat_mode and planner_action != ACTION_FORWARD)
    ):
        mask[1:] = -1e9
    # A wall is never a useful forward action. Known hazards and repeats are
    # blocked only if the agent can visibly choose a safe new adjacent tile.
    # Meat recovery is allowed to traverse a safe repeated bridge.
    forward_blocked = False
    if conserve_meat:
        mask[ACTION_FORWARD] = -1e9
        forward_blocked = True
    elif forward_status == "wall":
        mask[ACTION_FORWARD] = -1e9
        forward_blocked = True
    elif not resource_bridge and forward_status in {"hazard", "visited"} and safe_nonforward_directions:
        mask[ACTION_FORWARD] = -1e9
        forward_blocked = True
    directed_turn = None
    if forward_blocked:
        directed_turn = directed_turn_action(memory, planner_target, planner_action, mask)
        if directed_turn is not None:
            # In the exact situation where a player would reject forward, do
            # not leave left vs. right to an untrained random policy.
            for action in (1, 2):
                if action != directed_turn:
                    mask[action] = -1e9
    elif turn_escape:
        # Forward is already a known repeat/hazard. Select the turn that faces
        # a visible frontier rather than deterministically bouncing backward.
        mask[ACTION_FORWARD] = -1e9
        directed_turn = directed_turn_action(memory, planner_target, planner_action, mask)
        if directed_turn is not None:
            for action in (1, 2):
                if action != directed_turn:
                    mask[action] = -1e9
    forced_action = None
    forced_reason = None
    if meat_mode:
        forced_action, forced_reason = planner_action, "meat_recovery"
    elif turn_escape and directed_turn is not None:
        forced_action, forced_reason = directed_turn, "turn_escape"
    elif forward_status == "unvisited" and not conserve_meat:
        # In coverage mode, a locally visible safe frontier is more valuable
        # than a distant warm-up target that begins by turning away from it.
        forced_action, forced_reason = ACTION_FORWARD, "safe_frontier_forward"
    elif planner_committed:
        forced_action, forced_reason = planner_action, "planner_warmup"
    if forced_action is not None:
        if mask[forced_action] < -1e8:
            forced_action = directed_turn_action(memory, planner_target, planner_action, mask)
            forced_reason = "safe_turn_fallback" if forced_action is not None else None
        if forced_action is not None:
            for action in range(3):
                if action != forced_action:
                    mask[action] = -1e9
    # Keep at least one legal action in an unexpected edge case.
    if np.all(mask < -1e8):
        mask[ACTION_FORWARD] = 0.0
    return mask, {
        "forward_status": forward_status,
        "safe_unvisited_directions": safe_directions,
        "turn_streak_before": int(turn_streak),
        "planner_action": int(planner_action),
        "planner_target_flat_index": int(planner_target),
        "directed_turn_action": directed_turn,
        "forward_blocked_by_local_rule": forward_blocked,
        "resource_bridge_override": resource_bridge,
        "conserve_forward_meat": conserve_meat,
        "turn_escape": turn_escape,
        "planner_committed": bool(planner_committed),
        "forced_action": forced_action,
        "forced_action_reason": forced_reason,
    }


def run_episode(
    net: NeuralMapActorCritic,
    env: SensoryGridEnv,
    switches: ObservationSwitches,
    device: torch.device,
    cfg: TrainConfig,
    episode: int,
    seed: int,
    stochastic: bool,
    trace_dir: Path | None = None,
    guidance_episode: int | None = None,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    planner = StaticFrontierPlanner()
    obs, _ = env.reset(seed=seed)
    memory = CoverageMemory(env.config.grid_size, env.config.patch_size, env.config.ambient_temperature_c)
    memory.reset(int(obs["direction"]))
    memory.update(obs)
    trace = EpisodeTrace(trace_dir, episode, seed, env, obs) if trace_dir is not None else None
    rows: List[Dict[str, object]] = []
    done = False
    forwards = repeats = 0
    health_loss = 0.0
    total_shaped_reward = 0.0
    previous_action: int | None = None
    # Rollout tools may use a positive trace label while explicitly requesting
    # the released inference guidance (episode 0).
    guidance_step = episode if guidance_episode is None else int(guidance_episode)

    while not done:
        world_map, local_sensor, scalars = memory.state(obs, env.config.max_steps - env.steps)
        prior, target, meat_mode = planner.action_prior(
            memory, float(obs["health_norm"]), float(obs["energy_norm"])
        )
        planner_action = int(np.argmax(prior))
        planner_expected_energy, planner_expected_thermal, planner_temperature_c = planner.action_energy_estimate(
            memory, planner_action
        )
        _, forward_expected_thermal, forward_temperature_c = planner.action_energy_estimate(memory, ACTION_FORWARD)
        commitment_probability = planner_commit_probability(guidance_step, cfg)
        soft_planner_weight, residual_limit = policy_guidance(guidance_step, cfg)
        planner_committed = stochastic and random.random() < commitment_probability
        action_mask, decision_context = local_action_mask(
            memory, float(obs["health_norm"]), int(obs["consecutive_turn_steps"]), previous_action,
            target, planner_action, meat_mode, planner_committed, cfg,
        )
        decision_context["resource_mode"] = "meat" if meat_mode else "coverage"
        decision_context["planner_commitment_probability"] = commitment_probability
        decision_context["planner_expected_action_energy"] = planner_expected_energy
        decision_context["planner_expected_thermal_energy"] = planner_expected_thermal
        decision_context["planner_target_temperature_c"] = planner_temperature_c
        decision_context["forward_expected_thermal_energy"] = forward_expected_thermal
        decision_context["forward_temperature_c"] = forward_temperature_c
        decision_context["soft_planner_weight"] = soft_planner_weight
        decision_context["residual_logit_limit"] = residual_limit
        map_t, local_t, scalar_t = tensors(world_map, local_sensor, scalars, device)
        prior_t = torch.from_numpy(prior).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(action_mask).unsqueeze(0).to(device)
        with torch.no_grad():
            dist, value, _ = policy_distribution(
                net, map_t, local_t, scalar_t, prior_t, mask_t, cfg,
                planner_weight=soft_planner_weight,
                residual_logit_limit=residual_limit,
            )
        action = int(dist.sample().item()) if stochastic else int(torch.argmax(dist.logits[0]).item())
        log_prob = float(dist.log_prob(torch.tensor(action, device=device)).item())
        was_new = memory.forward_is_new() if action == ACTION_FORWARD else False
        prior_visits = 0
        if action == ACTION_FORWARD:
            target_row, target_col = memory.forward_target()
            if 0 <= target_row < memory.grid_size and 0 <= target_col < memory.grid_size:
                prior_visits = int(memory.visit_count[target_row, target_col])
        position_before = user_coordinate(memory.pos[0], memory.pos[1], memory.grid_size)
        heading_before = memory.direction
        energy_before = float(obs["energy"])

        memory.advance(action)
        next_obs, environment_reward, terminated, truncated, info = env.step(action, switches)
        new_known = memory.update(next_obs)
        done = terminated or truncated
        reward = shaped_reward(
            action, was_new, prior_visits, new_known, info, terminated, previous_action, decision_context, cfg
        )
        position_after = user_coordinate(memory.pos[0], memory.pos[1], memory.grid_size)
        if trace is not None:
            trace.add(
                action, position_before, position_after, heading_before, memory.direction, energy_before,
                next_obs, environment_reward, info, was_new, prior_visits, int(obs["consecutive_turn_steps"]), previous_action,
                decision_context, action_mask, target,
            )

        rows.append({
            "map": world_map,
            "local_sensor": local_sensor,
            "scalars": scalars,
            "prior": prior,
            "action_mask": action_mask,
            "planner_action": planner_action,
            "soft_planner_weight": soft_planner_weight,
            "residual_logit_limit": residual_limit,
            "action": action,
            "log_prob": log_prob,
            "value": float(value.item()),
            "reward": reward,
            "done": float(done),
        })
        forwards += int(action == ACTION_FORWARD)
        repeats += int(action == ACTION_FORWARD and not was_new)
        health_loss += max(0.0, -float(info["health_delta"]))
        total_shaped_reward += reward
        obs = next_obs
        previous_action = action

    summary: Dict[str, float] = {
        "coverage": float(info["coverage"]),
        "steps": float(info["steps"]),
        "forward_actions": float(forwards),
        "repeat_forwards": float(repeats),
        "unique_per_forward": float((forwards - repeats) / max(1, forwards)),
        "health_loss": health_loss,
        "survived": float(not info["terminated"]),
        "shaped_reward": total_shaped_reward,
        "terminated": float(info["terminated"]),
    }
    if trace is not None:
        trace.write(summary)
    return rows, summary


def add_gae(rows: List[Dict[str, object]], cfg: TrainConfig) -> None:
    advantage = 0.0
    next_value = 0.0
    for row in reversed(rows):
        done = float(row["done"])
        value = float(row["value"])
        delta = float(row["reward"]) + cfg.gamma * next_value * (1.0 - done) - value
        advantage = delta + cfg.gamma * cfg.gae_lambda * (1.0 - done) * advantage
        row["advantage"] = advantage
        row["return"] = advantage + value
        next_value = value


def planner_imitation_weight(episode: int, cfg: TrainConfig) -> float:
    if episode >= cfg.planner_imitation_end_episode:
        return 0.0
    return cfg.planner_imitation_start * (1.0 - episode / max(1, cfg.planner_imitation_end_episode))


def update(
    net: NeuralMapActorCritic,
    optimizer: torch.optim.Optimizer,
    rows: List[Dict[str, object]],
    cfg: TrainConfig,
    device: torch.device,
    episode: int,
) -> Dict[str, float]:
    maps = torch.from_numpy(np.stack([row["map"] for row in rows])).to(device)
    locals_ = torch.from_numpy(np.stack([row["local_sensor"] for row in rows])).to(device)
    scalars = torch.from_numpy(np.stack([row["scalars"] for row in rows])).to(device)
    priors = torch.from_numpy(np.stack([row["prior"] for row in rows])).to(device)
    action_masks = torch.from_numpy(np.stack([row["action_mask"] for row in rows])).to(device)
    actions = torch.tensor([row["action"] for row in rows], device=device)
    planner_actions = torch.tensor([row["planner_action"] for row in rows], device=device)
    planner_weights = torch.tensor(
        [row["soft_planner_weight"] for row in rows], dtype=torch.float32, device=device
    )
    residual_limits = torch.tensor(
        [row["residual_logit_limit"] for row in rows], dtype=torch.float32, device=device
    )
    old_log_probs = torch.tensor([row["log_prob"] for row in rows], dtype=torch.float32, device=device)
    returns = torch.tensor([row["return"] for row in rows], dtype=torch.float32, device=device)
    advantages = torch.tensor([row["advantage"] for row in rows], dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
    indices = np.arange(len(rows))
    total = []
    stop_early = False
    imitation_weight = planner_imitation_weight(episode, cfg)

    for _ in range(cfg.ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), cfg.minibatch_size):
            idx = torch.as_tensor(indices[start:start + cfg.minibatch_size], device=device)
            dist, values, residual = policy_distribution(
                net, maps[idx], locals_[idx], scalars[idx], priors[idx], action_masks[idx], cfg,
                planner_weight=planner_weights[idx],
                residual_logit_limit=residual_limits[idx],
            )
            log_probs = dist.log_prob(actions[idx])
            log_ratio = log_probs - old_log_probs[idx]
            ratio = log_ratio.exp()
            unclipped = ratio * advantages[idx]
            clipped = ratio.clamp(1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * advantages[idx]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, returns[idx])
            imitation_loss = F.cross_entropy(residual, planner_actions[idx])
            entropy = dist.entropy().mean()
            loss = policy_loss + cfg.value_weight * value_loss + imitation_weight * imitation_loss - cfg.entropy_weight * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimizer.step()
            approx_kl = float((ratio - 1.0 - log_ratio).mean().item())
            total.append((float(loss.item()), float(policy_loss.item()), float(value_loss.item()), float(imitation_loss.item()), approx_kl))
            if approx_kl > cfg.target_kl:
                stop_early = True
                break
        if stop_early:
            break
    mean = np.mean(total, axis=0)
    return dict(zip(("loss", "policy_loss", "value_loss", "imitation_loss", "approx_kl"), mean.tolist()))


@torch.no_grad()
def evaluate(net: NeuralMapActorCritic, cfg: TrainConfig, device: torch.device, episodes: int, seed_base: int = 50_000) -> Dict[str, float]:
    env, switches = build_env()
    net.eval()
    summaries = [run_episode(net, env, switches, device, cfg, -(index + 1), seed_base + index, False)[1] for index in range(episodes)]
    net.train()
    metrics = {key: float(np.mean([summary[key] for summary in summaries])) for key in summaries[0]}
    metrics["coverage_p10"] = float(np.quantile([summary["coverage"] for summary in summaries], 0.10))
    return metrics


def write_metrics_header(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow([
            "episode", "coverage", "steps", "forward_actions", "repeat_forwards", "unique_per_forward",
            "health_loss", "survived", "shaped_reward", "loss", "policy_loss", "value_loss", "imitation_loss",
            "approx_kl", "planner_weight", "residual_logit_limit", "eval_coverage", "eval_coverage_p10",
            "eval_unique_per_forward", "eval_survived",
        ])


def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Traceable neural-map PPO; existing project files remain untouched")
    parser.add_argument("--episodes", type=int, default=defaults.episodes)
    parser.add_argument("--save_dir", default=defaults.save_dir)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--trace_every", type=int, default=defaults.trace_every)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(episodes=args.episodes, save_dir=args.save_dir, device=args.device, seed=args.seed, trace_every=max(1, args.trace_every))
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = choose_device(cfg.device)
    save_dir = Path(cfg.save_dir)
    trace_dir = save_dir / "traces"
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "train_config.txt").write_text("\n".join(f"{key}: {value}" for key, value in asdict(cfg).items()) + "\n", encoding="utf-8")
    metrics_path = save_dir / "episode_metrics.csv"
    write_metrics_header(metrics_path)

    env, switches = build_env()
    net = NeuralMapActorCritic().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate, eps=1e-5)
    best_coverage = -float("inf")
    pending: List[Dict[str, object]] = []

    for episode in range(1, cfg.episodes + 1):
        episode_trace_dir = trace_dir if episode % cfg.trace_every == 0 else None
        rows, summary = run_episode(net, env, switches, device, cfg, episode, cfg.seed + episode, True, episode_trace_dir)
        add_gae(rows, cfg)
        pending.extend(rows)
        if episode % cfg.rollout_episodes == 0 or episode == cfg.episodes:
            losses = update(net, optimizer, pending, cfg, device, episode)
            pending.clear()
        else:
            losses = {key: float("nan") for key in ("loss", "policy_loss", "value_loss", "imitation_loss", "approx_kl")}

        evaluation: Dict[str, float] = {}
        if episode == 1 or episode % cfg.eval_every == 0 or episode == cfg.episodes:
            evaluation = evaluate(net, cfg, device, cfg.eval_episodes)
            payload = {"model_state_dict": net.state_dict(), "train_config": asdict(cfg), "evaluation": evaluation, "episode": episode}
            torch.save(payload, save_dir / f"checkpoint_ep{episode:04d}.pt")
            if evaluation["coverage"] > best_coverage:
                best_coverage = evaluation["coverage"]
                torch.save(payload, save_dir / "best_coverage.pt")

        with metrics_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([
                episode, summary["coverage"], summary["steps"], summary["forward_actions"], summary["repeat_forwards"],
                summary["unique_per_forward"], summary["health_loss"], summary["survived"], summary["shaped_reward"],
                losses["loss"], losses["policy_loss"], losses["value_loss"], losses["imitation_loss"], losses["approx_kl"],
                policy_guidance(episode, cfg)[0], policy_guidance(episode, cfg)[1],
                evaluation.get("coverage", float("nan")), evaluation.get("coverage_p10", float("nan")),
                evaluation.get("unique_per_forward", float("nan")), evaluation.get("survived", float("nan")),
            ])
        print(
            f"episode={episode:04d} coverage={summary['coverage']:.3f} unique/fwd={summary['unique_per_forward']:.3f} "
            f"repeat={summary['repeat_forwards']:.0f} guide={policy_guidance(episode, cfg)[0]:.2f}/"
            f"{policy_guidance(episode, cfg)[1]:.2f} loss={losses['loss']:.4f} eval={evaluation}",
            flush=True,
        )
    torch.save({"model_state_dict": net.state_dict(), "train_config": asdict(cfg), "episode": cfg.episodes}, save_dir / "final.pt")


if __name__ == "__main__":
    raise SystemExit(
        "The legacy direct-action PPO trainer was removed. "
        "Run python models.model_based.ppo.spatial_target_ppo_with_planner.train instead."
    )
