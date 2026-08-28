"""Planner-anchored residual spatial-target PPO.

The neural network chooses an exploration target.  A persistent score map from
the standalone planner remains in the policy logits throughout training and
evaluation; the CNN learns a bounded residual rather than replacing it.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from gui.current_environment.sensory_grid_env import ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT, ObservationSwitches, SensoryGridEnv
from models.non_model_based.partial_observation_frontier_planner.run import BASELINE_PRESET, ENVIRONMENT_PRESETS, environment_config
from models.model_based.ppo.spatial_target_ppo_with_planner.memory import CoverageMemory, HAZARD, MEAT, user_coordinate
from models.model_based.ppo.spatial_target_ppo_with_planner.trace import EpisodeTrace
from models.non_model_based.partial_observation_frontier_planner.core import StaticFrontierPlanner
from models.model_based.ppo.planner_residual_target_ppo.model import PlannerResidualTargetActorCritic


MODEL_TYPE = "planner_residual_spatial_target_ppo"
LEGACY_MODEL_TYPE = "planner_residual_target_v8"


@dataclass
class PlannerResidualTrainConfig:
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
    # Planner target scores persist for the entire run and released rollout.
    planner_score_weight: float = 1.10
    planner_teacher_tie_bonus: float = 0.25
    # The residual starts too small to overpower a good planner, then gains
    # enough range to select a different target when PPO has evidence.
    residual_logit_limit_start: float = 0.15
    residual_logit_limit_final: float = 1.00
    residual_logit_ramp_end_episode: int = 1200
    # Imitation does not decay to zero: it is a guardrail against late drift.
    imitation_weight_start: float = 0.50
    imitation_weight_final: float = 0.15
    imitation_weight_decay_end_episode: int = 1000
    max_grad_norm: float = 0.6
    revisit_penalty: float = 0.30
    repeat_visit_penalty: float = 0.08
    turn_penalty: float = 0.015
    seed: int = 7
    device: str = "auto"
    save_dir: str = "runs/15x15/planner_residual_target_ppo"
    trace_every: int = 25
    eval_every: int = 100
    eval_episodes: int = 50
    environment_preset: str = BASELINE_PRESET


def choose_device(name: str) -> torch.device:
    requested = name.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            print("CUDA unavailable; using Apple MPS.", flush=True)
            return torch.device("mps")
        print("CUDA unavailable; using CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def build_env(preset: str = BASELINE_PRESET) -> Tuple[SensoryGridEnv, ObservationSwitches]:
    return SensoryGridEnv(environment_config(preset)), ObservationSwitches(
        include_vision=True,
        include_temperature=False,
        include_smell=False,
        include_temperature_patch=True,
        include_smell_patch=True,
        include_visited_memory=True,
        include_hazard_memory=True,
    )


def target_tensors(centred_map: np.ndarray, local_sensor: np.ndarray, scalars: np.ndarray, device: torch.device):
    return (
        torch.from_numpy(centred_map).unsqueeze(0).to(device),
        torch.from_numpy(local_sensor).unsqueeze(0).to(device),
        torch.from_numpy(scalars).unsqueeze(0).to(device),
    )


def candidate_cells(memory: CoverageMemory, meat_mode: bool) -> List[Tuple[int, int]]:
    cells = [
        (row, col)
        for row in range(memory.grid_size)
        for col in range(memory.grid_size)
        if memory.visit_count[row, col] == 0
        and memory.map[HAZARD, row, col] <= 0.5
        and (not meat_mode or memory.map[MEAT, row, col] > 0.5)
    ]
    return cells if cells or not meat_mode else candidate_cells(memory, False)


def flat_for_world(memory: CoverageMemory, row: int, col: int) -> int:
    size = 2 * memory.grid_size - 1
    ego_row, ego_col = memory.world_to_agent_frame(row, col)
    return int(ego_row * size + ego_col)


def target_mask_and_lookup(memory: CoverageMemory, candidates: Iterable[Tuple[int, int]]) -> Tuple[np.ndarray, Dict[int, Tuple[int, int]]]:
    size = 2 * memory.grid_size - 1
    mask = np.full(size * size, -1e9, dtype=np.float32)
    lookup: Dict[int, Tuple[int, int]] = {}
    for row, col in candidates:
        flat = flat_for_world(memory, row, col)
        mask[flat] = 0.0
        lookup[flat] = (row, col)
    if not lookup:
        flat = flat_for_world(memory, *memory.pos)
        mask[flat] = 0.0
        lookup[flat] = memory.pos
    return mask, lookup


def residual_limit(episode: int, cfg: PlannerResidualTrainConfig) -> float:
    if episode <= 0 or episode >= cfg.residual_logit_ramp_end_episode:
        return float(cfg.residual_logit_limit_final)
    fraction = episode / max(1, cfg.residual_logit_ramp_end_episode)
    return float(cfg.residual_logit_limit_start + fraction * (cfg.residual_logit_limit_final - cfg.residual_logit_limit_start))


def imitation_weight(episode: int, cfg: PlannerResidualTrainConfig) -> float:
    if episode <= 0 or episode >= cfg.imitation_weight_decay_end_episode:
        return float(cfg.imitation_weight_final)
    fraction = episode / max(1, cfg.imitation_weight_decay_end_episode)
    return float(cfg.imitation_weight_start + fraction * (cfg.imitation_weight_final - cfg.imitation_weight_start))


def planner_logits(
    memory: CoverageMemory,
    planner: StaticFrontierPlanner,
    lookup: Dict[int, Tuple[int, int]],
    meat_mode: bool,
    teacher_target: int,
    cfg: PlannerResidualTrainConfig,
) -> np.ndarray:
    """Project normalised planner scores into the neural policy's 29x29 frame."""
    size = 2 * memory.grid_size - 1
    output = np.zeros(size * size, dtype=np.float32)
    scores = planner.frontier_scores(memory, lookup.values(), meat_mode=meat_mode)
    valid = [(flat, scores[cell]) for flat, cell in lookup.items() if cell in scores]
    if valid:
        values = np.asarray([score for _, score in valid], dtype=np.float32)
        normalized = (values - values.mean()) / max(float(values.std()), 1e-4)
        normalized = np.clip(normalized, -3.0, 3.0)
        for (flat, _), score in zip(valid, normalized):
            output[flat] = cfg.planner_score_weight * score
    # Preserve the planner's deterministic tie-break as a small, permanent
    # preference.  The neural residual can still overcome this 0.25 bonus.
    output[int(teacher_target)] += cfg.planner_teacher_tie_bonus
    return output


def policy_distribution(
    net: PlannerResidualTargetActorCritic,
    centred_maps: torch.Tensor,
    local_sensors: torch.Tensor,
    scalars: torch.Tensor,
    target_masks: torch.Tensor,
    planner_score_maps: torch.Tensor,
    residual_limits: float | torch.Tensor,
):
    raw_logits, values = net(centred_maps, local_sensors, scalars)
    limit = torch.as_tensor(residual_limits, dtype=raw_logits.dtype, device=raw_logits.device)
    if limit.ndim:
        limit = limit.reshape(-1, 1)
    residual = limit * torch.tanh(raw_logits)
    return Categorical(logits=planner_score_maps + residual + target_masks), values, residual


def shaped_reward(action: int, was_new: bool, prior_visits: int, new_known: int, info: Dict[str, object], terminated: bool, cfg: PlannerResidualTrainConfig) -> float:
    energy_spent = sum(float(info.get(key, 0.0)) for key in ("time_base_cost", "forward_extra_cost", "turn_extra_cost", "thermal_extra_this_tick"))
    reward = 1.0 if was_new else 0.0
    if action == ACTION_FORWARD and not was_new:
        reward -= cfg.revisit_penalty + cfg.repeat_visit_penalty * min(4, max(0, prior_visits - 1))
    if action in (ACTION_LEFT, ACTION_RIGHT):
        reward -= cfg.turn_penalty
    reward -= 0.010 * energy_spent
    reward += 0.018 * min(8, new_known)
    reward -= 0.30 * max(0.0, -float(info.get("health_delta", 0.0)))
    if terminated:
        reward -= 1.2
    return float(reward)


def run_episode(
    net: PlannerResidualTargetActorCritic,
    env: SensoryGridEnv,
    switches: ObservationSwitches,
    device: torch.device,
    cfg: PlannerResidualTrainConfig,
    episode: int,
    seed: int,
    stochastic: bool,
    trace_dir: Path | None = None,
    schedule_episode: int | None = None,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    scheduled = episode if schedule_episode is None else int(schedule_episode)
    planner = StaticFrontierPlanner()
    obs, _ = env.reset(seed=seed)
    memory = CoverageMemory(env.config.grid_size, env.config.patch_size, env.config.ambient_temperature_c)
    memory.reset(int(obs["direction"]))
    memory.update(obs)
    trace = EpisodeTrace(trace_dir, episode, seed, env, obs) if trace_dir else None
    rows: List[Dict[str, object]] = []
    forwards = repeats = 0
    health_loss = total_reward = 0.0
    active_target: Tuple[int, int] | None = None
    active_row: Dict[str, object] | None = None
    active_reward, active_discount, active_steps = 0.0, 1.0, 0
    active_teacher_world = memory.pos
    active_teacher_target = flat_for_world(memory, *memory.pos)
    active_selected_target = active_teacher_target
    active_matched_teacher = True

    def finish_active(done: bool) -> None:
        nonlocal active_row
        if active_row is not None:
            active_row.update(reward=float(active_reward), done=float(done), discount=float(active_discount), duration=int(active_steps))
            rows.append(active_row)
            active_row = None

    while True:
        _, local_sensor, scalars = memory.state(obs, env.config.max_steps - env.steps)
        planner_prior, teacher_world_flat, meat_mode = planner.action_prior(memory, float(obs["health_norm"]), float(obs["energy_norm"]))
        target_invalid = active_target is None
        if active_target is not None:
            row, col = active_target
            target_invalid = memory.visit_count[row, col] > 0 or memory.map[HAZARD, row, col] > 0.5 or (meat_mode and memory.map[MEAT, row, col] <= 0.5)
        if target_invalid:
            finish_active(False)
            teacher_row, teacher_col = divmod(int(teacher_world_flat), memory.grid_size)
            mask, lookup = target_mask_and_lookup(memory, candidate_cells(memory, meat_mode))
            if (teacher_row, teacher_col) not in lookup.values():
                teacher_row, teacher_col = min(lookup.values(), key=lambda cell: abs(cell[0] - teacher_row) + abs(cell[1] - teacher_col))
            active_teacher_world = (teacher_row, teacher_col)
            active_teacher_target = flat_for_world(memory, teacher_row, teacher_col)
            score_map = planner_logits(memory, planner, lookup, meat_mode, active_teacher_target, cfg)
            centred_map = memory.agent_centred_map()
            map_t, local_t, scalar_t = target_tensors(centred_map, local_sensor, scalars, device)
            mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
            score_t = torch.from_numpy(score_map).unsqueeze(0).to(device)
            limit = residual_limit(scheduled, cfg)
            with torch.no_grad():
                dist, value, _ = policy_distribution(net, map_t, local_t, scalar_t, mask_t, score_t, limit)
            active_selected_target = int(dist.sample().item()) if stochastic else int(torch.argmax(dist.logits[0]).item())
            active_target = lookup[active_selected_target]
            active_matched_teacher = active_selected_target == active_teacher_target
            active_row = {
                "centred_map": centred_map,
                "local_sensor": local_sensor,
                "scalars": scalars,
                "target_mask": mask,
                "planner_score_map": score_map,
                "residual_limit": limit,
                "teacher_target": active_teacher_target,
                "selected_target": active_selected_target,
                "log_prob": float(dist.log_prob(torch.tensor(active_selected_target, device=device)).item()),
                "value": float(value.item()),
            }
            active_reward, active_discount, active_steps = 0.0, 1.0, 0

        assert active_target is not None
        selected_row, selected_col = active_target
        action, route_cost, route_energy = planner.action_toward(memory, selected_row, selected_col)
        if action is None:
            action = int(np.argmax(planner_prior))
        action = int(action)
        was_new = memory.forward_is_new() if action == ACTION_FORWARD else False
        prior_visits = 0
        if action == ACTION_FORWARD:
            row, col = memory.forward_target()
            if 0 <= row < memory.grid_size and 0 <= col < memory.grid_size:
                prior_visits = int(memory.visit_count[row, col])
        position_before, heading_before = user_coordinate(*memory.pos, memory.grid_size), memory.direction
        energy_before, turn_streak = float(obs["energy"]), int(obs["consecutive_turn_steps"])
        expected_energy, thermal_energy, temperature = planner.action_energy_estimate(memory, action)
        _, forward_thermal, forward_temperature = planner.action_energy_estimate(memory, ACTION_FORWARD)
        memory.advance(action)
        next_obs, environment_reward, terminated, truncated, info = env.step(action, switches)
        new_known = memory.update(next_obs)
        done = terminated or truncated
        reward = shaped_reward(action, was_new, prior_visits, new_known, info, terminated, cfg)
        if trace:
            context: Dict[str, object] = {
                "forward_status": "unvisited" if was_new else "visited",
                "safe_unvisited_directions": [],
                "planner_action": action,
                "resource_mode": "meat" if meat_mode else "coverage",
                "planner_committed": False,
                "planner_commitment_probability": 0.0,
                "soft_planner_weight": cfg.planner_score_weight,
                "residual_logit_limit": residual_limit(scheduled, cfg),
                "planner_expected_action_energy": expected_energy,
                "planner_expected_thermal_energy": thermal_energy,
                "planner_target_temperature_c": temperature,
                "forward_expected_thermal_energy": forward_thermal,
                "forward_temperature_c": forward_temperature,
                "directed_turn_action": None,
                "forward_blocked_by_local_rule": False,
                "resource_bridge_override": bool(meat_mode),
                "conserve_forward_meat": False,
                "thermal_turn_allowed": False,
                "turn_escape": False,
                "forced_action": action if meat_mode else None,
                "forced_action_reason": "meat_recovery" if meat_mode else None,
                "selected_target_flat_index": selected_row * memory.grid_size + selected_col,
                "teacher_target_flat_index": active_teacher_world[0] * memory.grid_size + active_teacher_world[1],
                "selected_target_coordinate": user_coordinate(selected_row, selected_col, memory.grid_size),
                "teacher_target_coordinate": user_coordinate(active_teacher_world[0], active_teacher_world[1], memory.grid_size),
                "target_matched_teacher": active_matched_teacher,
                "route_cost": route_cost,
                "route_energy": route_energy,
            }
            trace.add(action, position_before, user_coordinate(*memory.pos, memory.grid_size), heading_before, memory.direction, energy_before, next_obs, environment_reward, info, was_new, prior_visits, turn_streak, None, context, np.zeros(3, dtype=np.float32), selected_row * memory.grid_size + selected_col)
        active_reward += active_discount * reward
        active_discount *= cfg.gamma
        active_steps += 1
        forwards += int(action == ACTION_FORWARD)
        repeats += int(action == ACTION_FORWARD and not was_new)
        health_loss += max(0.0, -float(info["health_delta"]))
        total_reward += reward
        obs = next_obs
        if done:
            break

    finish_active(True)
    summary = {
        "coverage": float(info["coverage"]), "steps": float(info["steps"]), "forward_actions": float(forwards),
        "repeat_forwards": float(repeats), "unique_per_forward": float((forwards - repeats) / max(1, forwards)),
        "health_loss": health_loss, "survived": float(not info["terminated"]), "shaped_reward": total_reward,
        "terminated": float(info["terminated"]),
    }
    if trace:
        trace.write(summary)
    return rows, summary


def add_gae(rows: List[Dict[str, object]], cfg: PlannerResidualTrainConfig) -> None:
    advantage = next_value = 0.0
    for row in reversed(rows):
        done, value, discount, duration = float(row["done"]), float(row["value"]), float(row["discount"]), int(row["duration"])
        delta = float(row["reward"]) + discount * next_value * (1.0 - done) - value
        advantage = delta + discount * (cfg.gae_lambda ** duration) * (1.0 - done) * advantage
        row["advantage"], row["return"] = advantage, advantage + value
        next_value = value


def update(net: PlannerResidualTargetActorCritic, optimizer: torch.optim.Optimizer, rows: List[Dict[str, object]], cfg: PlannerResidualTrainConfig, device: torch.device, episode: int) -> Dict[str, float]:
    maps = torch.from_numpy(np.stack([row["centred_map"] for row in rows])).to(device)
    locals_ = torch.from_numpy(np.stack([row["local_sensor"] for row in rows])).to(device)
    scalars = torch.from_numpy(np.stack([row["scalars"] for row in rows])).to(device)
    masks = torch.from_numpy(np.stack([row["target_mask"] for row in rows])).to(device)
    planner_maps = torch.from_numpy(np.stack([row["planner_score_map"] for row in rows])).to(device)
    limits = torch.tensor([row["residual_limit"] for row in rows], dtype=torch.float32, device=device)
    teachers = torch.tensor([row["teacher_target"] for row in rows], dtype=torch.long, device=device)
    targets = torch.tensor([row["selected_target"] for row in rows], dtype=torch.long, device=device)
    old_log_probs = torch.tensor([row["log_prob"] for row in rows], dtype=torch.float32, device=device)
    returns = torch.tensor([row["return"] for row in rows], dtype=torch.float32, device=device)
    advantages = torch.tensor([row["advantage"] for row in rows], dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
    indices = np.arange(len(rows))
    metrics: List[Tuple[float, float, float, float, float]] = []
    stop = False
    imitation = imitation_weight(episode, cfg)
    for _ in range(cfg.ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), cfg.minibatch_size):
            idx = torch.as_tensor(indices[start:start + cfg.minibatch_size], device=device)
            dist, values, _ = policy_distribution(net, maps[idx], locals_[idx], scalars[idx], masks[idx], planner_maps[idx], limits[idx])
            log_probs = dist.log_prob(targets[idx])
            log_ratio = log_probs - old_log_probs[idx]
            ratio = log_ratio.exp()
            policy_loss = -torch.minimum(ratio * advantages[idx], ratio.clamp(1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * advantages[idx]).mean()
            value_loss = F.mse_loss(values, returns[idx])
            imitation_loss = F.cross_entropy(dist.logits, teachers[idx])
            loss = policy_loss + cfg.value_weight * value_loss + imitation * imitation_loss - cfg.entropy_weight * dist.entropy().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimizer.step()
            approx_kl = float((ratio - 1.0 - log_ratio).mean().item())
            metrics.append((float(loss.item()), float(policy_loss.item()), float(value_loss.item()), float(imitation_loss.item()), approx_kl))
            if approx_kl > cfg.target_kl:
                stop = True
                break
        if stop:
            break
    values = np.mean(metrics, axis=0)
    return dict(zip(("loss", "policy_loss", "value_loss", "imitation_loss", "approx_kl"), values.tolist()))


@torch.no_grad()
def evaluate(net: PlannerResidualTargetActorCritic, cfg: PlannerResidualTrainConfig, device: torch.device, episodes: int, seed_base: int = 50_000) -> Dict[str, float]:
    env, switches = build_env(cfg.environment_preset)
    net.eval()
    summaries = [run_episode(net, env, switches, device, cfg, -(index + 1), seed_base + index, False, schedule_episode=0)[1] for index in range(episodes)]
    net.train()
    result = {key: float(np.mean([summary[key] for summary in summaries])) for key in summaries[0]}
    result["coverage_p10"] = float(np.quantile([summary["coverage"] for summary in summaries], 0.10))
    return result


def parse_args() -> argparse.Namespace:
    defaults = PlannerResidualTrainConfig()
    parser = argparse.ArgumentParser(description="Planner-anchored residual spatial-target PPO")
    parser.add_argument("--episodes", type=int, default=defaults.episodes)
    parser.add_argument("--rollout-episodes", type=int, default=defaults.rollout_episodes)
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    parser.add_argument("--eval-episodes", type=int, default=defaults.eval_episodes)
    parser.add_argument("--save-dir", default=defaults.save_dir)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--trace-every", type=int, default=defaults.trace_every)
    parser.add_argument("--no-traces", action="store_true", help="Disable JSON episode trace files while retaining checkpoints, metrics, and terminal logs.")
    parser.add_argument("--preset", choices=ENVIRONMENT_PRESETS, default=defaults.environment_preset)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PlannerResidualTrainConfig(episodes=args.episodes, rollout_episodes=max(1, args.rollout_episodes), eval_every=max(1, args.eval_every), eval_episodes=max(1, args.eval_episodes), save_dir=args.save_dir, device=args.device, seed=args.seed, trace_every=0 if args.no_traces else max(1, args.trace_every), environment_preset=args.preset)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = choose_device(cfg.device)
    output = Path(cfg.save_dir)
    traces = output / "traces"
    output.mkdir(parents=True, exist_ok=True)
    (output / "train_config.txt").write_text("\n".join(f"{key}: {value}" for key, value in asdict(cfg).items()) + "\n", encoding="utf-8")
    metrics_path = output / "episode_metrics.csv"
    if not metrics_path.exists():
        metrics_path.write_text("episode,coverage,steps,forward_actions,repeat_forwards,unique_per_forward,health_loss,survived,shaped_reward,loss,policy_loss,value_loss,imitation_loss,approx_kl,residual_limit,imitation_weight,eval_coverage,eval_coverage_p10,eval_unique_per_forward,eval_survived\n", encoding="utf-8")
    env, switches = build_env(cfg.environment_preset)
    net = PlannerResidualTargetActorCritic().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate, eps=1e-5)
    best_coverage, pending = -float("inf"), []
    for episode in range(1, cfg.episodes + 1):
        trace_dir = traces if cfg.trace_every > 0 and episode % cfg.trace_every == 0 else None
        rows, summary = run_episode(net, env, switches, device, cfg, episode, cfg.seed + episode, True, trace_dir)
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
            payload = {"model_state_dict": net.state_dict(), "model_type": MODEL_TYPE, "model_arch": "planner_scores_plus_bounded_spatial_residual", "train_config": asdict(cfg), "env_config": asdict(env.config), "switches": asdict(switches), "evaluation": evaluation, "episode": episode}
            torch.save(payload, output / f"checkpoint_ep{episode:04d}.pt")
            if evaluation["coverage"] > best_coverage:
                best_coverage = evaluation["coverage"]
                torch.save(payload, output / "best_coverage.pt")
        with metrics_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([episode, summary["coverage"], summary["steps"], summary["forward_actions"], summary["repeat_forwards"], summary["unique_per_forward"], summary["health_loss"], summary["survived"], summary["shaped_reward"], losses["loss"], losses["policy_loss"], losses["value_loss"], losses["imitation_loss"], losses["approx_kl"], residual_limit(episode, cfg), imitation_weight(episode, cfg), evaluation.get("coverage", float("nan")), evaluation.get("coverage_p10", float("nan")), evaluation.get("unique_per_forward", float("nan")), evaluation.get("survived", float("nan"))])
        print(f"episode={episode:04d} coverage={summary['coverage']:.3f} unique/fwd={summary['unique_per_forward']:.3f} repeat={summary['repeat_forwards']:.0f} residual={residual_limit(episode, cfg):.2f} imit={imitation_weight(episode, cfg):.2f} loss={losses['loss']:.4f} eval={evaluation}", flush=True)
    torch.save({"model_state_dict": net.state_dict(), "model_type": MODEL_TYPE, "model_arch": "planner_scores_plus_bounded_spatial_residual", "train_config": asdict(cfg), "env_config": asdict(env.config), "switches": asdict(switches), "episode": cfg.episodes}, output / "final.pt")


if __name__ == "__main__":
    main()
