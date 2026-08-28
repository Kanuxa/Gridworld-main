"""Planner-executed spatial target PPO for maximum Gridworld coverage.

The policy chooses a target cell from the agent-owned map.  A thermal-aware
Dijkstra planner executes the first route action, then the policy replans after
the next 5x5 observation.  This keeps route safety structural while allowing a
neural model to learn exploration strategy.
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

from gui.current_environment.sensory_grid_env import ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT, EnvConfig, ObservationSwitches, SensoryGridEnv
from models.model_based.ppo.spatial_target_ppo_with_planner.memory import CoverageMemory, HAZARD, MEAT, StaticFrontierPlanner, user_coordinate
from models.model_based.ppo.spatial_target_ppo_with_planner.model import TargetMapActorCritic
from models.model_based.ppo.spatial_target_ppo_with_planner.trace import EpisodeTrace
from models.model_based.ppo.spatial_target_ppo_with_planner.direct_action_support import build_env, choose_device, visible_safe_unvisited_directions


MODEL_TYPE = "spatial_target_ppo_with_planner_executor"
LEGACY_MODEL_TYPE = "target_hierarchy_v1"


@dataclass
class TargetTrainConfig:
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
    imitation_start: float = 0.80
    imitation_end_episode: int = 1000
    # Teacher guidance exists only during curriculum.  It is exactly zero for
    # released evaluation and later PPO collection, unlike the old action bias.
    # There can be roughly 225 legal frontier targets.  A large initial value
    # keeps sampled target choices close to the teacher despite that action
    # count; it still decays to exactly zero before final evaluation.
    teacher_weight_start: float = 12.0
    teacher_weight_end_episode: int = 1000
    target_logit_limit: float = 1.50
    max_grad_norm: float = 0.6
    revisit_penalty: float = 0.30
    repeat_visit_penalty: float = 0.08
    turn_penalty: float = 0.015
    seed: int = 7
    device: str = "auto"
    save_dir: str = "runs/15x15/spatial_target_ppo_with_planner"
    trace_every: int = 25
    eval_every: int = 100
    eval_episodes: int = 50


def target_tensors(
    centred_map: np.ndarray,
    local_sensor: np.ndarray,
    scalars: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(centred_map).unsqueeze(0).to(device),
        torch.from_numpy(local_sensor).unsqueeze(0).to(device),
        torch.from_numpy(scalars).unsqueeze(0).to(device),
    )


def teacher_weight(episode: int, cfg: TargetTrainConfig) -> float:
    """Linearly remove planner target guidance after demonstrations."""
    if episode <= 0 or episode >= cfg.teacher_weight_end_episode:
        return 0.0
    return float(cfg.teacher_weight_start * (1.0 - episode / max(1, cfg.teacher_weight_end_episode)))


def imitation_weight(episode: int, cfg: TargetTrainConfig) -> float:
    if episode <= 0 or episode >= cfg.imitation_end_episode:
        return 0.0
    return float(cfg.imitation_start * (1.0 - episode / max(1, cfg.imitation_end_episode)))


def candidate_cells(memory: CoverageMemory, meat_mode: bool) -> List[Tuple[int, int]]:
    """Return safe, unvisited cells that the target policy may select."""
    candidates: List[Tuple[int, int]] = []
    for row in range(memory.grid_size):
        for col in range(memory.grid_size):
            if memory.visit_count[row, col] > 0 or memory.map[HAZARD, row, col] > 0.5:
                continue
            if meat_mode and memory.map[MEAT, row, col] <= 0.5:
                continue
            candidates.append((row, col))
    # If there is no observed unvisited meat, normal frontier targeting remains
    # safer than presenting an empty target distribution.
    if meat_mode and not candidates:
        return candidate_cells(memory, False)
    return candidates


def target_mask_and_lookup(
    memory: CoverageMemory,
    candidates: Iterable[Tuple[int, int]],
) -> Tuple[np.ndarray, Dict[int, Tuple[int, int]]]:
    size = 2 * memory.grid_size - 1
    mask = np.full(size * size, -1e9, dtype=np.float32)
    lookup: Dict[int, Tuple[int, int]] = {}
    for row, col in candidates:
        ego_row, ego_col = memory.world_to_agent_frame(row, col)
        flat = int(ego_row * size + ego_col)
        mask[flat] = 0.0
        lookup[flat] = (row, col)
    if not lookup:
        # This should occur only after all cells are known/visited.  A legal
        # repeated move avoids an invalid categorical distribution.
        row, col = memory.pos
        ego_row, ego_col = memory.world_to_agent_frame(row, col)
        flat = int(ego_row * size + ego_col)
        mask[flat] = 0.0
        lookup[flat] = (row, col)
    return mask, lookup


def flat_for_world(memory: CoverageMemory, row: int, col: int) -> int:
    size = 2 * memory.grid_size - 1
    ego_row, ego_col = memory.world_to_agent_frame(row, col)
    return int(ego_row * size + ego_col)


def target_policy_distribution(
    net: TargetMapActorCritic,
    centred_maps: torch.Tensor,
    local_sensors: torch.Tensor,
    scalars: torch.Tensor,
    target_masks: torch.Tensor,
    teacher_targets: torch.Tensor,
    teacher_weights: float | torch.Tensor,
    cfg: TargetTrainConfig,
):
    raw_logits, value = net(centred_maps, local_sensors, scalars)
    residual = cfg.target_logit_limit * torch.tanh(raw_logits)
    prior = torch.zeros_like(residual).scatter_(1, teacher_targets.reshape(-1, 1), 1.0)
    weight = torch.as_tensor(teacher_weights, dtype=residual.dtype, device=residual.device)
    if weight.ndim > 0:
        weight = weight.reshape(-1, 1)
    logits = residual + weight * prior + target_masks
    return Categorical(logits=logits), value, residual


def forward_status(memory: CoverageMemory) -> str:
    row, col = memory.forward_target()
    if not (0 <= row < memory.grid_size and 0 <= col < memory.grid_size):
        return "wall"
    if memory.map[HAZARD, row, col] > 0.5:
        return "hazard"
    if memory.visit_count[row, col] > 0:
        return "visited"
    return "unvisited"


def shaped_reward(
    action: int,
    was_new: bool,
    prior_visits: int,
    new_known: int,
    info: Dict[str, object],
    terminated: bool,
    cfg: TargetTrainConfig,
) -> float:
    energy_spent = sum(
        float(info.get(key, 0.0))
        for key in ("time_base_cost", "forward_extra_cost", "turn_extra_cost", "thermal_extra_this_tick")
    )
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
    net: TargetMapActorCritic,
    env: SensoryGridEnv,
    switches: ObservationSwitches,
    device: torch.device,
    cfg: TargetTrainConfig,
    episode: int,
    seed: int,
    stochastic: bool,
    trace_dir: Path | None = None,
    guidance_episode: int | None = None,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    guidance_step = episode if guidance_episode is None else int(guidance_episode)
    planner = StaticFrontierPlanner()
    obs, _ = env.reset(seed=seed)
    memory = CoverageMemory(env.config.grid_size, env.config.patch_size, env.config.ambient_temperature_c)
    memory.reset(int(obs["direction"]))
    memory.update(obs)
    trace = EpisodeTrace(trace_dir, episode, seed, env, obs) if trace_dir is not None else None
    rows: List[Dict[str, object]] = []
    done = False
    forwards = repeats = 0
    health_loss = total_reward = 0.0
    previous_action: int | None = None
    active_target: Tuple[int, int] | None = None
    active_teacher_target = 0
    active_teacher_world = (0, 0)
    active_selected_target = 0
    active_target_matched_teacher = False
    active_teacher_weight = 0.0
    active_row: Dict[str, object] | None = None
    active_reward = 0.0
    active_discount = 1.0
    active_steps = 0

    def finish_active(done_flag: bool) -> None:
        nonlocal active_row
        if active_row is None:
            return
        active_row["reward"] = float(active_reward)
        active_row["done"] = float(done_flag)
        active_row["discount"] = float(active_discount)
        active_row["duration"] = int(active_steps)
        rows.append(active_row)
        active_row = None

    while not done:
        _, local_sensor, scalars = memory.state(obs, env.config.max_steps - env.steps)
        planner_prior, teacher_world_flat, meat_mode = planner.action_prior(
            memory, float(obs["health_norm"]), float(obs["energy_norm"])
        )
        target_invalid = active_target is None
        if active_target is not None:
            target_row, target_col = active_target
            target_invalid = (
                memory.visit_count[target_row, target_col] > 0
                or memory.map[HAZARD, target_row, target_col] > 0.5
                or (meat_mode and memory.map[MEAT, target_row, target_col] <= 0.5)
            )

        if target_invalid:
            # A target is held until reached/invalid.  We only ask the neural
            # policy again at a strategic decision boundary; Dijkstra still
            # replans the *route* on every intervening 5x5 observation.
            finish_active(False)
            teacher_row, teacher_col = divmod(int(teacher_world_flat), memory.grid_size)
            target_mask, lookup = target_mask_and_lookup(memory, candidate_cells(memory, meat_mode))
            if (teacher_row, teacher_col) not in lookup.values():
                teacher_row, teacher_col = min(
                    lookup.values(),
                    key=lambda cell: abs(cell[0] - teacher_row) + abs(cell[1] - teacher_col),
                )
            active_teacher_target = flat_for_world(memory, teacher_row, teacher_col)
            active_teacher_world = (teacher_row, teacher_col)
            active_teacher_weight = teacher_weight(guidance_step, cfg)
            centred_map = memory.agent_centred_map()
            map_t, local_t, scalar_t = target_tensors(centred_map, local_sensor, scalars, device)
            mask_t = torch.from_numpy(target_mask).unsqueeze(0).to(device)
            teacher_t = torch.tensor([active_teacher_target], dtype=torch.long, device=device)
            with torch.no_grad():
                dist, value, _ = target_policy_distribution(
                    net, map_t, local_t, scalar_t, mask_t, teacher_t, active_teacher_weight, cfg
                )
            active_selected_target = int(dist.sample().item()) if stochastic else int(torch.argmax(dist.logits[0]).item())
            active_target = lookup[active_selected_target]
            active_target_matched_teacher = active_selected_target == active_teacher_target
            active_row = {
                "centred_map": centred_map,
                "local_sensor": local_sensor,
                "scalars": scalars,
                "target_mask": target_mask,
                "teacher_target": active_teacher_target,
                "teacher_weight": active_teacher_weight,
                "selected_target": active_selected_target,
                "log_prob": float(dist.log_prob(torch.tensor(active_selected_target, device=device)).item()),
                "value": float(value.item()),
            }
            active_reward = 0.0
            active_discount = 1.0
            active_steps = 0

        assert active_target is not None
        selected_row, selected_col = active_target
        action, route_cost, route_energy = planner.action_toward(memory, selected_row, selected_col)
        if action is None:
            action = int(np.argmax(planner_prior))
        action = int(action)

        was_new = memory.forward_is_new() if action == ACTION_FORWARD else False
        forward_status_before = forward_status(memory)
        safe_directions_before = visible_safe_unvisited_directions(memory)
        planner_energy, planner_thermal, planner_temperature = planner.action_energy_estimate(memory, action)
        _, forward_thermal, forward_temperature = planner.action_energy_estimate(memory, ACTION_FORWARD)
        prior_visits = 0
        if action == ACTION_FORWARD:
            target_row, target_col = memory.forward_target()
            if 0 <= target_row < memory.grid_size and 0 <= target_col < memory.grid_size:
                prior_visits = int(memory.visit_count[target_row, target_col])
        position_before = user_coordinate(memory.pos[0], memory.pos[1], memory.grid_size)
        heading_before = memory.direction
        energy_before = float(obs["energy"])
        before_turns = int(obs["consecutive_turn_steps"])

        memory.advance(action)
        next_obs, environment_reward, terminated, truncated, info = env.step(action, switches)
        new_known = memory.update(next_obs)
        done = terminated or truncated
        reward = shaped_reward(action, was_new, prior_visits, new_known, info, terminated, cfg)
        position_after = user_coordinate(memory.pos[0], memory.pos[1], memory.grid_size)

        decision_context: Dict[str, object] = {
            "forward_status": forward_status_before,
            "safe_unvisited_directions": safe_directions_before,
            "planner_action": action,
            "resource_mode": "meat" if meat_mode else "coverage",
            "planner_committed": False,
            "planner_commitment_probability": 0.0,
            "soft_planner_weight": active_teacher_weight,
            "residual_logit_limit": cfg.target_logit_limit,
            "planner_expected_action_energy": planner_energy,
            "planner_expected_thermal_energy": planner_thermal,
            "planner_target_temperature_c": planner_temperature,
            "forward_expected_thermal_energy": forward_thermal,
            "forward_temperature_c": forward_temperature,
            "directed_turn_action": None,
            "forward_blocked_by_local_rule": False,
            "resource_bridge_override": False,
            "conserve_forward_meat": False,
            "thermal_turn_allowed": False,
            "turn_escape": False,
            "forced_action": action if meat_mode else None,
            "forced_action_reason": "meat_recovery" if meat_mode else None,
            "selected_target_flat_index": selected_row * memory.grid_size + selected_col,
            "teacher_target_flat_index": active_teacher_world[0] * memory.grid_size + active_teacher_world[1],
            "selected_target_coordinate": user_coordinate(selected_row, selected_col, memory.grid_size),
            "teacher_target_coordinate": user_coordinate(active_teacher_world[0], active_teacher_world[1], memory.grid_size),
            "target_matched_teacher": active_target_matched_teacher,
            "route_cost": route_cost,
            "route_energy": route_energy,
        }
        if trace is not None:
            trace.add(
                action, position_before, position_after, heading_before, memory.direction, energy_before,
                next_obs, environment_reward, info, was_new, prior_visits, before_turns, previous_action,
                decision_context, np.zeros(3, dtype=np.float32), selected_row * memory.grid_size + selected_col,
            )

        active_reward += active_discount * reward
        active_discount *= cfg.gamma
        active_steps += 1
        forwards += int(action == ACTION_FORWARD)
        repeats += int(action == ACTION_FORWARD and not was_new)
        health_loss += max(0.0, -float(info["health_delta"]))
        total_reward += reward
        obs = next_obs
        previous_action = action

    finish_active(True)

    summary: Dict[str, float] = {
        "coverage": float(info["coverage"]),
        "steps": float(info["steps"]),
        "forward_actions": float(forwards),
        "repeat_forwards": float(repeats),
        "unique_per_forward": float((forwards - repeats) / max(1, forwards)),
        "health_loss": health_loss,
        "survived": float(not info["terminated"]),
        "shaped_reward": total_reward,
        "terminated": float(info["terminated"]),
    }
    if trace is not None:
        trace.write(summary)
    return rows, summary


def add_gae(rows: List[Dict[str, object]], cfg: TargetTrainConfig) -> None:
    advantage = next_value = 0.0
    for row in reversed(rows):
        done = float(row["done"])
        value = float(row["value"])
        discount = float(row["discount"])
        duration = int(row["duration"])
        delta = float(row["reward"]) + discount * next_value * (1.0 - done) - value
        advantage = delta + discount * (cfg.gae_lambda ** duration) * (1.0 - done) * advantage
        row["advantage"] = advantage
        row["return"] = advantage + value
        next_value = value


def update(
    net: TargetMapActorCritic,
    optimizer: torch.optim.Optimizer,
    rows: List[Dict[str, object]],
    cfg: TargetTrainConfig,
    device: torch.device,
    episode: int,
) -> Dict[str, float]:
    maps = torch.from_numpy(np.stack([row["centred_map"] for row in rows])).to(device)
    locals_ = torch.from_numpy(np.stack([row["local_sensor"] for row in rows])).to(device)
    scalars = torch.from_numpy(np.stack([row["scalars"] for row in rows])).to(device)
    masks = torch.from_numpy(np.stack([row["target_mask"] for row in rows])).to(device)
    teacher_targets = torch.tensor([row["teacher_target"] for row in rows], dtype=torch.long, device=device)
    teacher_weights = torch.tensor([row["teacher_weight"] for row in rows], dtype=torch.float32, device=device)
    targets = torch.tensor([row["selected_target"] for row in rows], dtype=torch.long, device=device)
    old_log_probs = torch.tensor([row["log_prob"] for row in rows], dtype=torch.float32, device=device)
    returns = torch.tensor([row["return"] for row in rows], dtype=torch.float32, device=device)
    advantages = torch.tensor([row["advantage"] for row in rows], dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
    indices = np.arange(len(rows))
    total: List[Tuple[float, float, float, float, float]] = []
    stop_early = False
    current_imitation = imitation_weight(episode, cfg)

    for _ in range(cfg.ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), cfg.minibatch_size):
            idx = torch.as_tensor(indices[start:start + cfg.minibatch_size], device=device)
            dist, values, residual = target_policy_distribution(
                net, maps[idx], locals_[idx], scalars[idx], masks[idx], teacher_targets[idx], teacher_weights[idx], cfg
            )
            log_probs = dist.log_prob(targets[idx])
            log_ratio = log_probs - old_log_probs[idx]
            ratio = log_ratio.exp()
            policy_loss = -torch.minimum(
                ratio * advantages[idx],
                ratio.clamp(1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * advantages[idx],
            ).mean()
            value_loss = F.mse_loss(values, returns[idx])
            # Imitate only among legal target cells.  Penalising all 841 frame
            # positions would waste capacity on padded/out-of-world locations.
            imitation_loss = F.cross_entropy(residual + masks[idx], teacher_targets[idx])
            entropy = dist.entropy().mean()
            loss = policy_loss + cfg.value_weight * value_loss + current_imitation * imitation_loss - cfg.entropy_weight * entropy
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
def evaluate(
    net: TargetMapActorCritic,
    cfg: TargetTrainConfig,
    device: torch.device,
    episodes: int,
    seed_base: int = 50_000,
) -> Dict[str, float]:
    env, switches = build_env()
    net.eval()
    summaries = [
        run_episode(net, env, switches, device, cfg, -(index + 1), seed_base + index, False, guidance_episode=0)[1]
        for index in range(episodes)
    ]
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
            "approx_kl", "teacher_weight", "imitation_weight", "eval_coverage", "eval_coverage_p10",
            "eval_unique_per_forward", "eval_survived",
        ])


def parse_args() -> argparse.Namespace:
    defaults = TargetTrainConfig()
    parser = argparse.ArgumentParser(description="Spatial target PPO with Dijkstra route execution")
    parser.add_argument("--episodes", type=int, default=defaults.episodes)
    parser.add_argument("--save_dir", default=defaults.save_dir)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--trace_every", type=int, default=defaults.trace_every)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TargetTrainConfig(
        episodes=args.episodes,
        save_dir=args.save_dir,
        device=args.device,
        seed=args.seed,
        trace_every=max(1, args.trace_every),
    )
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = choose_device(cfg.device)
    save_dir = Path(cfg.save_dir)
    trace_dir = save_dir / "traces"
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "train_config.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in asdict(cfg).items()) + "\n", encoding="utf-8"
    )
    metrics_path = save_dir / "episode_metrics.csv"
    write_metrics_header(metrics_path)

    env, switches = build_env()
    net = TargetMapActorCritic().to(device)
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
            payload = {
                "model_state_dict": net.state_dict(),
                "model_arch": "agent_centred_spatial_target_ppo_with_dijkstra_executor",
                "model_type": MODEL_TYPE,
                "train_config": asdict(cfg),
                "env_config": asdict(env.config),
                "switches": asdict(switches),
                "evaluation": evaluation,
                "episode": episode,
            }
            torch.save(payload, save_dir / f"checkpoint_ep{episode:04d}.pt")
            if evaluation["coverage"] > best_coverage:
                best_coverage = evaluation["coverage"]
                torch.save(payload, save_dir / "best_coverage.pt")

        with metrics_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([
                episode, summary["coverage"], summary["steps"], summary["forward_actions"], summary["repeat_forwards"],
                summary["unique_per_forward"], summary["health_loss"], summary["survived"], summary["shaped_reward"],
                losses["loss"], losses["policy_loss"], losses["value_loss"], losses["imitation_loss"], losses["approx_kl"],
                teacher_weight(episode, cfg), imitation_weight(episode, cfg),
                evaluation.get("coverage", float("nan")), evaluation.get("coverage_p10", float("nan")),
                evaluation.get("unique_per_forward", float("nan")), evaluation.get("survived", float("nan")),
            ])
        print(
            f"episode={episode:04d} coverage={summary['coverage']:.3f} unique/fwd={summary['unique_per_forward']:.3f} "
            f"repeat={summary['repeat_forwards']:.0f} teacher={teacher_weight(episode, cfg):.2f} "
            f"imit={imitation_weight(episode, cfg):.2f} loss={losses['loss']:.4f} eval={evaluation}",
            flush=True,
        )
    torch.save({
        "model_state_dict": net.state_dict(),
        "model_arch": "agent_centred_spatial_target_ppo_with_dijkstra_executor",
        "model_type": MODEL_TYPE,
        "train_config": asdict(cfg),
        "env_config": asdict(env.config),
        "switches": asdict(switches),
        "episode": cfg.episodes,
    }, save_dir / "final.pt")


if __name__ == "__main__":
    main()
