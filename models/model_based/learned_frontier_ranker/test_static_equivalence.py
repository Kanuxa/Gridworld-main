"""Focused regression tests for V12 cross-environment adaptation."""

from __future__ import annotations

import gzip
import json
import unittest
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from models.shared.belief_map_tools.belief_map import BeliefMap
from models.shared.belief_map_tools.environment_profiles import environment_config_for_seed
from models.non_model_based.two_phase_belief_map_planner.run import run_episode as run_v11_episode
from models.model_based.learned_frontier_ranker.evaluate import AdaptiveExplorerConfig, run_episode
from gui.current_environment.sensory_grid_env import ACTION_FORWARD, ACTION_LEFT, EnvConfig, OBJ_FIRE, ObservationSwitches, SensoryGridEnv


def copied_observation(observation: dict) -> dict:
    return {key: value.copy() if isinstance(value, np.ndarray) else deepcopy(value) for key, value in observation.items()}


class AdaptiveV12Tests(unittest.TestCase):
    def test_static_defaults_are_seed_deterministic(self) -> None:
        first = SensoryGridEnv(EnvConfig())
        second = SensoryGridEnv(EnvConfig())
        first_observation, _ = first.reset(seed=37)
        second_observation, _ = second.reset(seed=37)
        np.testing.assert_array_equal(first_observation["vision"], second_observation["vision"])
        np.testing.assert_allclose(first_observation["temperature_patch_c"], second_observation["temperature_patch_c"])
        for action in (ACTION_FORWARD, ACTION_LEFT, ACTION_FORWARD):
            first_observation, first_reward, first_terminated, first_truncated, first_info = first.step(action, ObservationSwitches())
            second_observation, second_reward, second_terminated, second_truncated, second_info = second.step(action, ObservationSwitches())
            self.assertEqual(first_reward, second_reward)
            self.assertEqual(first_terminated, second_terminated)
            self.assertEqual(first_truncated, second_truncated)
            self.assertEqual(first_info["object_counts"], second_info["object_counts"])
            self.assertNotIn("did_move", first_info)
            self.assertNotIn("environment_changed", first_info)
            np.testing.assert_array_equal(first_observation["vision"], second_observation["vision"])

    def test_diverse_environment_profiles_are_seed_deterministic(self) -> None:
        variant, cfg = environment_config_for_seed(91, "diverse", max_steps=40)
        duplicate_variant, duplicate_cfg = environment_config_for_seed(91, "diverse", max_steps=40)
        other_variant, other_cfg = environment_config_for_seed(92, "diverse", max_steps=40)
        self.assertEqual(variant, duplicate_variant)
        self.assertEqual(asdict(cfg), asdict(duplicate_cfg))
        self.assertNotEqual(variant, other_variant)
        self.assertNotEqual(asdict(cfg), asdict(other_cfg))

        first = SensoryGridEnv(cfg)
        second = SensoryGridEnv(deepcopy(duplicate_cfg))
        first.reset(seed=91)
        second.reset(seed=91)
        for action in (ACTION_FORWARD, ACTION_LEFT, ACTION_FORWARD):
            _, _, _, _, first_info = first.step(action, ObservationSwitches())
            _, _, _, _, second_info = second.step(action, ObservationSwitches())
            self.assertEqual(first_info["object_counts"], second_info["object_counts"])
            np.testing.assert_array_equal(first.grid, second.grid)
            np.testing.assert_allclose(first.temperature_field_c, second.temperature_field_c)

    def test_belief_marks_visible_observation_change_without_hidden_state(self) -> None:
        env = SensoryGridEnv(EnvConfig())
        observation, _ = env.reset(seed=11)
        belief = BeliefMap(env.config.grid_size, env.config.patch_size)
        belief.reset(observation)
        changed = copied_observation(observation)
        centre = env.config.patch_size // 2
        changed["vision"][centre, centre] = OBJ_FIRE
        changed["temperature_patch_c"][centre, centre] += 2.0
        changed["smell_patch"][centre, centre] += 0.2
        belief.observe(changed)
        self.assertTrue(belief.last_delta.surprise)
        self.assertEqual(belief.last_delta.object_changes, 1)
        self.assertGreaterEqual(belief.last_delta.temperature_changes, 1)
        self.assertGreaterEqual(belief.last_delta.smell_changes, 1)
        self.assertEqual(belief.export_channels(1).shape, (11, 15, 15))
        self.assertEqual(belief.export_adaptive_channels(1).shape, (16, 15, 15))

    def test_failed_forward_keeps_belief_position(self) -> None:
        env = SensoryGridEnv(EnvConfig())
        observation, _ = env.reset(seed=5)
        belief = BeliefMap(env.config.grid_size, env.config.patch_size)
        belief.reset(observation)
        starting_position = belief.position
        belief.update_after_action(ACTION_FORWARD, observation, {"did_move": False})
        self.assertEqual(belief.position, starting_position)
        self.assertEqual(belief.steps, 1)

    def test_adaptive_runner_handles_short_varied_episode(self) -> None:
        variant, env_config = environment_config_for_seed(123, "diverse", max_steps=12)
        result = run_episode(
            123,
            env_config=env_config,
            controller_config=AdaptiveExplorerConfig(),
            environment_set="diverse",
            environment_variant=variant,
        )
        self.assertGreater(result.actions, 0)
        self.assertEqual(result.environment_set, "diverse")
        self.assertEqual(result.environment_variant, variant)
        self.assertEqual(result.direct_hazard_entries, 0)

    def test_no_checkpoint_matches_v11_on_same_stationary_world(self) -> None:
        original_step = SensoryGridEnv.step
        trace: list[int] = []

        def recorded_step(environment, action, *args, **kwargs):
            trace.append(int(action))
            return original_step(environment, action, *args, **kwargs)

        with patch.object(SensoryGridEnv, "step", recorded_step):
            expected = run_v11_episode(19, config=EnvConfig(max_steps=12))
            expected_trace = trace.copy()
            trace.clear()
            actual = run_episode(19, env_config=EnvConfig(max_steps=12))
            actual_trace = trace.copy()

        self.assertEqual(actual_trace, expected_trace)
        self.assertEqual(expected.phase1_replans, actual.information_replans)
        self.assertEqual(expected.phase2_replans, actual.coverage_replans)
        self.assertEqual(expected.actions - expected.turns, actual.forward_actions)
        for name in (
            "seen_fraction",
            "coverage",
            "phase",
            "actions",
            "turns",
            "replans",
            "direct_hazard_entries",
            "meat_collected",
            "health",
            "energy",
            "terminated",
            "truncated",
        ):
            self.assertEqual(getattr(actual, name), getattr(expected, name), name)

    def test_trace_contains_agent_observable_plan_step_and_endpoint_records(self) -> None:
        """A diagnostic trace is complete without changing the controller outcome."""
        baseline = run_episode(29, env_config=EnvConfig(max_steps=5))
        with TemporaryDirectory() as temporary_directory:
            result = run_episode(
                29,
                env_config=EnvConfig(max_steps=5),
                trace_dir=Path(temporary_directory),
                trace_observations=True,
            )
            trace_path = Path(temporary_directory) / result.trace_file
            self.assertTrue(trace_path.is_file())
            with gzip.open(trace_path, mode="rt", encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle]

        self.assertEqual(records[0]["event"], "episode_start")
        self.assertEqual(records[-1]["event"], "episode_end")
        self.assertEqual(records[-1]["end_reason"], result.end_reason)
        step_records = [record for record in records if record["event"] == "step"]
        plan_records = [record for record in records if record["event"] == "plan"]
        self.assertEqual(len(step_records), result.actions)
        self.assertGreaterEqual(len(plan_records), 1)
        self.assertIn("selected_plan", plan_records[0])
        self.assertIn("belief_before", step_records[0])
        self.assertIn("observation", step_records[0])
        for name in asdict(result):
            if name == "trace_file":
                continue
            self.assertEqual(getattr(result, name), getattr(baseline, name), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
