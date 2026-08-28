"""Stationary, seed-deterministic environment families for cross-world tests.

Each episode receives a fresh ``SensoryGridEnv`` with no exogenous changes
after ``reset``.  The original agent-induced consumption behavior remains
unchanged.  Profiles only construct different existing ``EnvConfig`` instances
outside the environment implementation, allowing an explorer to be trained and
evaluated on different layouts, hazard mixes, and thermal contexts without
changing ``sensory_grid_env_v5.py``.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

from gui.current_environment.sensory_grid_env import EnvConfig


Profile = tuple[str, dict[str, int | float]]


# Keep map and sensory dimensions fixed so V11/V12 trajectory and model
# contracts remain compatible.  The reset seed still controls the placement of
# every object within a selected profile.
ENVIRONMENT_SETS: dict[str, tuple[Profile, ...]] = {
    "standard": (
        ("standard", {}),
    ),
    "diverse": (
        ("balanced", {}),
        (
            "warm_hazard",
            {
                "n_fire": 3,
                "n_ice": 1,
                "n_glass": 2,
                "n_meat": 3,
                "n_flower": 1,
                "ambient_temperature_c": 21.5,
            },
        ),
        (
            "cold_hazard",
            {
                "n_fire": 1,
                "n_ice": 3,
                "n_glass": 2,
                "n_meat": 3,
                "n_flower": 2,
                "ambient_temperature_c": 22.5,
            },
        ),
        (
            "resource_dense",
            {
                "n_fire": 2,
                "n_ice": 1,
                "n_glass": 3,
                "n_meat": 4,
                "n_flower": 2,
                "ambient_temperature_c": 22.0,
            },
        ),
    ),
}


def environment_set_names() -> tuple[str, ...]:
    """Return valid CLI names in stable order."""
    return tuple(ENVIRONMENT_SETS)


def environment_config_for_seed(
    seed: int,
    environment_set: str,
    *,
    max_steps: int | None = None,
) -> tuple[str, EnvConfig]:
    """Choose one stationary profile deterministically and build its config."""
    try:
        profiles = ENVIRONMENT_SETS[environment_set]
    except KeyError as error:
        available = ", ".join(environment_set_names())
        raise ValueError(f"Unknown environment set {environment_set!r}; choose one of: {available}") from error
    profile_name, overrides = profiles[int(seed) % len(profiles)]
    base = EnvConfig()
    if max_steps is not None:
        base = replace(base, max_steps=int(max_steps))
    return profile_name, replace(base, **overrides)


def environment_profile_configs(environment_set: str, *, max_steps: int | None = None) -> dict[str, dict[str, Any]]:
    """Return every possible profile config for reproducible run metadata."""
    if environment_set not in ENVIRONMENT_SETS:
        available = ", ".join(environment_set_names())
        raise ValueError(f"Unknown environment set {environment_set!r}; choose one of: {available}")
    configs: dict[str, dict[str, Any]] = {}
    for profile_name, overrides in ENVIRONMENT_SETS[environment_set]:
        base = EnvConfig()
        if max_steps is not None:
            base = replace(base, max_steps=int(max_steps))
        configs[profile_name] = asdict(replace(base, **overrides))
    return configs
