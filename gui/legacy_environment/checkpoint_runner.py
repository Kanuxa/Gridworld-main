
from __future__ import annotations

from dataclasses import asdict
import json
from typing import Dict

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from gui.legacy_environment.sensory_grid_env import (
    ACTION_FORWARD,
    ACTION_LEFT,
    ACTION_RIGHT,
    DIR_SYMBOLS,
    OBJ_EMPTY,
    OBJ_FIRE,
    OBJ_FLOWER,
    OBJ_GLASS,
    OBJ_ICE,
    OBJ_MEAT,
    OBJ_SYMBOLS,
    EnvConfig,
    ObservationSwitches,
    SensoryGridEnv,
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.0rem;
            padding-bottom: 1.2rem;
            max-width: 1480px;
        }
        .app-title {
            font-size: 2.0rem;
            font-weight: 760;
            letter-spacing: -0.02em;
            margin-bottom: 0.18rem;
        }
        .app-subtitle {
            color: #64748b;
            margin-bottom: 0.9rem;
        }
        .card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid rgba(148,163,184,0.24);
            border-radius: 18px;
            padding: 0.95rem 1rem 0.85rem 1rem;
            box-shadow: 0 8px 24px rgba(15,23,42,0.06);
            margin-bottom: 1rem;
            overflow-x: auto;
        }
        .card-title {
            font-size: 1.02rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }
        .tiny-note {
            font-size: 0.84rem;
            color: #64748b;
        }
        .world-grid {
            display: grid;
            gap: 4px;
            justify-content: start;
            margin-top: 0.45rem;
        }
        .cell {
            border-radius: 9px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            border: 1px solid rgba(148,163,184,0.18);
            background: #f1f5f9;
            color: #334155;
            box-sizing: border-box;
        }
        .visited {
            background: rgba(191,219,254,0.75);
            box-shadow: inset 0 0 0 2px rgba(59,130,246,0.22);
        }
        .obj-fire {
            background: linear-gradient(180deg, #fb923c 0%, #ef4444 100%);
            color: white;
        }
        .obj-meat {
            background: linear-gradient(180deg, #4ade80 0%, #16a34a 100%);
            color: white;
        }
        .obj-flower {
            background: linear-gradient(180deg, #f9a8d4 0%, #ec4899 100%);
            color: white;
        }
        .obj-glass {
            background: linear-gradient(180deg, #7dd3fc 0%, #0ea5e9 100%);
            color: white;
        }
        .obj-ice {
            background: linear-gradient(180deg, #dbeafe 0%, #60a5fa 100%);
            color: #0f172a;
        }
        .agent {
            background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
            color: white;
            border: 2px solid #93c5fd;
        }
        .fog-unseen {
            background: #0f172a;
            color: transparent;
            border: 1px solid rgba(148,163,184,0.08);
            box-shadow: inset 0 0 0 999px rgba(2, 6, 23, 0.78);
        }
        .fog-memory {
            box-shadow: inset 0 0 0 999px rgba(15, 23, 42, 0.30);
        }
        .metric-box {
            background: #ffffff;
            border: 1px solid rgba(148,163,184,0.22);
            border-radius: 16px;
            padding: 0.78rem 0.9rem;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
            margin-bottom: 0.75rem;
        }
        .metric-label {
            font-size: 0.82rem;
            color: #64748b;
            margin-bottom: 0.12rem;
        }
        .metric-value {
            font-size: 1.4rem;
            font-weight: 750;
            color: #0f172a;
        }
        .metric-subtext {
            font-size: 0.78rem;
            color: #64748b;
            margin-top: 0.18rem;
        }
        .bar-track {
            width: 100%;
            height: 12px;
            border-radius: 999px;
            background: #e2e8f0;
            overflow: hidden;
            margin-top: 0.42rem;
        }
        .bar-fill-temp {
            height: 100%;
            background: linear-gradient(90deg, #93c5fd 0%, #f87171 100%);
            border-radius: 999px;
        }
        .bar-fill-smell {
            height: 100%;
            background: linear-gradient(90deg, #c4b5fd 0%, #7c3aed 100%);
            border-radius: 999px;
        }
        .bar-fill-health {
            height: 100%;
            background: linear-gradient(90deg, #4ade80 0%, #16a34a 100%);
            border-radius: 999px;
        }
        .bar-fill-energy {
            height: 100%;
            background: linear-gradient(90deg, #60a5fa 0%, #2563eb 100%);
            border-radius: 999px;
        }
        .bar-fill-neutral {
            height: 100%;
            background: linear-gradient(90deg, #cbd5e1 0%, #64748b 100%);
            border-radius: 999px;
        }
        .legend-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 0.55rem;
        }
        .legend-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border-radius: 999px;
            padding: 0.36rem 0.68rem;
            border: 1px solid rgba(148,163,184,0.22);
            background: white;
            font-size: 0.9rem;
        }
        .legend-swatch {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
        }
        .chip-row {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 0.45rem;
        }
        .kv-chip {
            display: inline-flex;
            gap: 6px;
            align-items: center;
            border-radius: 999px;
            padding: 0.32rem 0.65rem;
            border: 1px solid rgba(148,163,184,0.25);
            background: white;
            font-size: 0.83rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_box(label: str, value: str, subtext: str | None = None) -> None:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {f'<div class="metric-subtext">{subtext}</div>' if subtext else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_bar(label: str, value: float, max_value: float, kind: str, text_value: str, subtext: str | None = None) -> None:
    width = 0.0 if max_value <= 0 else max(0.0, min(value / max_value, 1.0)) * 100.0
    fill_class = {
        "temp": "bar-fill-temp",
        "smell": "bar-fill-smell",
        "health": "bar-fill-health",
        "energy": "bar-fill-energy",
        "neutral": "bar-fill-neutral",
    }[kind]
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{text_value}</div>
            <div class="bar-track"><div class="{fill_class}" style="width:{width:.1f}%"></div></div>
            {f'<div class="metric-subtext">{subtext}</div>' if subtext else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def world_cell_size(grid_size: int, patch_mode: bool = False) -> int:
    if patch_mode:
        return {3: 52, 5: 44, 7: 36}.get(grid_size, 34)
    return max(22, min(36, int(520 / max(1, grid_size))))


def current_vision_mask(shape: tuple[int, int], agent_pos: tuple[int, int], patch_size: int) -> np.ndarray:
    rows, cols = shape
    mask = np.zeros((rows, cols), dtype=bool)
    radius = patch_size // 2
    r0 = max(0, agent_pos[0] - radius)
    r1 = min(rows, agent_pos[0] + radius + 1)
    c0 = max(0, agent_pos[1] - radius)
    c1 = min(cols, agent_pos[1] + radius + 1)
    mask[r0:r1, c0:c1] = True
    return mask


def temperature_outline_style(temp_c: float, cfg: EnvConfig) -> str:
    if cfg.comfort_low_c <= temp_c <= cfg.comfort_high_c:
        return "box-shadow: inset 0 0 0 1px rgba(148,163,184,0.16);"
    if temp_c < cfg.comfort_low_c:
        frac = min((cfg.comfort_low_c - temp_c) / max(1e-6, cfg.discomfort_temp_scale_c), 1.0)
        return f"box-shadow: inset 0 0 0 2px rgba(59,130,246,{0.20 + 0.70 * frac:.3f});"
    frac = min((temp_c - cfg.comfort_high_c) / max(1e-6, cfg.discomfort_temp_scale_c), 1.0)
    return f"box-shadow: inset 0 0 0 2px rgba(239,68,68,{0.20 + 0.70 * frac:.3f});"


def grid_html(
    world: np.ndarray,
    visited: np.ndarray,
    temp_field: np.ndarray,
    cfg: EnvConfig,
    agent_pos,
    direction,
    reveal_objects: bool,
    patch_mode: bool = False,
    fog_enabled: bool = False,
    visible_mask: np.ndarray | None = None,
) -> str:
    rows, cols = world.shape
    cell_size = world_cell_size(rows if not patch_mode else cols, patch_mode=patch_mode)
    font_size = max(12, int(cell_size * 0.55))
    if visible_mask is None:
        visible_mask = np.ones((rows, cols), dtype=bool)
    html = [f'<div class="world-grid" style="grid-template-columns: repeat({cols}, {cell_size}px);">']
    cls_map = {
        OBJ_FIRE: "obj-fire",
        OBJ_MEAT: "obj-meat",
        OBJ_FLOWER: "obj-flower",
        OBJ_GLASS: "obj-glass",
        OBJ_ICE: "obj-ice",
    }
    for r in range(rows):
        for c in range(cols):
            currently_visible = bool(visible_mask[r, c])
            explored = bool(visited[r, c] > 0.5)
            classes = ["cell"]
            text = ""

            if fog_enabled and not currently_visible and not explored:
                classes.append("fog-unseen")
                outline = ""
                html.append(
                    f'<div class="{" ".join(classes)}" style="width:{cell_size}px;height:{cell_size}px;font-size:{font_size}px;{outline}">{text}</div>'
                )
                continue

            if explored:
                classes.append("visited")
            if (r, c) == agent_pos:
                classes.append("agent")
                text = DIR_SYMBOLS[direction]
            else:
                obj = int(world[r, c])
                if reveal_objects and obj != OBJ_EMPTY:
                    classes.append(cls_map[obj])
                    text = OBJ_SYMBOLS[obj]
            if fog_enabled and not currently_visible and explored:
                classes.append("fog-memory")
            outline = temperature_outline_style(float(temp_field[r, c]), cfg)
            html.append(
                f'<div class="{" ".join(classes)}" style="width:{cell_size}px;height:{cell_size}px;font-size:{font_size}px;{outline}">{text}</div>'
            )
    html.append("</div>")
    return "".join(html)



def rotate_patch_for_display(patch: np.ndarray, direction: int) -> np.ndarray:
    return np.rot90(np.asarray(patch), k=(-int(direction)) % 4).copy()


def patch_html(patch: np.ndarray, active: bool, title: str, direction: int) -> None:
    st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
    if not active:
        st.caption("This channel is visible here, but it is currently disabled in the agent input.")
    rotated_patch = rotate_patch_for_display(patch, direction)
    visited = np.zeros_like(rotated_patch, dtype=np.float32)
    temp_field = np.full_like(rotated_patch, 22.0, dtype=np.float32)
    agent_pos = (rotated_patch.shape[0] // 2, rotated_patch.shape[1] // 2)
    dummy_cfg = EnvConfig(grid_size=int(rotated_patch.shape[0]))
    st.markdown(
        grid_html(rotated_patch, visited, temp_field, dummy_cfg, agent_pos, direction, reveal_objects=True, patch_mode=True),
        unsafe_allow_html=True,
    )
    st.caption("Patch is rotated into world orientation. The center arrow matches the world-grid direction.")


def float_patch_html(patch: np.ndarray, title: str, mode: str, active: bool, cfg: EnvConfig, direction: int) -> None:
    st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
    if not active:
        st.caption("This channel is visible here, but it is currently disabled in the agent input.")
    rotated_patch = rotate_patch_for_display(patch, direction)
    rows, cols = rotated_patch.shape
    cell_size = world_cell_size(rows, patch_mode=True)
    font_size = max(11, int(cell_size * 0.28))
    html = [f'<div class="world-grid" style="grid-template-columns: repeat({cols}, {cell_size}px);">']
    center = rows // 2
    smell_max = max(1.0, float(np.max(rotated_patch)))
    temp_span = max(0.1, max(cfg.fire_temp_delta_amp, cfg.ice_temp_delta_amp))
    for r in range(rows):
        for c in range(cols):
            val = float(rotated_patch[r, c])
            if mode == "visited":
                a = max(0.08, min(0.92, 0.10 + 0.80 * max(0.0, min(val, 1.0))))
                bg = f"rgba(59,130,246,{a:.3f})"
                text = f"{val:.1f}"
            elif mode == "hazard":
                a = max(0.08, min(0.92, 0.10 + 0.80 * max(0.0, min(val, 1.0))))
                bg = f"rgba(239,68,68,{a:.3f})"
                text = f"{val:.1f}"
            elif mode == "smell":
                frac = max(0.0, min(val / smell_max, 1.0))
                a = 0.08 + 0.84 * frac
                bg = f"rgba(124,58,237,{a:.3f})"
                text = f"{val:.2f}"
            elif mode == "temperature":
                if cfg.comfort_low_c <= val <= cfg.comfort_high_c:
                    bg = "rgba(226,232,240,0.65)"
                elif val < cfg.comfort_low_c:
                    frac = min((cfg.comfort_low_c - val) / max(1e-6, temp_span), 1.0)
                    bg = f"rgba(59,130,246,{0.12 + 0.78 * frac:.3f})"
                else:
                    frac = min((val - cfg.comfort_high_c) / max(1e-6, temp_span), 1.0)
                    bg = f"rgba(239,68,68,{0.12 + 0.78 * frac:.3f})"
                text = f"{val:.1f}"
            else:
                bg = "rgba(226,232,240,0.8)"
                text = f"{val:.2f}"
            if (r, c) == (center, center):
                text = DIR_SYMBOLS[direction]
                extra = "border: 2px solid rgba(15,23,42,0.7); box-shadow: inset 0 0 0 999px rgba(255,255,255,0.08);"
            else:
                extra = ""
            html.append(
                f'<div class="cell" style="width:{cell_size}px;height:{cell_size}px;background:{bg};font-size:{font_size}px;{extra}">{text}</div>'
            )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)
    st.caption("Patch is rotated into world orientation. The center arrow matches the world-grid direction.")


def world_direction_sequence(current_direction: int, target_direction: int) -> list[int]:
    current_direction = int(current_direction) % 4
    target_direction = int(target_direction) % 4
    diff = (target_direction - current_direction) % 4
    if diff == 0:
        return [ACTION_FORWARD]
    if diff == 1:
        return [ACTION_RIGHT, ACTION_FORWARD]
    if diff == 3:
        return [ACTION_LEFT, ACTION_FORWARD]
    return [ACTION_RIGHT, ACTION_RIGHT, ACTION_FORWARD]


def sequence_name(seq: list[int]) -> str:
    mapping = {ACTION_FORWARD: 'forward', ACTION_LEFT: 'left', ACTION_RIGHT: 'right'}
    return ' -> '.join(mapping[a] for a in seq)


def execute_action_sequence(actions: list[int], world_label: str) -> None:
    env = st.session_state.env
    if env.terminated or env.truncated:
        return
    total_reward = 0.0
    info = None
    for action in actions:
        if env.terminated or env.truncated:
            break
        _, reward, terminated, truncated, info = env.step(action, st.session_state.switches)
        total_reward += float(reward)
        if terminated or truncated:
            break
    if info is None:
        return
    st.session_state.event_log.insert(
        0,
        f"step {info['steps']:03d} | move {world_label} | seq {sequence_name(actions)} | total reward {total_reward:+.3f} | health {info['health']} | energy {info['energy']:.2f} | {info['last_event']}",
    )
    st.session_state.event_log = st.session_state.event_log[:20]
    if env.terminated or env.truncated:
        st.session_state.event_log.insert(0, "Episode ended. Actions are disabled until reset.")


def do_world_move(target_direction: int, label: str) -> None:
    seq = world_direction_sequence(st.session_state.env.direction, target_direction)
    execute_action_sequence(seq, label)


def render_keyboard_controls() -> None:
    meta_html = (
        '<div id="keyboard-world-move-meta" '
        'data-w="Move up" '
        'data-a="Move left" '
        'data-s="Move down" '
        'data-d="Move right" '
        'style="display:none"></div>'
    )
    st.markdown(meta_html, unsafe_allow_html=True)
    components.html(
        """
        <script>
        (function () {
            const parentWindow = window.parent;
            const parentDoc = parentWindow.document;
            if (!parentDoc) {
                return;
            }
            if (!parentWindow.__sensoryGridWasdWorldHandlerInstalled) {
                parentWindow.__sensoryGridWasdWorldHandlerInstalled = true;
                parentWindow.__sensoryGridWasdWorldHandler = function (event) {
                    const tag = (event.target && event.target.tagName ? event.target.tagName : '').toLowerCase();
                    if (tag === 'input' || tag === 'textarea' || tag === 'select') {
                        return;
                    }
                    const key = (event.key || '').toLowerCase();
                    if (!['w', 'a', 's', 'd'].includes(key)) {
                        return;
                    }
                    const meta = parentDoc.getElementById('keyboard-world-move-meta');
                    if (!meta) {
                        return;
                    }
                    const label = meta.dataset[key];
                    if (!label) {
                        return;
                    }
                    const buttons = Array.from(parentDoc.querySelectorAll('button'));
                    const target = buttons.find((button) => ((button.innerText || button.textContent || '').trim() === label));
                    if (target && !target.disabled) {
                        event.preventDefault();
                        target.click();
                    }
                };
                parentDoc.addEventListener('keydown', parentWindow.__sensoryGridWasdWorldHandler, true);
            }
        })();
        </script>
        """,
        height=0,
        width=0,
    )

def reward_table(reward_terms: Dict[str, float]) -> None:
    if not reward_terms:
        st.caption("No reward breakdown yet. Reset and take a step.")
        return
    data = {
        "term": list(reward_terms.keys()),
        "value": [float(v) for v in reward_terms.values()],
    }
    st.dataframe(data, use_container_width=True, hide_index=True)


def object_legend() -> None:
    st.markdown(
        """
        <div class="legend-row">
            <div class="legend-pill"><span class="legend-swatch" style="background:#ef4444"></span> Hot tiles outlined red</div>
            <div class="legend-pill"><span class="legend-swatch" style="background:#60a5fa"></span> Cold tiles outlined blue</div>
            <div class="legend-pill"><span class="legend-swatch" style="background:#ef4444"></span> Fire</div>
            <div class="legend-pill"><span class="legend-swatch" style="background:#60a5fa"></span> Ice</div>
            <div class="legend-pill"><span class="legend-swatch" style="background:#16a34a"></span> Meat</div>
            <div class="legend-pill"><span class="legend-swatch" style="background:#ec4899"></span> Flower</div>
            <div class="legend-pill"><span class="legend-swatch" style="background:#0ea5e9"></span> Glass</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_default_switches() -> ObservationSwitches:
    return ObservationSwitches(
        include_vision=True,
        include_temperature=False,
        include_smell=False,
        include_temperature_patch=True,
        include_smell_patch=True,
        include_visited_memory=True,
        include_hazard_memory=True,
    )


def init_state() -> None:
    if "config" not in st.session_state:
        st.session_state.config = EnvConfig()
    if "switches" not in st.session_state:
        st.session_state.switches = build_default_switches()
    if "env" not in st.session_state:
        env = SensoryGridEnv(st.session_state.config)
        env.reset(seed=None)
        st.session_state.env = env
    if "event_log" not in st.session_state:
        st.session_state.event_log = ["Episode reset."]
    if "fog_of_war" not in st.session_state:
        st.session_state.fog_of_war = False


def reset_environment(seed=None) -> None:
    env = SensoryGridEnv(st.session_state.config)
    env.reset(seed=seed)
    st.session_state.env = env
    st.session_state.event_log = ["Episode reset."]



def do_action(action: int) -> None:
    execute_action_sequence([action], {ACTION_FORWARD: 'forward', ACTION_LEFT: 'turn left', ACTION_RIGHT: 'turn right'}[action])


def sidebar_controls() -> None:
    st.sidebar.header("Environment")
    config = st.session_state.config
    switches = st.session_state.switches

    grid_size = st.sidebar.slider("Grid size", 9, 21, config.grid_size, step=2)
    patch_size = st.sidebar.selectbox("Patch size", [3, 5, 7], index=[3, 5, 7].index(config.patch_size))
    init_health = st.sidebar.slider("Initial health", 4, 20, config.init_health)
    init_energy = st.sidebar.slider("Initial energy", 4, 20, int(round(config.init_energy)))
    max_steps = st.sidebar.slider("Max steps", 30, 250, config.max_steps, step=10)

    st.sidebar.subheader("Objects")
    n_fire = st.sidebar.slider("Fire count", 0, 6, config.n_fire)
    n_ice = st.sidebar.slider("Ice count", 0, 6, config.n_ice)
    n_meat = st.sidebar.slider("Meat count", 0, 6, config.n_meat)
    n_flower = st.sidebar.slider("Flower count", 0, 6, config.n_flower)
    n_glass = st.sidebar.slider("Glass count", 0, 6, config.n_glass)

    st.sidebar.subheader("Contact outcomes")
    fire_damage = st.sidebar.slider("Fire damage", 1, 5, config.fire_damage)
    ice_damage = st.sidebar.slider("Ice damage", 1, 5, config.ice_damage)
    glass_damage = st.sidebar.slider("Glass damage", 1, 5, config.glass_damage)
    meat_heal = st.sidebar.slider("Meat heal", 1, 5, config.meat_heal)

    st.sidebar.subheader("Temperature field")
    ambient_temperature_c = st.sidebar.slider("Ambient temperature °C", -5.0, 35.0, float(config.ambient_temperature_c), step=0.5)
    comfort_low_c = st.sidebar.slider("Comfort lower bound °C", -5.0, 35.0, float(config.comfort_low_c), step=0.5)
    comfort_high_c = st.sidebar.slider("Comfort upper bound °C", -5.0, 35.0, float(config.comfort_high_c), step=0.5)
    fire_temp_delta_amp = st.sidebar.slider("Fire temperature amplitude °C", 1.0, 20.0, float(config.fire_temp_delta_amp), step=0.5)
    ice_temp_delta_amp = st.sidebar.slider("Ice temperature amplitude °C", 1.0, 20.0, float(config.ice_temp_delta_amp), step=0.5)
    temp_sigma = st.sidebar.slider("Temperature sigma", 0.8, 5.0, float(config.temp_sigma), step=0.1)

    st.sidebar.subheader("Smell field")
    meat_smell_amp = st.sidebar.slider("Meat smell amplitude", 0.0, 3.0, float(config.meat_smell_amp), step=0.05)
    flower_smell_amp = st.sidebar.slider("Flower smell amplitude", 0.0, 3.0, float(config.flower_smell_amp), step=0.05)
    smell_sigma_meat = st.sidebar.slider("Meat smell sigma", 0.8, 5.0, float(config.smell_sigma_meat), step=0.1)
    smell_sigma_flower = st.sidebar.slider("Flower smell sigma", 0.8, 5.0, float(config.smell_sigma_flower), step=0.1)

    st.sidebar.subheader("Energy model")
    time_energy_cost = st.sidebar.slider("Time base cost", 0.0, 1.0, float(config.time_energy_cost), step=0.05)
    turn_energy_cost = st.sidebar.slider("Turn extra cost", 0.0, 1.0, float(config.turn_energy_cost), step=0.05)
    forward_energy_cost = st.sidebar.slider("Forward extra cost", 0.0, 2.0, float(config.forward_energy_cost), step=0.05)
    thermal_extra_energy_max = st.sidebar.slider("Max thermal extra energy", 0.0, 1.5, float(config.thermal_extra_energy_max), step=0.05)

    st.sidebar.subheader("Reward shaping")
    step_penalty = st.sidebar.slider("Step penalty", -0.20, 0.0, float(config.step_penalty), step=0.005)
    explore_reward = st.sidebar.slider("Explore reward", 0.0, 0.30, float(config.explore_reward), step=0.005)
    wall_penalty = st.sidebar.slider("Wall penalty", -0.30, 0.0, float(config.wall_penalty), step=0.005)
    damage_reward_scale = st.sidebar.slider("Damage reward scale", 0.0, 0.50, float(config.damage_reward_scale), step=0.01)
    heal_reward_scale = st.sidebar.slider("Heal reward scale", 0.0, 0.30, float(config.heal_reward_scale), step=0.01)
    flower_penalty = st.sidebar.slider("Flower penalty", -0.20, 0.0, float(config.flower_penalty), step=0.005)
    death_penalty = st.sidebar.slider("Death penalty", 0.0, 3.0, float(config.death_penalty), step=0.05)
    energy_reward_scale = st.sidebar.slider("Energy reward scale", 0.0, 0.10, float(config.energy_reward_scale), step=0.0025)
    discomfort_reward_scale = st.sidebar.slider("Discomfort reward scale", 0.0, 0.10, float(config.discomfort_reward_scale), step=0.0025)
    no_move_penalty = st.sidebar.slider("No-move streak penalty", 0.0, 0.10, float(config.no_move_penalty), step=0.0025)
    turn_streak_penalty = st.sidebar.slider("Turn streak penalty", 0.0, 0.05, float(config.turn_streak_penalty), step=0.001)

    st.sidebar.subheader("Agent input switches")
    include_vision = st.sidebar.toggle("Include vision patch", value=switches.include_vision)
    include_temperature = st.sidebar.toggle("Include temperature scalar", value=switches.include_temperature)
    include_smell = st.sidebar.toggle("Include smell scalar", value=switches.include_smell)
    include_temperature_patch = st.sidebar.toggle("Include temperature egocentric patch", value=switches.include_temperature_patch)
    include_smell_patch = st.sidebar.toggle("Include smell egocentric patch", value=switches.include_smell_patch)
    include_visited = st.sidebar.toggle("Include visited memory patch", value=switches.include_visited_memory)
    include_hazard = st.sidebar.toggle("Include hazard memory patch", value=switches.include_hazard_memory)

    st.session_state.switches = ObservationSwitches(
        include_vision=include_vision,
        include_temperature=include_temperature,
        include_smell=include_smell,
        include_temperature_patch=include_temperature_patch,
        include_smell_patch=include_smell_patch,
        include_visited_memory=include_visited,
        include_hazard_memory=include_hazard,
    )

    comfort_low_c, comfort_high_c = sorted((comfort_low_c, comfort_high_c))
    st.sidebar.subheader("World map display")
    st.session_state.fog_of_war = st.sidebar.toggle(
        "Fog of war",
        value=bool(st.session_state.fog_of_war),
        help="Only the current vision patch stays fully visible. Explored tiles remain visible under a dim memory cover.",
    )

    updated_config = EnvConfig(
        grid_size=grid_size,
        patch_size=patch_size,
        init_health=init_health,
        max_health=init_health,
        init_energy=float(init_energy),
        max_energy=float(init_energy),
        max_steps=max_steps,
        n_fire=n_fire,
        n_ice=n_ice,
        n_meat=n_meat,
        n_flower=n_flower,
        n_glass=n_glass,
        step_penalty=step_penalty,
        explore_reward=explore_reward,
        wall_penalty=wall_penalty,
        damage_reward_scale=damage_reward_scale,
        heal_reward_scale=heal_reward_scale,
        flower_penalty=flower_penalty,
        death_penalty=death_penalty,
        glass_damage=glass_damage,
        meat_heal=meat_heal,
        fire_damage=fire_damage,
        ice_damage=ice_damage,
        ambient_temperature_c=ambient_temperature_c,
        comfort_low_c=comfort_low_c,
        comfort_high_c=comfort_high_c,
        fire_temp_delta_amp=fire_temp_delta_amp,
        ice_temp_delta_amp=ice_temp_delta_amp,
        temp_sigma=temp_sigma,
        meat_smell_amp=meat_smell_amp,
        flower_smell_amp=flower_smell_amp,
        smell_sigma_meat=smell_sigma_meat,
        smell_sigma_flower=smell_sigma_flower,
        time_energy_cost=time_energy_cost,
        forward_energy_cost=forward_energy_cost,
        turn_energy_cost=turn_energy_cost,
        thermal_extra_energy_max=thermal_extra_energy_max,
        energy_reward_scale=energy_reward_scale,
        discomfort_reward_scale=discomfort_reward_scale,
        no_move_penalty=no_move_penalty,
        turn_streak_penalty=turn_streak_penalty,
    )

    col_a, col_b = st.sidebar.columns(2)
    if col_a.button("Apply + reset", use_container_width=True):
        st.session_state.config = updated_config
        reset_environment(seed=None)
    if col_b.button("Reset episode", use_container_width=True):
        st.session_state.config = updated_config
        reset_environment(seed=None)

    st.sidebar.subheader("Controls")
    disabled = st.session_state.env.terminated or st.session_state.env.truncated
    row1a, row1b = st.sidebar.columns(2)
    if row1a.button("Move up", use_container_width=True, disabled=disabled):
        do_world_move(0, "up")
    if row1b.button("Move right", use_container_width=True, disabled=disabled):
        do_world_move(1, "right")
    row2a, row2b = st.sidebar.columns(2)
    if row2a.button("Move left", use_container_width=True, disabled=disabled):
        do_world_move(3, "left")
    if row2b.button("Move down", use_container_width=True, disabled=disabled):
        do_world_move(2, "down")
    st.sidebar.caption("W/A/S/D move in world coordinates. The GUI internally applies the needed turn(s) plus forward, so cost and step count match the environment.")

    st.sidebar.caption("Training baseline now uses vision patch, temperature egocentric patch, smell egocentric patch, visited memory patch, and hazard memory patch. Scalar temperature and scalar smell are off by default.")
    st.sidebar.code(json.dumps(asdict(st.session_state.switches), indent=2), language="json")


def main() -> None:
    st.set_page_config(page_title="Sensory Grid Explorer v4.1 UX Upgrade", page_icon="🧭", layout="wide")
    inject_css()
    init_state()
    sidebar_controls()

    env = st.session_state.env
    render_keyboard_controls()
    obs = env.get_observation(st.session_state.switches)
    scalars = env.current_scalars()

    st.markdown('<div class="app-title">Sensory Grid Explorer v4.1 UX Upgrade</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">This GUI is aligned with the current v4.1 environment. W/A/S/D move in world coordinates, while the underlying cost still comes from the required turn sequence plus forward. Patch panels are rotated into world orientation and show the same center arrow direction as the world grid.</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_bar("Health", env.health, env.config.max_health, "health", f"{env.health} / {env.config.max_health}")
    with m2:
        render_bar("Energy", env.energy, env.config.max_energy, "energy", f"{env.energy:.2f} / {env.config.max_energy:.2f}", subtext="Hard fatigue is kept. Energy wraps back after each fatigue health loss.")
    with m3:
        render_metric_box("Step", str(env.steps))
    with m4:
        render_metric_box("Coverage", f"{100 * scalars['coverage']:.1f}%", subtext=f"Last reward {env.last_reward:+.3f}")

    left, right = st.columns([1.2, 1.15])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">World grid</div>', unsafe_allow_html=True)
        world_visible_mask = current_vision_mask(env.reveal_world_ids().shape, env.agent_pos, env.config.patch_size)
        if st.session_state.fog_of_war:
            st.caption("Fog of war is on. The current vision patch is fully visible. Explored tiles stay visible under a dim memory cover.")
        else:
            st.caption("Outline color shows temperature: neutral in comfort band, blue for colder, red for hotter.")
        st.markdown(
            grid_html(
                env.reveal_world_ids(),
                env.visited_map,
                env.reveal_temperature_field_c(),
                env.config,
                env.agent_pos,
                env.direction,
                reveal_objects=True,
                fog_enabled=bool(st.session_state.fog_of_war),
                visible_mask=world_visible_mask,
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="tiny-note">Agent pose: {env.agent_pos} facing {DIR_SYMBOLS[env.direction]} &nbsp; | &nbsp; Last event: {env.last_event}</div>',
            unsafe_allow_html=True,
        )
        object_legend()
        st.markdown(
            f'<div class="chip-row">'
            f'<span class="kv-chip">Time base cost: {scalars["time_base_cost"]:.2f}</span>'
            f'<span class="kv-chip">Turn extra cost: {scalars["turn_extra_cost"]:.2f}</span>'
            f'<span class="kv-chip">Forward extra cost: {scalars["forward_extra_cost"]:.2f}</span>'
            f'<span class="kv-chip">Thermal extra this tick: {scalars["thermal_extra_this_tick"]:.2f}</span>'
            f'<span class="kv-chip">No-move streak: {int(scalars["consecutive_no_move_steps"])} </span>'
            f'<span class="kv-chip">Turn streak: {int(scalars["consecutive_turn_steps"])} </span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Recent events</div>', unsafe_allow_html=True)
        for line in st.session_state.event_log[:10]:
            st.code(line, language=None)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Latest reward breakdown</div>', unsafe_allow_html=True)
        reward_table(obs["reward_terms"])
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        patch_html(obs["vision"], obs["vision_active"], "Egocentric vision patch", env.direction)
        st.markdown('</div>', unsafe_allow_html=True)

        row1a, row1b = st.columns(2)
        with row1a:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            float_patch_html(obs["temperature_patch_c"], "Egocentric temperature patch", "temperature", obs["temperature_patch_active"], env.config, env.direction)
            st.markdown('</div>', unsafe_allow_html=True)
        with row1b:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            float_patch_html(obs["smell_patch"], "Egocentric smell patch", "smell", obs["smell_patch_active"], env.config, env.direction)
            st.markdown('</div>', unsafe_allow_html=True)

        row2a, row2b = st.columns(2)
        with row2a:
            temp_range = max(1.0, scalars["temperature_max_c"] - scalars["temperature_min_c"])
            render_bar(
                "Temperature at agent",
                scalars["temperature_c"] - scalars["temperature_min_c"],
                temp_range,
                "temp",
                f"{scalars['temperature_c']:.2f} °C",
                subtext=f"Field range {scalars['temperature_min_c']:.1f} to {scalars['temperature_max_c']:.1f} °C",
            )
        with row2b:
            render_bar(
                "Smell at agent",
                scalars["smell"],
                scalars["smell_max"],
                "smell",
                f"{scalars['smell']:.2f}",
                subtext=f"Field max {scalars['smell_max']:.2f}",
            )

        row3a, row3b = st.columns(2)
        with row3a:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            float_patch_html(obs["visited_patch"], "Visited memory patch", "visited", obs["visited_active"], env.config, env.direction)
            st.markdown('</div>', unsafe_allow_html=True)
        with row3b:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            float_patch_html(obs["hazard_patch"], "Hazard memory patch", "hazard", obs["hazard_active"], env.config, env.direction)
            st.markdown('</div>', unsafe_allow_html=True)

        row4a, row4b = st.columns(2)
        with row4a:
            render_bar(
                "Discomfort",
                scalars["discomfort"],
                1.0,
                "neutral",
                f"{scalars['discomfort']:.2f}",
                subtext="Normalized distance from the comfort temperature band",
            )
        with row4b:
            render_bar(
                "Thermal extra this tick",
                scalars["thermal_extra_this_tick"],
                max(1e-6, env.config.thermal_extra_energy_max),
                "neutral",
                f"{scalars['thermal_extra_this_tick']:.2f}",
                subtext=f"Single-step cap = {env.config.thermal_extra_energy_max:.2f}",
            )

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">DQN input preview</div>', unsafe_allow_html=True)
        st.caption("Current training default is vision + temperature patch + smell patch + visited memory + hazard memory. Scalar temperature and scalar smell are off by default.")
        st.write(f"Input length: **{len(obs['agent_input'])}**")
        preview = np.array2string(obs["agent_input"][:120], precision=2, separator=", ")
        st.code(preview + (" ..." if len(obs["agent_input"]) > 120 else ""), language=None)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Current environment and switches"):
        st.code(json.dumps(asdict(env.config), indent=2), language="json")
        st.code(json.dumps(asdict(st.session_state.switches), indent=2), language="json")


if __name__ == "__main__":
    main()
