from __future__ import annotations

from dataclasses import asdict, fields
import importlib.util
import io
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINERS_DIR = PROJECT_ROOT / "trains"
DEFAULT_TRAINER_PATH = TRAINERS_DIR / "train_recurrent_patch_fusion_dqn_reference.py"

CUSTOM_PRESET = "Custom (current settings)"
BASELINE_PRESET = "15x15 baseline"
FOUR_X_AREA_SCALED_PRESET = "31x31 area-scaled challenge"
AREA_SCALED_PRESET = "45x45 area-scaled challenge"


def config_from_preset(module: ModuleType, preset_name: str):
    """Build one of the named environment configurations used by the GUI."""
    config = module.EnvConfig()
    if preset_name == BASELINE_PRESET:
        return config
    if preset_name == FOUR_X_AREA_SCALED_PRESET:
        return module.EnvConfig(
            grid_size=31,
            patch_size=5,
            init_health=40,
            max_health=40,
            init_energy=10.0,
            max_energy=10.0,
            max_steps=1000,
            n_fire=8,
            n_ice=4,
            n_meat=12,
            n_flower=8,
            n_glass=8,
            fire_damage=3,
            ice_damage=3,
            glass_damage=2,
            meat_heal=2,
        )
    if preset_name == AREA_SCALED_PRESET:
        return module.EnvConfig(
            grid_size=45,
            patch_size=5,
            init_health=90,
            max_health=90,
            init_energy=10.0,
            max_energy=10.0,
            max_steps=2250,
            n_fire=18,
            n_ice=9,
            n_meat=27,
            n_flower=18,
            n_glass=18,
            fire_damage=3,
            ice_damage=3,
            glass_damage=2,
            meat_heal=2,
        )
    raise ValueError(f"Unknown environment preset: {preset_name}")


def object_count_error(config) -> str | None:
    """Return an explanation when a world cannot place all requested objects."""
    total_objects = sum((
        int(config.n_fire),
        int(config.n_ice),
        int(config.n_meat),
        int(config.n_flower),
        int(config.n_glass),
    ))
    available_cells = int(config.grid_size) ** 2 - 1  # The agent occupies one cell.
    if total_objects > available_cells:
        return f"{total_objects} special cells do not fit in a {config.grid_size}x{config.grid_size} grid with an agent. Maximum: {available_cells}."
    return None


@st.cache_resource(show_spinner=False)
def load_python_module(module_path: str) -> ModuleType:
    path = Path(module_path).resolve()
    module_name = f"trainer_module_{path.stem}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {path}")

    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    added_sys_paths: list[str] = []
    try:
        # Trainer adapters import the shared GUI environment and model
        # packages, plus models under ``models/``.  Make those imports
        # independent of the directory from which Streamlit was launched.
        for directory in (str(PROJECT_ROOT), str(path.parent)):
            if directory not in sys.path:
                sys.path.insert(0, directory)
                added_sys_paths.append(directory)
        spec.loader.exec_module(module)
        return module
    except Exception:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    finally:
        for directory in reversed(added_sys_paths):
            try:
                sys.path.remove(directory)
            except ValueError:
                pass


def discover_trainer_files() -> list[str]:
    return sorted(str(p) for p in TRAINERS_DIR.glob("train_*.py"))


def dataclass_from_dict(cls, payload: Dict[str, Any]):
    names = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in payload.items() if k in names}
    return cls(**filtered)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1rem; padding-bottom: 1.2rem; max-width: 1480px; }
        .app-title { font-size: 2rem; font-weight: 760; letter-spacing: -0.02em; margin-bottom: 0.18rem; }
        .app-subtitle { color: #64748b; margin-bottom: 0.9rem; }
        .card { background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%); border: 1px solid rgba(148,163,184,0.24); border-radius: 18px; padding: 0.95rem 1rem 0.85rem 1rem; box-shadow: 0 8px 24px rgba(15,23,42,0.06); margin-bottom: 1rem; overflow-x: auto; }
        .card-title { font-size: 1.02rem; font-weight: 700; margin-bottom: 0.45rem; }
        .tiny-note { font-size: 0.84rem; color: #64748b; }
        .world-grid { display: grid; gap: 4px; justify-content: start; margin-top: 0.45rem; }
        .cell { border-radius: 9px; display: flex; align-items: center; justify-content: center; font-weight: 700; border: 1px solid rgba(148,163,184,0.18); background: #f1f5f9; color: #334155; box-sizing: border-box; }
        .visited { background: rgba(191,219,254,0.75); box-shadow: inset 0 0 0 2px rgba(59,130,246,0.22); }
        .obj-fire { background: linear-gradient(180deg, #fb923c 0%, #ef4444 100%); color: white; }
        .obj-meat { background: linear-gradient(180deg, #4ade80 0%, #16a34a 100%); color: white; }
        .obj-flower { background: linear-gradient(180deg, #f9a8d4 0%, #ec4899 100%); color: white; }
        .obj-glass { background: linear-gradient(180deg, #7dd3fc 0%, #0ea5e9 100%); color: white; }
        .obj-ice { background: linear-gradient(180deg, #dbeafe 0%, #60a5fa 100%); color: #0f172a; }
        .agent { background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%); color: white; border: 2px solid #93c5fd; }
        .fog-unseen { background: #0f172a; color: transparent; border: 1px solid rgba(148,163,184,0.08); box-shadow: inset 0 0 0 999px rgba(2, 6, 23, 0.78); }
        .fog-memory { box-shadow: inset 0 0 0 999px rgba(15, 23, 42, 0.30); }
        .metric-box { background: #ffffff; border: 1px solid rgba(148,163,184,0.22); border-radius: 16px; padding: 0.78rem 0.9rem; box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04); margin-bottom: 0.75rem; }
        .metric-label { font-size: 0.82rem; color: #64748b; margin-bottom: 0.12rem; }
        .metric-value { font-size: 1.4rem; font-weight: 750; color: #0f172a; }
        .metric-subtext { font-size: 0.78rem; color: #64748b; margin-top: 0.18rem; }
        .bar-track { width: 100%; height: 12px; border-radius: 999px; background: #e2e8f0; overflow: hidden; margin-top: 0.42rem; }
        .bar-fill-temp { height: 100%; background: linear-gradient(90deg, #93c5fd 0%, #f87171 100%); border-radius: 999px; }
        .bar-fill-smell { height: 100%; background: linear-gradient(90deg, #c4b5fd 0%, #7c3aed 100%); border-radius: 999px; }
        .bar-fill-health { height: 100%; background: linear-gradient(90deg, #4ade80 0%, #16a34a 100%); border-radius: 999px; }
        .bar-fill-energy { height: 100%; background: linear-gradient(90deg, #60a5fa 0%, #2563eb 100%); border-radius: 999px; }
        .bar-fill-neutral { height: 100%; background: linear-gradient(90deg, #cbd5e1 0%, #64748b 100%); border-radius: 999px; }
        .legend-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 0.55rem; }
        .legend-pill { display: inline-flex; align-items: center; gap: 8px; border-radius: 999px; padding: 0.36rem 0.68rem; border: 1px solid rgba(148,163,184,0.22); background: white; font-size: 0.9rem; }
        .legend-swatch { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
        .chip-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 0.45rem; }
        .kv-chip { display: inline-flex; gap: 6px; align-items: center; border-radius: 999px; padding: 0.32rem 0.65rem; border: 1px solid rgba(148,163,184,0.25); background: white; font-size: 0.83rem; }
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


def temperature_outline_style(temp_c: float, cfg) -> str:
    if cfg.comfort_low_c <= temp_c <= cfg.comfort_high_c:
        return "box-shadow: inset 0 0 0 1px rgba(148,163,184,0.16);"
    if temp_c < cfg.comfort_low_c:
        frac = min((cfg.comfort_low_c - temp_c) / max(1e-6, cfg.discomfort_temp_scale_c), 1.0)
        return f"box-shadow: inset 0 0 0 2px rgba(59,130,246,{0.20 + 0.70 * frac:.3f});"
    frac = min((temp_c - cfg.comfort_high_c) / max(1e-6, cfg.discomfort_temp_scale_c), 1.0)
    return f"box-shadow: inset 0 0 0 2px rgba(239,68,68,{0.20 + 0.70 * frac:.3f});"


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


def grid_html_fog(world: np.ndarray, visited: np.ndarray, temp_field: np.ndarray, cfg, agent_pos, direction, reveal_objects: bool, fog_enabled: bool = False, visible_mask: np.ndarray | None = None) -> str:
    rows, cols = world.shape
    cell_size = world_cell_size(rows, patch_mode=False)
    font_size = max(12, int(cell_size * 0.55))
    if visible_mask is None:
        visible_mask = np.ones((rows, cols), dtype=bool)
    cls_map = {1: "obj-fire", 2: "obj-meat", 3: "obj-flower", 4: "obj-glass", 5: "obj-ice"}
    sym_map = {0: "", 1: "🔥", 2: "🍖", 3: "🌸", 4: "🔷", 5: "🧊"}
    dir_map = {0: "↑", 1: "→", 2: "↓", 3: "←"}
    html = [f'<div class="world-grid" style="grid-template-columns: repeat({cols}, {cell_size}px);">']
    for r in range(rows):
        for c in range(cols):
            currently_visible = bool(visible_mask[r, c])
            explored = bool(visited[r, c] > 0.5)
            classes = ["cell"]
            text = ""
            if fog_enabled and not currently_visible and not explored:
                classes.append("fog-unseen")
                html.append(f'<div class="{" ".join(classes)}" style="width:{cell_size}px;height:{cell_size}px;font-size:{font_size}px;">{text}</div>')
                continue
            if explored:
                classes.append("visited")
            if (r, c) == agent_pos:
                classes.append("agent")
                text = dir_map[direction]
            else:
                obj = int(world[r, c])
                if reveal_objects and obj != 0:
                    classes.append(cls_map[obj])
                    text = sym_map[obj]
            if fog_enabled and not currently_visible and explored:
                classes.append("fog-memory")
            outline = temperature_outline_style(float(temp_field[r, c]), cfg)
            html.append(f'<div class="{" ".join(classes)}" style="width:{cell_size}px;height:{cell_size}px;font-size:{font_size}px;{outline}">{text}</div>')
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
    class DummyCfg:
        ambient_temperature_c = 22.0
        comfort_low_c = 18.0
        comfort_high_c = 24.0
        discomfort_temp_scale_c = 10.0
    st.markdown(grid_html_fog(rotated_patch, visited, temp_field, DummyCfg(), agent_pos, direction, True, False), unsafe_allow_html=True)
    st.caption("Patch is rotated into world orientation. The center arrow matches the world-grid direction.")


def float_patch_html(patch: np.ndarray, title: str, mode: str, active: bool, cfg, direction: int) -> None:
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
    dir_map = {0: "↑", 1: "→", 2: "↓", 3: "←"}
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
                bg = f"rgba(124,58,237,{0.08 + 0.84 * frac:.3f})"
                text = f"{val:.2f}"
            else:
                if cfg.comfort_low_c <= val <= cfg.comfort_high_c:
                    bg = "rgba(226,232,240,0.65)"
                elif val < cfg.comfort_low_c:
                    frac = min((cfg.comfort_low_c - val) / max(1e-6, temp_span), 1.0)
                    bg = f"rgba(59,130,246,{0.12 + 0.78 * frac:.3f})"
                else:
                    frac = min((val - cfg.comfort_high_c) / max(1e-6, temp_span), 1.0)
                    bg = f"rgba(239,68,68,{0.12 + 0.78 * frac:.3f})"
                text = f"{val:.1f}"
            extra = ""
            if (r, c) == (center, center):
                text = dir_map[direction]
                extra = "border: 2px solid rgba(15,23,42,0.7); box-shadow: inset 0 0 0 999px rgba(255,255,255,0.08);"
            html.append(f'<div class="cell" style="width:{cell_size}px;height:{cell_size}px;background:{bg};font-size:{font_size}px;{extra}">{text}</div>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)
    st.caption("Patch is rotated into world orientation. The center arrow matches the world-grid direction.")


def object_legend() -> None:
    st.markdown(
        """
        <div class="legend-row">
            <span class="legend-pill"><span class="legend-swatch" style="background:#ef4444"></span>Fire</span>
            <span class="legend-pill"><span class="legend-swatch" style="background:#16a34a"></span>Meat</span>
            <span class="legend-pill"><span class="legend-swatch" style="background:#ec4899"></span>Flower</span>
            <span class="legend-pill"><span class="legend-swatch" style="background:#0ea5e9"></span>Glass</span>
            <span class="legend-pill"><span class="legend-swatch" style="background:#60a5fa"></span>Ice</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def reward_table(reward_terms: Dict[str, float]) -> None:
    if not reward_terms:
        st.caption("No reward terms recorded yet.")
        return
    rows = [{"term": k, "value": round(float(v), 4)} for k, v in reward_terms.items()]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def compare_dict_fields(reference: Dict[str, Any], current: Dict[str, Any], prefix: str) -> list[str]:
    diffs = []
    all_keys = sorted(set(reference.keys()) | set(current.keys()))
    for key in all_keys:
        rv = reference.get(key, "<missing>")
        cv = current.get(key, "<missing>")
        if rv != cv:
            diffs.append(f"{prefix}.{key}: training={rv!r}, current={cv!r}")
    return diffs


def torch_load_checkpoint(uploaded_file) -> Dict[str, Any]:
    if uploaded_file is None:
        raise ValueError("No checkpoint uploaded.")
    return torch.load(io.BytesIO(uploaded_file.getvalue()), map_location="cpu")


def get_trainer_module() -> ModuleType | None:
    path = st.session_state.get("trainer_path")
    if not path:
        return None
    return load_python_module(path)


def trainer_ready() -> bool:
    return get_trainer_module() is not None


def trainer_name() -> str:
    module = get_trainer_module()
    if module is None:
        return "None"
    return getattr(module, "TRAINER_DISPLAY_NAME", Path(st.session_state.trainer_path).stem)


def init_state() -> None:
    if "trainer_path" not in st.session_state:
        st.session_state.trainer_path = None
    if "config" not in st.session_state:
        st.session_state.config = None
    if "switches" not in st.session_state:
        st.session_state.switches = None
    if "env" not in st.session_state:
        st.session_state.env = None
    if "event_log" not in st.session_state:
        st.session_state.event_log = ["Episode reset."]
    if "loaded_checkpoint" not in st.session_state:
        st.session_state.loaded_checkpoint = None
    if "loaded_net" not in st.session_state:
        st.session_state.loaded_net = None
    if "checkpoint_meta" not in st.session_state:
        st.session_state.checkpoint_meta = {}
    if "auto_step_delay_s" not in st.session_state:
        st.session_state.auto_step_delay_s = 0.25
    if "fog_of_war" not in st.session_state:
        st.session_state.fog_of_war = False
    if "hide_world_grid" not in st.session_state:
        st.session_state.hide_world_grid = False
    if "runtime_context" not in st.session_state:
        st.session_state.runtime_context = None


def bootstrap_from_trainer(module: ModuleType) -> None:
    env, switches = module.build_training_env()
    env.reset(seed=None)
    st.session_state.config = env.config
    st.session_state.switches = switches
    st.session_state.env = env
    st.session_state.loaded_checkpoint = None
    st.session_state.loaded_net = None
    st.session_state.checkpoint_meta = {}
    st.session_state.event_log = ["Episode reset."]
    st.session_state.runtime_context = module.init_runtime_context()


def reset_environment(seed=None) -> bool:
    module = get_trainer_module()
    if module is None or st.session_state.config is None:
        return False
    config_error = object_count_error(st.session_state.config)
    if config_error is not None:
        st.sidebar.error(config_error)
        return False
    env = module.SensoryGridEnv(st.session_state.config)
    env.reset(seed=seed)
    st.session_state.env = env
    st.session_state.event_log = ["Episode reset."]
    st.session_state.runtime_context = module.reset_runtime_context(st.session_state.runtime_context)
    return True


def log_event(prefix: str, info: Dict[str, Any], reward: float, action_name: str) -> None:
    st.session_state.event_log.insert(
        0,
        f"{prefix} step {info['steps']:03d} | action {action_name} | reward {reward:+.3f} | health {info['health']} | energy {info['energy']:.2f} | {info['last_event']}",
    )
    st.session_state.event_log = st.session_state.event_log[:25]


def world_direction_sequence(current_direction: int, target_direction: int, action_forward: int, action_left: int, action_right: int) -> list[int]:
    current_direction = int(current_direction) % 4
    target_direction = int(target_direction) % 4
    diff = (target_direction - current_direction) % 4
    if diff == 0:
        return [action_forward]
    if diff == 1:
        return [action_right, action_forward]
    if diff == 3:
        return [action_left, action_forward]
    return [action_right, action_right, action_forward]


def sequence_name(seq: list[int], module: ModuleType) -> str:
    mapping = {module.ACTION_FORWARD: "forward", module.ACTION_LEFT: "left", module.ACTION_RIGHT: "right"}
    return " -> ".join(mapping[a] for a in seq)


def execute_manual_sequence(actions: list[int], world_label: str) -> None:
    module = get_trainer_module()
    env = st.session_state.env
    if module is None or env is None or env.terminated or env.truncated:
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
    log_event("manual", info, total_reward, f"{world_label} [{sequence_name(actions, module)}]")
    done = env.terminated or env.truncated
    st.session_state.runtime_context = module.reset_runtime_context(st.session_state.runtime_context) if done else module.reset_runtime_context(st.session_state.runtime_context)
    if done:
        st.session_state.event_log.insert(0, "Episode ended. Reset to start again.")


def do_world_move(target_direction: int, label: str) -> None:
    module = get_trainer_module()
    if module is None:
        return
    seq = world_direction_sequence(st.session_state.env.direction, target_direction, module.ACTION_FORWARD, module.ACTION_LEFT, module.ACTION_RIGHT)
    execute_manual_sequence(seq, label)


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
            if (!parentDoc) { return; }
            if (!parentWindow.__sensoryGridWasdWorldHandlerInstalled) {
                parentWindow.__sensoryGridWasdWorldHandlerInstalled = true;
                parentWindow.__sensoryGridWasdWorldHandler = function (event) {
                    const tag = (event.target && event.target.tagName ? event.target.tagName : '').toLowerCase();
                    if (tag === 'input' || tag === 'textarea' || tag === 'select') { return; }
                    const key = (event.key || '').toLowerCase();
                    if (!['w', 'a', 's', 'd'].includes(key)) { return; }
                    const meta = parentDoc.getElementById('keyboard-world-move-meta');
                    if (!meta) { return; }
                    const label = meta.dataset[key];
                    if (!label) { return; }
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


def current_input_dim() -> int:
    env = st.session_state.env
    if env is None:
        return 0
    obs = env.get_observation(st.session_state.switches)
    return int(len(obs["agent_input"]))


def model_input_matches_checkpoint() -> bool:
    payload = st.session_state.loaded_checkpoint
    if payload is None:
        return False
    return int(payload.get("input_dim", -1)) == current_input_dim()


def checkpoint_env_config():
    payload = st.session_state.loaded_checkpoint
    module = get_trainer_module()
    if payload is None or module is None:
        return None
    return dataclass_from_dict(module.EnvConfig, payload.get("env_config", {}))


def checkpoint_switches():
    payload = st.session_state.loaded_checkpoint
    module = get_trainer_module()
    if payload is None or module is None:
        return None
    return dataclass_from_dict(module.ObservationSwitches, payload.get("switches", {}))


def current_env_matches_checkpoint() -> bool:
    ckpt_cfg = checkpoint_env_config()
    return ckpt_cfg is not None and asdict(ckpt_cfg) == asdict(st.session_state.config)


def current_switches_match_checkpoint() -> bool:
    ckpt_switches = checkpoint_switches()
    return ckpt_switches is not None and asdict(ckpt_switches) == asdict(st.session_state.switches)


def checkpoint_mismatch_messages() -> list[str]:
    payload = st.session_state.loaded_checkpoint
    module = get_trainer_module()
    if payload is None or module is None:
        return ["No checkpoint loaded."]
    messages = []
    ckpt_cfg = dataclass_from_dict(module.EnvConfig, payload.get("env_config", {}))
    ckpt_switches = dataclass_from_dict(module.ObservationSwitches, payload.get("switches", {}))
    if not model_input_matches_checkpoint():
        messages.append(f"input_dim: training={int(payload.get('input_dim', -1))!r}, current={current_input_dim()!r}")
    messages.extend(compare_dict_fields(asdict(ckpt_cfg), asdict(st.session_state.config), "env"))
    messages.extend(compare_dict_fields(asdict(ckpt_switches), asdict(st.session_state.switches), "switches"))
    return messages


def mismatch_severity() -> str:
    if st.session_state.loaded_checkpoint is None:
        return "none"
    if model_input_matches_checkpoint() and current_env_matches_checkpoint() and current_switches_match_checkpoint():
        return "match"
    if not model_input_matches_checkpoint():
        return "input_mismatch"
    return "config_mismatch"


def apply_loaded_checkpoint(payload: Dict[str, Any]) -> None:
    module = get_trainer_module()
    if module is None:
        raise RuntimeError("Load trainer module first.")
    env_cfg = dataclass_from_dict(module.EnvConfig, payload.get("env_config", {}))
    switches = dataclass_from_dict(module.ObservationSwitches, payload.get("switches", {}))
    net = module.build_model_from_checkpoint(payload)
    st.session_state.loaded_checkpoint = payload
    st.session_state.loaded_net = net
    st.session_state.checkpoint_meta = {
        "trainer_name": payload.get("trainer_name", trainer_name()),
        "model_arch": payload.get("model_arch", "unknown"),
        "input_dim": int(payload.get("input_dim", -1)),
        "num_actions": int(payload.get("num_actions", getattr(module, "N_ACTIONS", 0))),
        "episode": payload.get("episode"),
        "global_step": payload.get("global_step"),
        "eval_metrics": payload.get("eval_metrics", {}),
        "soft_score": payload.get("composite_eval_score"),
    }
    st.session_state.config = env_cfg
    st.session_state.switches = switches
    reset_environment(seed=None)


def predict_action() -> int:
    module = get_trainer_module()
    if module is None or st.session_state.loaded_net is None:
        raise RuntimeError("No trainer/model loaded.")
    obs = st.session_state.env.get_observation(st.session_state.switches)
    action, runtime_context = module.predict_action_for_gui(
        st.session_state.loaded_net,
        obs,
        st.session_state.env.config,
        st.session_state.switches,
        st.session_state.runtime_context,
    )
    st.session_state.runtime_context = runtime_context
    return int(action)


def step_with_model() -> bool:
    module = get_trainer_module()
    env = st.session_state.env
    if module is None or env is None or env.terminated or env.truncated:
        return True
    action = predict_action()
    _, reward, terminated, truncated, info = env.step(action, st.session_state.switches)
    st.session_state.runtime_context = module.update_runtime_context_after_env_step(st.session_state.runtime_context, action, reward, terminated or truncated)
    action_name = {module.ACTION_FORWARD: "forward", module.ACTION_LEFT: "left", module.ACTION_RIGHT: "right"}[action]
    log_event("model", info, reward, action_name)
    if terminated or truncated:
        st.session_state.event_log.insert(0, "Model run reached episode end.")
    return terminated or truncated


def checkpoint_status_box() -> None:
    payload = st.session_state.loaded_checkpoint
    if payload is None:
        st.info("No checkpoint loaded yet.")
        return
    meta = st.session_state.checkpoint_meta
    severity = mismatch_severity()
    color = {"match": "#16a34a", "config_mismatch": "#f59e0b", "input_mismatch": "#ef4444"}.get(severity, "#94a3b8")
    label = {"match": "Exact match", "config_mismatch": "Mismatch allowed", "input_mismatch": "Input mismatch allowed"}.get(severity, "Unknown")
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;margin:0.25rem 0 0.75rem 0;'><span style='display:inline-block;width:12px;height:12px;border-radius:999px;background:{color};'></span><span><strong>{label}</strong></span></div>",
        unsafe_allow_html=True,
    )
    st.write(f"Trainer: **{meta.get('trainer_name', trainer_name())}**")
    st.write(f"Model architecture: **{meta.get('model_arch', 'unknown')}**")
    st.write(f"Checkpoint input dim: **{meta.get('input_dim')}**")
    st.write(f"Current GUI input dim: **{current_input_dim()}**")
    st.write(f"Input compatible: **{model_input_matches_checkpoint()}**")
    st.write(f"Environment config matches checkpoint: **{current_env_matches_checkpoint()}**")
    st.write(f"Observation switches match checkpoint: **{current_switches_match_checkpoint()}**")
    mismatches = checkpoint_mismatch_messages()
    if mismatches and mismatches != ["No checkpoint loaded."]:
        st.caption("Differences from the checkpoint training setup")
        for message in mismatches:
            st.warning(message)
    if meta.get("episode") is not None:
        st.write(f"Saved at episode: **{meta.get('episode')}**")
    if meta.get("global_step") is not None:
        st.write(f"Saved at global step: **{meta.get('global_step')}**")
    if meta.get("soft_score") is not None:
        st.write(f"Soft score: **{float(meta.get('soft_score')):.4f}**")
    eval_metrics = meta.get("eval_metrics", {})
    if eval_metrics:
        st.json(eval_metrics)


def sidebar_controls() -> bool:
    auto_run_clicked = False

    st.sidebar.header("1. Trainer module")
    trainer_files = discover_trainer_files()
    default_index = trainer_files.index(st.session_state.trainer_path) if st.session_state.trainer_path in trainer_files else (trainer_files.index(str(DEFAULT_TRAINER_PATH)) if str(DEFAULT_TRAINER_PATH) in trainer_files else 0)
    selected_trainer = st.sidebar.selectbox("Select trainer .py", trainer_files, index=default_index if trainer_files else None)
    if st.sidebar.button("Load trainer", use_container_width=True):
        module = load_python_module(selected_trainer)
        required = ["build_training_env", "build_model_from_checkpoint", "predict_action_for_gui", "update_runtime_context_after_env_step", "reset_runtime_context", "SensoryGridEnv", "EnvConfig", "ObservationSwitches"]
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            st.sidebar.error(f"Trainer missing interface functions: {missing}")
        else:
            st.session_state.trainer_path = selected_trainer
            bootstrap_from_trainer(module)
            st.sidebar.success(f"Loaded trainer: {getattr(module, 'TRAINER_DISPLAY_NAME', Path(selected_trainer).stem)}")

    module = get_trainer_module()
    if module is None:
        st.sidebar.info("Load a trainer module first.")
        return auto_run_clicked

    spec = getattr(module, "get_gui_interface_spec", lambda: {})()
    if spec:
        st.sidebar.caption(json.dumps(spec, indent=2))

    st.sidebar.header("2. Checkpoint")
    uploaded = st.sidebar.file_uploader("Upload .pt checkpoint", type=["pt"])
    if st.sidebar.button("Load checkpoint", use_container_width=True):
        if uploaded is None:
            st.sidebar.error("Upload a .pt checkpoint first.")
        else:
            try:
                payload = torch_load_checkpoint(uploaded)
                apply_loaded_checkpoint(payload)
                st.sidebar.success("Checkpoint loaded.")
            except Exception as exc:
                st.sidebar.error(f"Failed to load checkpoint: {exc}")

    checkpoint_status_box()

    st.sidebar.header("Environment")
    config = st.session_state.config
    switches = st.session_state.switches

    preset_name = st.sidebar.selectbox(
        "Environment preset",
        [CUSTOM_PRESET, BASELINE_PRESET, FOUR_X_AREA_SCALED_PRESET, AREA_SCALED_PRESET],
        help="Presets replace every environment setting shown below. Custom keeps the values you choose manually.",
    )
    if preset_name != CUSTOM_PRESET:
        preset_config = config_from_preset(module, preset_name)
        total_special_cells = sum((
            preset_config.n_fire,
            preset_config.n_ice,
            preset_config.n_meat,
            preset_config.n_flower,
            preset_config.n_glass,
        ))
        st.sidebar.caption(
            f"{preset_config.grid_size}x{preset_config.grid_size}; "
            f"health {preset_config.init_health}; energy {preset_config.init_energy:g}; "
            f"{total_special_cells} special cells; {preset_config.max_steps} steps."
        )
        if st.sidebar.button(f"Apply {preset_name} + reset", use_container_width=True):
            st.session_state.config = preset_config
            reset_environment(seed=None)
            st.rerun()

    grid_size = st.sidebar.slider("Grid size", 9, 61, config.grid_size, step=2)
    patch_size = st.sidebar.selectbox("Patch size", [3, 5, 7], index=[3, 5, 7].index(config.patch_size))
    init_health = st.sidebar.slider("Initial health", 1, 200, config.init_health)
    init_energy = st.sidebar.slider("Initial energy", 1, 100, int(round(config.init_energy)))
    max_steps = st.sidebar.slider("Max steps", 30, 5000, config.max_steps, step=10)

    st.sidebar.subheader("Objects")
    n_fire = st.sidebar.slider("Fire count", 0, 100, config.n_fire)
    n_ice = st.sidebar.slider("Ice count", 0, 100, config.n_ice)
    n_meat = st.sidebar.slider("Meat count", 0, 100, config.n_meat)
    n_flower = st.sidebar.slider("Flower count", 0, 100, config.n_flower)
    n_glass = st.sidebar.slider("Glass count", 0, 100, config.n_glass)

    st.sidebar.subheader("Contact outcomes")
    fire_damage = st.sidebar.slider("Fire damage", 1, 100, config.fire_damage)
    ice_damage = st.sidebar.slider("Ice damage", 1, 100, config.ice_damage)
    glass_damage = st.sidebar.slider("Glass damage", 1, 100, config.glass_damage)
    meat_heal = st.sidebar.slider("Meat heal", 1, 100, config.meat_heal)

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
    survival_bonus = st.sidebar.slider("Survival end bonus", 0.0, 2.0, float(getattr(config, "survival_bonus", 0.0)), step=0.05)
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

    st.session_state.switches = module.ObservationSwitches(
        include_vision=include_vision,
        include_temperature=include_temperature,
        include_smell=include_smell,
        include_temperature_patch=include_temperature_patch,
        include_smell_patch=include_smell_patch,
        include_visited_memory=include_visited,
        include_hazard_memory=include_hazard,
    )

    comfort_low_c, comfort_high_c = sorted((comfort_low_c, comfort_high_c))
    st.session_state.config = module.EnvConfig(
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
        survival_bonus=survival_bonus,
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

    total_special_cells = n_fire + n_ice + n_meat + n_flower + n_glass
    st.sidebar.caption(f"Special cells: {total_special_cells} / {grid_size * grid_size - 1} available cells")

    col_a, col_b = st.sidebar.columns(2)
    if col_a.button("Apply + reset", use_container_width=True):
        reset_environment(seed=None)
    if col_b.button("Reset episode", use_container_width=True):
        reset_environment(seed=None)

    st.sidebar.subheader("Manual controls")
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

    st.sidebar.subheader("Display")
    st.session_state.fog_of_war = st.sidebar.toggle("Fog of war", value=bool(st.session_state.fog_of_war))
    st.session_state.hide_world_grid = st.sidebar.toggle("Hide world grid", value=bool(st.session_state.hide_world_grid))

    st.sidebar.subheader("Model controls")
    st.session_state.auto_step_delay_s = st.sidebar.slider("Step delay (seconds)", min_value=0.0, max_value=2.0, value=float(st.session_state.auto_step_delay_s), step=0.05)
    model_disabled = st.session_state.loaded_net is None or disabled
    if st.sidebar.button("Model one step", use_container_width=True, disabled=model_disabled):
        step_with_model()
    auto_run_clicked = st.sidebar.button("Auto run to game over", use_container_width=True, disabled=model_disabled)

    st.sidebar.caption("Load order is now trainer module first, then checkpoint. The GUI uses the trainer module runtime interface to decode observations and step the model.")
    return auto_run_clicked


def render_dashboard(container) -> None:
    env = st.session_state.env
    if env is None:
        return
    obs = env.get_observation(st.session_state.switches)
    scalars = env.current_scalars()

    with container.container():
        st.markdown('<div class="app-title">Sensory Grid Current Checkpoint Runner</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="app-subtitle">Trainer-first modular GUI. Load a trainer module, then load a checkpoint. The GUI reads the model through the trainer runtime interface rather than hardcoding architecture logic.</div>',
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_bar("Health", env.health, env.config.max_health, "health", f"{env.health} / {env.config.max_health}")
        with m2:
            render_bar("Energy", env.energy, env.config.max_energy, "energy", f"{env.energy:.2f} / {env.config.max_energy:.2f}", subtext="Hard fatigue kept")
        with m3:
            render_metric_box("Step", str(env.steps), subtext=f"Trainer {trainer_name()}")
        with m4:
            render_metric_box("Coverage", f"{100 * scalars['coverage']:.1f}%", subtext=f"Last reward {env.last_reward:+.3f}")

        left, right = st.columns([1.2, 1.15])

        with left:
            if not st.session_state.hide_world_grid:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">World grid</div>', unsafe_allow_html=True)
                world_visible_mask = current_vision_mask(env.reveal_world_ids().shape, env.agent_pos, env.config.patch_size)
                st.markdown(
                    grid_html_fog(
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
                st.markdown(f'<div class="tiny-note">Agent pose: {env.agent_pos} &nbsp; | &nbsp; Last event: {env.last_event}</div>', unsafe_allow_html=True)
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
            else:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Memory patches</div>', unsafe_allow_html=True)
                st.caption("World grid is hidden. These two memory patches are shown in its place.")
                mem_a, mem_b = st.columns(2)
                with mem_a:
                    float_patch_html(obs["visited_patch"], "Visited memory patch", "visited", obs["visited_active"], env.config, env.direction)
                with mem_b:
                    float_patch_html(obs["hazard_patch"], "Hazard memory patch", "hazard", obs["hazard_active"], env.config, env.direction)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Recent events</div>', unsafe_allow_html=True)
            for line in st.session_state.event_log[:12]:
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
                render_bar("Temperature at agent", scalars["temperature_c"] - scalars["temperature_min_c"], temp_range, "temp", f"{scalars['temperature_c']:.2f} °C", subtext=f"Field range {scalars['temperature_min_c']:.1f} to {scalars['temperature_max_c']:.1f} °C")
            with row2b:
                render_bar("Smell at agent", scalars["smell"], scalars["smell_max"], "smell", f"{scalars['smell']:.2f}", subtext=f"Field max {scalars['smell_max']:.2f}")

            if not st.session_state.hide_world_grid:
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
                render_bar("Discomfort", scalars["discomfort"], 1.0, "neutral", f"{scalars['discomfort']:.2f}", subtext="Normalized distance from comfort band")
            with row4b:
                render_bar("Thermal extra this tick", scalars["thermal_extra_this_tick"], max(1e-6, env.config.thermal_extra_energy_max), "neutral", f"{scalars['thermal_extra_this_tick']:.2f}", subtext=f"Single-step cap = {env.config.thermal_extra_energy_max:.2f}")

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">DQN input preview</div>', unsafe_allow_html=True)
            st.write(f"Input length: **{len(obs['agent_input'])}**")
            preview = np.array2string(obs["agent_input"][:120], precision=2, separator=", ")
            st.code(preview + (" ..." if len(obs["agent_input"]) > 120 else ""), language=None)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Checkpoint metadata</div>', unsafe_allow_html=True)
            if st.session_state.checkpoint_meta:
                st.json(st.session_state.checkpoint_meta)
            else:
                st.caption("No checkpoint metadata yet.")
            st.markdown('</div>', unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Sensory Grid Current Checkpoint Runner", page_icon="🤖", layout="wide")
    inject_css()
    init_state()
    render_keyboard_controls()
    auto_run_clicked = sidebar_controls()

    dashboard = st.empty()
    if st.session_state.env is not None:
        render_dashboard(dashboard)

    if auto_run_clicked:
        if st.session_state.loaded_net is None:
            st.error("Load trainer and checkpoint first.")
            return
        while True:
            done = step_with_model()
            render_dashboard(dashboard)
            if done:
                break
            time.sleep(float(st.session_state.auto_step_delay_s))


if __name__ == "__main__":
    main()
