"""Compressed, agent-observable execution traces for experiment diagnosis."""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


TRACE_SCHEMA_VERSION = 1


def jsonable(value: Any) -> Any:
    """Convert NumPy/dataclass values to a stable JSON-compatible form."""
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, set):
        return [jsonable(item) for item in sorted(value)]
    return value


class EpisodeTraceWriter:
    """Write one gzip-compressed JSONL trace per seeded episode.

    Each record is deliberately limited to observations, transition information,
    and planner state available to the agent.  Hidden grid/object locations are
    never read or serialized here.
    """

    def __init__(self, directory: Path, seed: int) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"seed_{int(seed):05d}.jsonl.gz"
        if self.path.exists():
            raise FileExistsError(f"Trace already exists: {self.path}")
        self._handle = gzip.open(self.path, mode="wt", encoding="utf-8")
        self._closed = False

    def record(self, event: str, **payload: Any) -> None:
        if self._closed:
            raise RuntimeError("Cannot write a closed episode trace")
        row = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "event": str(event),
            **jsonable(payload),
        }
        self._handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        self._handle.write("\n")

    def close(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True

    def __enter__(self) -> "EpisodeTraceWriter":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
