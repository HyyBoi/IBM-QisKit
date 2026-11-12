"""Saving / loading results with provenance, and replay-compatibility checks."""

from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

REPLAY_KEYS = ("scenario_seed", "n", "k", "weights", "tau", "scenario_fingerprint")


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def git_hash() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def provenance() -> dict:
    import qiskit
    import scipy

    from . import __version__

    return {
        "floodcut_version": __version__,
        "git_commit": git_hash(),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "qiskit": qiskit.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "simulator": "qiskit.quantum_info.Statevector (exact) + qiskit.primitives.StatevectorSampler (shots)",
        "synthetic_data_notice": "All scenario data are synthetic; no real geography, infrastructure or hydrology.",
    }


def save_json(path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(obj), fh, indent=1)
    return path


def load_json(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _norm(v):
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    if isinstance(v, float):
        return round(v, 10)
    return v


def settings_match(saved: dict, current: dict) -> tuple[bool, list[str]]:
    """Compare the problem-defining settings of a saved instance with the current UI
    settings. Returns (match, list of mismatching keys)."""
    bad = [k for k in REPLAY_KEYS if _norm(saved.get(k)) != _norm(current.get(k))]
    return (not bad, bad)


def find_matching_instance(bench: dict, current: dict):
    for inst in bench.get("instances", []):
        ok, _ = settings_match(inst["settings"], current)
        if ok:
            return inst
    return None
