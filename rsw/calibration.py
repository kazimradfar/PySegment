"""Calibration profiles per radiopharmaceutical, typed in by hand and kept on this computer (JSON in the user's
home folder). A profile holds the system sensitivity (cps/MBq) for the isotope and energy window used, and the
isotope; the acquisition time belongs to each scan and is not stored."""
from __future__ import annotations

import json
from pathlib import Path


def default_path() -> Path:
    return Path.home() / ".pysegment" / "calibrations.json"


def load(path: Path | None = None) -> dict:
    p = Path(path or default_path())
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_profile(name: str, sensitivity: float, isotope: str, half_life_h: float, path: Path | None = None) -> None:
    if not name.strip() or sensitivity <= 0:
        raise ValueError("a profile needs a name and a sensitivity above zero")
    p = Path(path or default_path())
    d = load(p)
    d[name.strip()] = {"sensitivity_cps_per_MBq": float(sensitivity), "isotope": isotope, "half_life_h": float(half_life_h)}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=1), encoding="utf-8")


def delete_profile(name: str, path: Path | None = None) -> None:
    p = Path(path or default_path())
    d = load(p)
    if d.pop(name, None) is not None:
        p.write_text(json.dumps(d, indent=1), encoding="utf-8")
