"""Registration accuracy against expert landmarks: POPI 4D CT data (Léon Bérard Cancer Center & CREATIS, Lyon;
Vandemeulebroucke et al., Med Phys 2011;38:166-178): six patients, about 100 landmarks in the inhale (00) and exhale
phase. Fixed image = inhale, moving image = exhale; target registration error TRE = |R(p_inhale) - p_exhale| (mm).

Methods (PySegment API, default settings unless named):
  none                      : no registration (landmark motion)
  simpleitk rigid / affine  : default engine
  elastix lung bspline      : engine 'elastix', profile 'lung' (affine + B-spline, 10 mm final grid, lung mask)

Split: the settings of the lung profile were chosen on the tuning patients (bh, mm2) only; the other four patients
are the independent test set. The default SimpleITK B-spline is not run here (47 min per patient on two cores; one
patient: 3.25 mm, see README).

Data: https://github.com/open-vv/popi-model (dir_validation_data.md); per patient folder the inhale and exhale
.mhd/.raw and .pts files, named 00.mhd / mhd_00.mhd and 00.pts / pts_00.pts. Images are not stored in the repository.

Reproduce:  python tools/validate_registration_landmarks.py --popi <folder with bl, ng, dx, gt, mm2, bh>
Results are kept per patient in <out>/raw (an interrupted run continues).
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsw import registration as rg  # noqa: E402
from rsw.io import Geometry, Volume  # noqa: E402

EXHALE = {"bl": "60", "ng": "50", "dx": "50", "gt": "50", "mm2": "50", "bh": "50"}
TUNING = ("bh", "mm2")
METHODS = {"simpleitk rigid": dict(mode="rigid"), "simpleitk affine": dict(mode="affine"),
           "elastix lung bspline": dict(mode="bspline", engine="elastix", profile="lung")}


def find(folder: Path, phase: str, ext: str) -> Path:
    for name in (f"{phase}.{ext}", f"{ext}_{phase}.{ext}"):
        if (folder / name).exists():
            return folder / name
    raise FileNotFoundError(f"{folder}: no {phase}.{ext}")


def read_mhd(path: Path) -> Volume:
    """MetaImage header + raw data next to it (the raw file may have been renamed together with the header)."""
    h = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
    h = {k.strip(): v.strip() for k, v in h.items()}
    size = [int(v) for v in h["DimSize"].split()]
    raw = path.with_suffix(".raw")
    dtype = {"MET_SHORT": np.int16, "MET_USHORT": np.uint16, "MET_FLOAT": np.float32}[h["ElementType"]]
    arr = np.fromfile(raw, dtype=dtype).reshape(size[::-1]).astype(np.float32)
    geo = Geometry(tuple(size), tuple(float(v) for v in h["ElementSpacing"].split()),
                   tuple(float(v) for v in h["Offset"].split()), tuple(float(v) for v in h["TransformMatrix"].split()))
    return Volume(path.stem, arr, geo, "CT", "HU")


def read_pts(path: Path) -> np.ndarray:
    return np.array([[float(v) for v in line.split()] for line in path.read_text().splitlines() if line.strip()])


def stats(e: np.ndarray) -> dict:
    return {"tre_mean": float(e.mean()), "tre_sd": float(e.std()), "tre_median": float(np.median(e)),
            "tre_p95": float(np.percentile(e, 95)), "tre_max": float(e.max()), "pct_below_2mm": float(100 * (e < 2).mean())}


def run_case(folder: Path, only: set | None = None) -> list[dict]:
    case = folder.name
    ex = EXHALE[case]
    fixed, moving = read_mhd(find(folder, "00", "mhd")), read_mhd(find(folder, ex, "mhd"))
    pf, pm = read_pts(find(folder, "00", "pts")), read_pts(find(folder, ex, "pts"))
    split = "tuning" if case in TUNING else "test"
    rows = [{"patient": case, "split": split, "method": "none", "landmarks": len(pf), "seconds": 0.0,
             **stats(np.linalg.norm(pf - pm, axis=1))}]
    for name, kw in METHODS.items():
        if only and name not in only:
            continue
        t0 = time.perf_counter()
        res = rg.register(fixed, moving, iterations=500, **kw)
        sec = time.perf_counter() - t0
        e = np.linalg.norm(np.array([res.transform.TransformPoint(tuple(p)) for p in pf]) - pm, axis=1)
        rows.append({"patient": case, "split": split, "method": name, "landmarks": len(pf), "seconds": sec, **stats(e),
                     "folding_fraction": (res.jacobian or {}).get("fraction_nonpositive"), "kept_start": res.kept_start,
                     "warnings": " | ".join(w.split(":")[0] for w in res.warnings)})
        print(f"{case} {name}: TRE {e.mean():.2f} mm ({sec:.0f} s)", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--popi", required=True)
    ap.add_argument("--out", default=str(ROOT / "validation" / "registration_landmarks"))
    ap.add_argument("--cases", default=",".join(EXHALE))
    ap.add_argument("--rerun", default="", help="re-run only these methods (comma list) for patients already done, "
                    "replacing their rows, e.g. after a change of that method")
    a = ap.parse_args(argv)
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(2)
    out, raw = Path(a.out), Path(a.out) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rerun = set(filter(None, a.rerun.split(",")))
    for case in a.cases.split(","):
        folder, done = Path(a.popi) / case, raw / f"{case}.json"
        if not folder.exists():
            continue
        if done.exists() and rerun:
            old = [r for r in json.loads(done.read_text()) if r["method"] not in rerun]
            new = [r for r in run_case(folder, rerun) if r["method"] != "none"]
            done.write_text(json.dumps(old + new))
        elif not done.exists():
            done.write_text(json.dumps(run_case(folder)))
    rows = [r for f in sorted(raw.glob("*.json")) for r in json.loads(f.read_text())]
    df = pd.DataFrame(rows)
    df.round(4).to_csv(out / "cases.csv", index=False)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    try:
        import itk
        itk_version = itk.Version.GetITKVersion()
    except Exception:  # noqa: BLE001
        itk_version = None
    summary = {"pysegment_commit": commit, "python": platform.python_version(), "sitk": sitk.Version_VersionString(),
               "itk": itk_version, "patients": sorted(df.patient.unique().tolist()), "by_split": {}}
    for (split, method), g in df.groupby(["split", "method"], sort=False):
        summary["by_split"].setdefault(split, {})[method] = {
            "patients": len(g), "tre_mean_of_patient_means_mm": round(float(g.tre_mean.mean()), 2),
            "tre_mean_range_mm": f"{g.tre_mean.min():.2f}-{g.tre_mean.max():.2f}",
            "tre_p95_max_mm": round(float(g.tre_p95.max()), 2), "tre_max_mm": round(float(g.tre_max.max()), 2),
            "seconds_mean": round(float(g.seconds.mean()), 0)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
