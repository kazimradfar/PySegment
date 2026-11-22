"""Validation of SPECT quantification in PySegment: counts -> activity, decay correction and conservation of counts
when a counts image is resampled or registered.

A. A digital SPECT phantom (counts per voxel, known total) is measured with a spherical ROI through
   rsw.measure.roi_stats / roi_table with a calibration (sensitivity, acquisition time, decay): the activity must
   equal total counts / (sensitivity x time) x 2^(elapsed / half-life), computed by hand.
B. Conservation of counts: the same phantom resampled by PySegment (rsw.registration.resample_volume) onto coarser
   and finer grids and through affine transforms with scaling (Jacobian) must keep the total counts.
C. Size of the decay during the acquisition, which PySegment does not correct (lambda t / (1 - exp(-lambda t))).

Reproduce (a few seconds): python tools/validate_spect_counts.py
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsw import measure  # noqa: E402
from rsw import registration as rg  # noqa: E402
from rsw.io import Geometry, Volume  # noqa: E402


def phantom():
    """64^3 grid of 4.8 mm voxels: body cylinder 10 counts/voxel, sphere of 30 mm radius 200 counts/voxel, smoothed
    (FWHM 10 mm) as a reconstructed SPECT image."""
    sp = 4.8
    z, y, x = np.mgrid[:64, :64, :64].astype(float)
    c = 31.5
    body = ((y - c) ** 2 + (x - c) ** 2) * sp ** 2 <= 110 ** 2
    body &= np.abs(z - c) * sp <= 120
    sph = ((z - c) ** 2 + (y - c) ** 2 + (x - c - 6) ** 2) * sp ** 2 <= 30 ** 2
    arr = 10.0 * body + 190.0 * sph
    img = sitk.SmoothingRecursiveGaussian(sitk.GetImageFromArray(arr.astype(np.float32)), 10 / 2.355 / sp)
    arr = np.clip(sitk.GetArrayFromImage(img), 0, None)
    geo = Geometry((64, 64, 64), (sp, sp, sp), (-151.2, -151.2, -151.2), (1, 0, 0, 0, 1, 0, 0, 0, 1))
    return Volume("spect", arr.astype(np.float32), geo, "SPECT", "CNTS"), sph


def part_a(vol: Volume, sph: np.ndarray) -> dict:
    sp = (vol.geometry.spacing[2], vol.geometry.spacing[1], vol.geometry.spacing[0])
    region = np.ones_like(sph)                                     # the whole field of view: all counts
    st = measure.roi_stats(vol.array, region, sp)
    calib = {"sensitivity": 9.4, "time_s": 1800.0, "decay": measure.decay_factor(159.53, 24.0)}
    tab = measure.roi_table([{"kind": "roi", "roi": "fov", "image": "spect", "unit": "CNTS", **st}], calib)
    counts = float(vol.array.astype(np.float64).sum())
    hand = counts / (9.4 * 1800.0) * 2 ** (24.0 / 159.53)
    return {"total_counts_true": counts, "total_counts_pysegment": float(tab.total_counts[0]),
            "activity_MBq_hand": hand, "activity_MBq_pysegment": float(tab.activity_MBq[0]),
            "relative_difference": float(tab.activity_MBq[0] / hand - 1),
            "decay_factor_24h_lu177": calib["decay"], "decay_factor_hand": 2 ** (24 / 159.53)}


def part_b(vol: Volume) -> list[dict]:
    total = float(vol.array.astype(np.float64).sum())
    rows = []
    for name, sp in (("coarser grid, 6.4 mm voxels", 6.4), ("finer grid, 2.4 mm voxels", 2.4)):
        n = int(round(64 * 4.8 / sp))
        ref = Geometry((n, n, n), (sp, sp, sp), (-151.2 - 2.4 + sp / 2,) * 3, (1, 0, 0, 0, 1, 0, 0, 0, 1))
        out = rg.resample_volume(vol, ref, None)
        rows.append({"case": name, "total_change_pct": 100 * (float(out.array.astype(np.float64).sum()) / total - 1)})
    for name, scale in (("affine, scale 0.9 (moving image smaller)", 0.9), ("affine, scale 1.1", 1.1),
                        ("rigid, 8 deg rotation + 7 mm shift", 1.0)):
        t = sitk.AffineTransform(3)
        r = sitk.Euler3DTransform()
        r.SetRotation(0.0, 0.0, np.radians(8.0 if scale == 1.0 else 0.0))
        t.SetMatrix((np.array(r.GetMatrix()).reshape(3, 3) * scale).ravel().tolist())
        t.SetTranslation((7.0, 0.0, 0.0) if scale == 1.0 else (0.0, 0.0, 0.0))
        out = rg.resample_volume(vol, vol.geometry, t)
        rows.append({"case": name, "total_change_pct": 100 * (float(out.array.astype(np.float64).sum()) / total - 1)})
    return rows


def part_c() -> list[dict]:
    rows = []
    for nuc, th_h, t_min in (("Lu-177", 159.53, 30), ("Tc-99m", 6.0067, 20), ("I-131", 192.5, 30)):
        lt = np.log(2) / th_h * t_min / 60
        rows.append({"nuclide": nuc, "acquisition_min": t_min, "underestimate_pct": round(100 * (lt / (1 - np.exp(-lt)) - 1), 2)})
    return rows


def main():
    vol, sph = phantom()
    a, b, c = part_a(vol, sph), part_b(vol), part_c()
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = ""
    out = ROOT / "validation" / "spect_counts"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"pysegment_commit": commit, "python": platform.python_version(), "sitk": sitk.Version_VersionString(),
               "counts_to_activity": a, "count_conservation": b, "decay_during_acquisition_not_corrected": c}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
