"""Validation of the PySegment agreement metrics (rsw/metrics.py): Dice, Jaccard, HD95 and ICC(A,1).

A. Dice and Jaccard: shapes with an exact answer (shifted cubes) and 200 random pairs of ellipsoids on anisotropic
   grids, compared with SimpleITK's LabelOverlapMeasuresImageFilter.
B. HD95: concentric spheres (every surface distance equals the difference of the radii) and the same 200 random pairs
   compared with MedPy (medpy.metric.binary.hd95), which uses the same definition (95th percentile of the pooled
   distances between the two surfaces). Surfaces: PySegment uses a 26-neighbour erosion, MedPy a 6-neighbour one;
   the comparison is made with both.
C. ICC(A,1) = ICC(2,1): the published example of Shrout & Fleiss (Psychol Bull 1979;86:420-428; 6 targets, 4 judges,
   ICC(2,1) = 0.29), and 500 random data sets (n 5-200 subjects, k 2-4 raters) compared with pingouin (ICC(A,1) with the
   McGraw & Wong confidence interval, which pingouin rounds to two decimals); Koo & Li (2016) category limits.

Reproduce (about a minute): pip install pingouin medpy; python tools/validate_agreement.py
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsw import metrics as mt  # noqa: E402

SHROUT_FLEISS = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], float)


def ellipsoid(shape, centre, radii_vox):
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return (((z - centre[0]) / radii_vox[0]) ** 2 + ((y - centre[1]) / radii_vox[1]) ** 2
            + ((x - centre[2]) / radii_vox[2]) ** 2) <= 1


def random_pairs(n, seed):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        shape = (40, 48, 48)
        sp_xyz = (float(rng.uniform(0.8, 4.0)), float(rng.uniform(0.8, 4.0)), float(rng.uniform(1.0, 5.0)))
        c = np.array([20, 24, 24]) + rng.uniform(-2, 2, 3)
        r = rng.uniform(4, 12, 3)
        a = ellipsoid(shape, c, r)
        b = ellipsoid(shape, c + rng.normal(0, 1.5, 3), r * rng.uniform(0.8, 1.2, 3))
        if a.any() and b.any():
            yield a, b, sp_xyz


def sitk_overlap(a, b, sp):
    f = sitk.LabelOverlapMeasuresImageFilter()
    ia, ib = sitk.GetImageFromArray(a.astype(np.uint8)), sitk.GetImageFromArray(b.astype(np.uint8))
    ia.SetSpacing(sp)
    ib.SetSpacing(sp)
    f.Execute(ia, ib)
    return f.GetDiceCoefficient(), f.GetJaccardCoefficient()


def part_ab():
    from medpy.metric.binary import hd95 as medpy_hd95
    exact = []
    for s in range(0, 21, 4):                                          # cube of 20 voxels shifted by s voxels
        a = np.zeros((30, 30, 50), bool)
        b = np.zeros_like(a)
        a[5:25, 5:25, 5:25] = True
        b[5:25, 5:25, 5 + s:25 + s] = True
        d_true = (20 - s) / 20
        exact.append({"case": f"cubes shifted {s} voxels", "dice": mt.dice(a, b), "dice_exact": d_true,
                      "jaccard": mt.jaccard(a, b), "jaccard_exact": d_true / (2 - d_true)})
    spheres = []
    for r_in, r_out in ((10, 12), (10, 14), (15, 20)):                 # 0.5 mm isotropic grid
        n = 2 * r_out * 2 + 10
        a = ellipsoid((n, n, n), (n / 2,) * 3, (2 * r_in,) * 3)
        b = ellipsoid((n, n, n), (n / 2,) * 3, (2 * r_out,) * 3)
        spheres.append({"case": f"concentric spheres r {r_in} and {r_out} mm (0.5 mm voxels)",
                        "hd95_mm": round(mt.hd95(a, b, (0.5, 0.5, 0.5)), 3), "expected_mm": r_out - r_in})
    rows = []
    for a, b, sp in random_pairs(200, 7):
        d_s, j_s = sitk_overlap(a, b, sp)
        sp_zyx = (sp[2], sp[1], sp[0])
        h_pys = mt.hd95(a, b, sp)
        h_med6 = float(medpy_hd95(a, b, voxelspacing=sp_zyx, connectivity=1))
        h_med26 = float(medpy_hd95(a, b, voxelspacing=sp_zyx, connectivity=3))
        rows.append({"dice": mt.dice(a, b), "dice_sitk": d_s, "jaccard": mt.jaccard(a, b), "jaccard_sitk": j_s,
                     "hd95": h_pys, "hd95_medpy_26": h_med26, "hd95_medpy_6": h_med6,
                     "voxel_mm": f"{sp[0]:.2f}x{sp[1]:.2f}x{sp[2]:.2f}"})
    return pd.DataFrame(exact), pd.DataFrame(spheres), pd.DataFrame(rows)


def part_c(seed=11):
    import pingouin as pg
    icc, lo, hi = mt.icc_a1(SHROUT_FLEISS)
    ref = icc_pingouin(SHROUT_FLEISS, pg)
    shrout = {"pysegment": [round(icc, 4), round(lo, 4), round(hi, 4)], "pingouin": [round(v, 4) for v in ref],
              "published_icc_2_1": 0.29}
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(500):
        n, k = int(rng.integers(5, 201)), int(rng.integers(2, 5))
        true = rng.normal(0, 1, (n, 1))
        y = true + rng.normal(0, rng.uniform(0.05, 1.5), (n, k)) + rng.normal(0, rng.uniform(0, 0.5), (1, k))
        a = mt.icc_a1(y)
        b = icc_pingouin(y, pg)
        rows.append({"n": n, "k": k, "icc": a[0], "icc_pingouin": b[0], "ci_low": a[1], "ci_low_pingouin": b[1],
                     "ci_high": a[2], "ci_high_pingouin": b[2]})
    cats = {str(v): mt.icc_category(v) for v in (0.49, 0.5, 0.749, 0.75, 0.9, 0.901)}
    return shrout, pd.DataFrame(rows), cats


def icc_pingouin(y, pg):
    n, k = y.shape
    df = pd.DataFrame({"t": np.repeat(np.arange(n), k), "r": np.tile(np.arange(k), n), "v": y.ravel()})
    r = pg.intraclass_corr(df, targets="t", raters="r", ratings="v").set_index("Type").loc["ICC(A,1)"]
    return float(r["ICC"]), float(r["CI95"][0]), float(r["CI95"][1])


def main():
    out = ROOT / "validation" / "agreement"
    out.mkdir(parents=True, exist_ok=True)
    exact, spheres, pairs = part_ab()
    shrout, iccs, cats = part_c()
    exact.round(6).to_csv(out / "dice_exact.csv", index=False)
    spheres.to_csv(out / "hd95_spheres.csv", index=False)
    pairs.round(6).to_csv(out / "random_pairs.csv", index=False)
    iccs.round(6).to_csv(out / "icc_random.csv", index=False)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = ""
    import medpy
    import pingouin
    rel = lambda a, b: (a - b).abs() / b.abs().clip(lower=1e-12)  # noqa: E731
    summary = {
        "pysegment_commit": commit, "python": platform.python_version(), "sitk": sitk.Version_VersionString(),
        "pingouin": pingouin.__version__, "medpy": getattr(medpy, "__version__", ""),
        "dice_exact_max_abs_diff": float((exact.dice - exact.dice_exact).abs().max()),
        "jaccard_exact_max_abs_diff": float((exact.jaccard - exact.jaccard_exact).abs().max()),
        "dice_vs_sitk_max_abs_diff": float((pairs.dice - pairs.dice_sitk).abs().max()),
        "jaccard_vs_sitk_max_abs_diff": float((pairs.jaccard - pairs.jaccard_sitk).abs().max()),
        "hd95_concentric_spheres": spheres.to_dict(orient="records"),
        "hd95_vs_medpy_26_neighbours_max_abs_diff_mm": float((pairs.hd95 - pairs.hd95_medpy_26).abs().max()),
        "hd95_vs_medpy_6_neighbours_median_abs_diff_mm": float((pairs.hd95 - pairs.hd95_medpy_6).abs().median()),
        "hd95_vs_medpy_6_neighbours_max_abs_diff_mm": float((pairs.hd95 - pairs.hd95_medpy_6).abs().max()),
        "icc_shrout_fleiss": shrout,
        "icc_vs_pingouin_max_abs_diff": float((iccs.icc - iccs.icc_pingouin).abs().max()),
        "ci_low_vs_pingouin_max_abs_diff": float((iccs.ci_low - iccs.ci_low_pingouin).abs().max()),
        "ci_high_vs_pingouin_max_abs_diff": float((iccs.ci_high - iccs.ci_high_pingouin).abs().max()),
        "koo_li_categories": cats,
        "n_random_mask_pairs": len(pairs), "n_random_icc_sets": len(iccs)}
    del rel
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
