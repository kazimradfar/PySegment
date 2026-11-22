"""Validation of the PySegment registration module (rsw/registration.py) on real PET, CT and MR images.

Data: IBSI validation data set (51 soft-tissue sarcoma patients from TCIA, CC BY 3.0): for every patient a CT,
an FDG PET (same hybrid PET/CT session, SUV) and a T1-weighted MR that the data set curators brought into the CT
space, each cropped 5 cm around the tumour, with the tumour (GTV) mask. Images are NOT stored in the repository.

Experiments (all with the default settings of the registration dialog: metric auto, start auto, 3 levels,
500 iterations, which stop earlier at convergence):

A. Known transform, same modality (exact ground truth). A random transform D is applied to a real image (cubic
   B-spline interpolation, noise added, so that the moving image is not an exact copy) and PySegment must recover it.
   CT->CT and PET->PET: rigid, affine, B-spline. Target registration error (TRE) at the tumour voxels:
   |D(R(q)) - q| for fixed points q, where R is the estimated transform (fixed -> moving).
B. Hybrid PET/CT: the moving PET/CT study is the same study moved by D; the CT-to-CT transform is applied to the PET
   (register_hybrid). TRE as in A, and tumour SUV after resampling compared with the true transform.
C. Different modalities (PET->CT, MR->CT). The reference alignment (scanner for PET, curators for MR) is not an
   exact ground truth (patient motion between scans; for the MR another session and limb position). Reported:
   (1) consistency: registration of the original pair (R0) and of the pair misaligned by a random rigid D (R1)
       must end at the same place, |D(R1(q)) - R0(q)|;
   (2) distance from the reference alignment, |R0(q) - q|, descriptive only.
D. Landmarks: six landmark pairs (tumour centre and five spread points) with a localisation error, rigid fit,
   TRE at the tumour centre compared with the prediction of Fitzpatrick et al. (IEEE TMI 1998;17:694-702).

Reproduce (several hours on two cores; B-spline CT->CT only for the first --bspline-ct patients because it is slow;
results are kept per patient in <out>/raw, so an interrupted run continues where it stopped):
    python tools/validate_registration.py --data <theibsi/data_sets>/ibsi_validation/nifti --jobs 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsw import registration as rg  # noqa: E402
from rsw.io import Geometry, Volume, load_volume  # noqa: E402

SETTINGS = dict(metric="auto", init="auto", levels=3, iterations=500)       # registration dialog defaults
N_POINTS = 2000
FLE_MM = 1.0                     # landmark localisation error, SD per axis (mm), on both images
N_LANDMARK_TRIALS = 50


# ---------------------------------------------------------------------------------------------- data
def load_case(folder: Path):
    ct = load_volume(folder / "CT_image.nii.gz", kind="CT", name="CT")
    pet = load_volume(folder / "PET_image.nii.gz", kind="PET", name="PET")
    pet.unit = "SUV"
    mr = load_volume(folder / "MR_T1_image.nii.gz", kind="MR", name="MR")
    masks = {k: sitk.GetArrayFromImage(sitk.ReadImage(str(folder / f"{k}_mask.nii.gz"))).astype(np.uint8)
             for k in ("CT", "PET")}
    return ct, pet, mr, masks


def voxel_points(mask: np.ndarray, geo: Geometry, n: int, rng) -> np.ndarray:
    """Physical coordinates (mm) of up to n voxel centres of a mask (z, y, x array)."""
    idx = np.argwhere(mask)
    if len(idx) > n:
        idx = idx[rng.choice(len(idx), n, replace=False)]
    ref = geo.reference()
    return np.array([ref.TransformContinuousIndexToPhysicalPoint([float(i[2]), float(i[1]), float(i[0])]) for i in idx])


def tp(t: sitk.Transform, pts: np.ndarray) -> np.ndarray:
    return np.array([t.TransformPoint(tuple(map(float, p))) for p in pts])


def stats(e: np.ndarray, prefix: str) -> dict:
    return {f"{prefix}_mean": float(e.mean()), f"{prefix}_median": float(np.median(e)),
            f"{prefix}_p95": float(np.percentile(e, 95)), f"{prefix}_max": float(e.max())}


# ---------------------------------------------------------------------------------------------- known transforms
def rigid_tfm(rng, centre, rot_deg=10.0, trans_mm=15.0) -> sitk.Euler3DTransform:
    t = sitk.Euler3DTransform()
    t.SetCenter(tuple(map(float, centre)))
    t.SetRotation(*np.radians(rng.uniform(-rot_deg, rot_deg, 3)))
    t.SetTranslation(tuple(rng.uniform(-trans_mm, trans_mm, 3)))
    return t


def affine_tfm(rng, centre) -> sitk.AffineTransform:
    r = rigid_tfm(rng, centre)
    a = np.diag(rng.uniform(0.95, 1.05, 3))
    a[np.triu_indices(3, 1)] = rng.uniform(-0.03, 0.03, 3)
    t = sitk.AffineTransform(3)
    t.SetCenter(r.GetCenter())
    t.SetMatrix((np.array(r.GetMatrix()).reshape(3, 3) @ a).ravel().tolist())
    t.SetTranslation(r.GetTranslation())
    return t


def deform_tfm(rng, centre, domain: sitk.Image) -> sitk.CompositeTransform:
    """Small rigid motion followed by a smooth B-spline deformation (control points displaced by N(0, 3 mm),
    limited to 6 mm; 3 x 3 x 3 mesh over the image, so that the deformation is invertible)."""
    r = rigid_tfm(rng, centre, rot_deg=5.0, trans_mm=8.0)
    bs = sitk.BSplineTransformInitializer(domain, [3, 3, 3], order=3)
    bs.SetParameters(np.clip(rng.normal(0, 3.0, len(bs.GetParameters())), -6, 6).tolist())
    c = sitk.CompositeTransform(3)
    c.AddTransform(r)
    c.AddTransform(bs)
    return c


def moved(vol: Volume, d: sitk.Transform, rng, noise: bool = True) -> Volume:
    """vol seen through D on its own grid: moved(p) = vol(D(p)). Cubic B-spline interpolation; noise optional."""
    img = rg.to_image(vol)
    arr = sitk.GetArrayFromImage(sitk.Resample(img, img, d, sitk.sitkBSpline, float(vol.array.min()), sitk.sitkFloat32))
    if vol.kind == "PET":
        arr = np.clip(arr, 0, None)
        if noise:
            arr = np.clip(arr + rng.normal(0, 1, arr.shape) * (0.05 * arr + 0.02), 0, None)
    elif noise and vol.kind == "CT":
        arr = arr + rng.normal(0, 10.0, arr.shape)
    elif noise:
        arr = arr + rng.normal(0, 0.01 * float(np.percentile(vol.array, 99)), arr.shape)
    return Volume(vol.name + "_moved", arr.astype(np.float32), vol.geometry, vol.kind, vol.unit)


def moved_mask(mask: np.ndarray, geo: Geometry, d: sitk.Transform) -> np.ndarray:
    img = geo.apply(sitk.GetImageFromArray(mask.astype(np.uint8)))
    return sitk.GetArrayFromImage(sitk.Resample(img, img, d, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8))


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def suv_stats(vol: Volume, mask: np.ndarray) -> dict:
    v = vol.array[mask.astype(bool)]
    return {"suv_mean": float(v.mean()), "suv_max": float(v.max()), "total_suv_ml": float(v.sum() * vol.geometry.voxel_ml)}


def pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / b if b else float("nan")


# ---------------------------------------------------------------------------------------------- landmarks
def fitzpatrick_tre2(points: np.ndarray, target: np.ndarray, fle2: float) -> float:
    """Expected squared TRE at ``target`` for a rigid point fit (Fitzpatrick, West & Maurer 1998):
    TRE^2 = FLE^2 / N * (1 + 1/3 * sum_k d_k^2 / f_k^2), FLE^2 = expected squared localisation error of a pair."""
    c = points.mean(0)
    x = points - c
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    f2 = ((x @ vt.T) ** 2).sum(0) / len(points)            # mean squared distance from each principal axis ...
    f2 = np.array([f2.sum() - f2[k] for k in range(3)])     # ... = rms distance of the points FROM axis k
    r = (target - c) @ vt.T
    d2 = np.array([r.dot(r) - r[k] ** 2 for k in range(3)])  # squared distance of the target from axis k
    return float(fle2 / len(points) * (1 + (d2 / f2).sum() / 3))


def spread_points(pts: np.ndarray, start: np.ndarray, n: int) -> np.ndarray:
    """Farthest-point sampling: n points from pts, the first chosen far from ``start``."""
    chosen = [start]
    for _ in range(n):
        d = np.min(np.linalg.norm(pts[:, None, :] - np.array(chosen)[None], axis=2), axis=1)
        chosen.append(pts[int(np.argmax(d))])
    return np.array(chosen[1:])


# ---------------------------------------------------------------------------------------------- one patient
def run_case(args) -> dict:
    folder, index, seed, with_bspline_ct, only = args
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    pid = folder.name
    rng = np.random.default_rng([seed, index])
    ct, pet, mr, masks = load_case(folder)
    q_ct = voxel_points(masks["CT"], ct.geometry, N_POINTS, rng)          # tumour points (CT grid)
    q_pet = voxel_points(masks["PET"], pet.geometry, N_POINTS, rng)       # tumour points (PET grid)
    centre = q_ct.mean(0)
    rows, quant, landmarks = [], [], []

    def record(pair, mode, d, res, q, seconds, extra=None):
        e = np.linalg.norm(tp(d, tp(res.transform, q)) - q, axis=1)
        e0 = np.linalg.norm(tp(d, q) - q, axis=1)
        row = {"patient": pid, "experiment": "known transform", "pair": pair, "mode": mode,
               **stats(e0, "initial_tre"), **stats(e, "tre"), "seconds": seconds, "init": res.init,
               "kept_start": res.kept_start, "warnings": " | ".join(w.split(":")[0] for w in res.warnings),
               "folding_fraction": (res.jacobian or {}).get("fraction_nonpositive", float("nan"))}
        row.update(extra or {})
        rows.append(row)
        return row

    # ---- A: same modality, exact ground truth
    for vol, q, mask, kind in [(ct, q_ct, masks["CT"], "CT"), (pet, q_pet, masks["PET"], "PET")]:
        for mode in ("rigid", "affine", "bspline"):
            if only and f"{kind}-{mode}" not in only:
                continue
            if mode == "bspline" and kind == "CT" and not with_bspline_ct:
                continue
            d = {"rigid": rigid_tfm, "affine": affine_tfm}.get(mode, None)
            d = d(rng, centre) if d else deform_tfm(rng, centre, rg.to_image(vol))
            mov = moved(vol, d, rng)
            t0 = time.perf_counter()
            if kind == "CT" and mode == "rigid":
                # B: hybrid PET/CT - the whole PET/CT study is moved by D; CT->CT transform applied to the PET
                pet_mov_clean = moved(pet, d, rng, noise=False)
                res, pet_on_ct = rg.register_hybrid(ct, mov, pet_mov_clean, mode=mode, **SETTINGS)
            else:
                res = rg.register(vol, mov, mode=mode, **SETTINGS)
            sec = time.perf_counter() - t0
            extra = {}
            if mode != "bspline":
                dinv = d.GetInverse()
                mm = moved_mask(mask, vol.geometry, d)
                extra["dice_vs_true"] = dice(rg.propagate_mask(mm, vol.geometry, vol.geometry, res.transform),
                                             rg.propagate_mask(mm, vol.geometry, vol.geometry, dinv))
            record(f"{kind}->{kind}", mode, d, res, q, sec, extra)
            if kind == "CT" and mode == "rigid":
                q_ref = rg.resample_volume(pet, ct.geometry)                       # PET on the CT grid, no motion
                true = rg.resample_volume(pet_mov_clean, ct.geometry, dinv)
                quant.append({"patient": pid, "path": "hybrid PET/CT (CT->CT applied to PET)", "mode": mode,
                              **{f"{k}_vs_true_pct": pct(v, suv_stats(true, masks["CT"])[k])
                                 for k, v in suv_stats(pet_on_ct, masks["CT"]).items()},
                              **{f"{k}_true_vs_original_pct": pct(v, suv_stats(q_ref, masks["CT"])[k])
                                 for k, v in suv_stats(true, masks["CT"]).items()}})
            if kind == "PET" and mode in ("rigid", "affine"):
                clean = moved(pet, d, rng, noise=False)
                est = rg.resample_volume(clean, pet.geometry, res.transform)
                true = rg.resample_volume(clean, pet.geometry, d.GetInverse())
                quant.append({"patient": pid, "path": "PET->PET", "mode": mode,
                              **{f"{k}_vs_true_pct": pct(v, suv_stats(true, masks["PET"])[k])
                                 for k, v in suv_stats(est, masks["PET"]).items()},
                              **{f"{k}_true_vs_original_pct": pct(v, suv_stats(pet, masks["PET"])[k])
                                 for k, v in suv_stats(true, masks["PET"]).items()}})

    # ---- C: different modalities - consistency and distance from the reference alignment
    for vol, kind in [(pet, "PET"), (mr, "MR")]:
        if only and f"{kind}->CT" not in only:
            continue
        t0 = time.perf_counter()
        r0 = rg.register(ct, moved(vol, sitk.Euler3DTransform(), rng), mode="rigid", **SETTINGS)
        s0 = time.perf_counter() - t0
        d = rigid_tfm(rng, centre)
        t0 = time.perf_counter()
        r1 = rg.register(ct, moved(vol, d, rng), mode="rigid", **SETTINGS)
        s1 = time.perf_counter() - t0
        p0 = tp(r0.transform, q_ct)
        p1 = tp(d, tp(r1.transform, q_ct))
        rows.append({"patient": pid, "experiment": "different modalities", "pair": f"{kind}->CT", "mode": "rigid",
                     **stats(np.linalg.norm(tp(d, q_ct) - q_ct, axis=1), "initial_tre"),
                     **stats(np.linalg.norm(p1 - p0, axis=1), "consistency"),
                     **stats(np.linalg.norm(p0 - q_ct, axis=1), "reference_distance"),
                     **stats(np.linalg.norm(p1 - q_ct, axis=1), "reference_distance_misaligned"),
                     "seconds": (s0 + s1) / 2, "init": r1.init, "kept_start": bool(r0.kept_start or r1.kept_start),
                     "warnings": " | ".join(w.split(":")[0] for w in r0.warnings + r1.warnings)})
        if kind == "PET":                         # tumour SUV after direct PET->CT registration (aligned start)
            on_ct = rg.resample_volume(pet, ct.geometry, r0.transform)
            ref = rg.resample_volume(pet, ct.geometry)
            quant.append({"patient": pid, "path": "PET->CT direct (aligned start, vs scanner alignment)", "mode": "rigid",
                          **{f"{k}_vs_true_pct": pct(v, suv_stats(ref, masks["CT"])[k])
                             for k, v in suv_stats(on_ct, masks["CT"]).items()}})

    # ---- D: landmarks
    if not only or "landmarks" in only:
        body = voxel_points(ct.array > -300, ct.geometry, 4000, rng)
        fixed_pts = np.vstack([centre, spread_points(body, centre, 5)])
        target = centre
        for trial in range(N_LANDMARK_TRIALS):
            d = rigid_tfm(rng, centre)
            moving_pts = tp(d.GetInverse(), fixed_pts)
            fp = fixed_pts + rng.normal(0, FLE_MM, fixed_pts.shape)
            mp = moving_pts + rng.normal(0, FLE_MM, moving_pts.shape)
            t = rg.landmark_transform(fp, mp)
            fre = float(np.sqrt(np.mean(np.sum((tp(t, fp) - mp) ** 2, axis=1))))
            tre = float(np.linalg.norm(np.array(d.TransformPoint(t.TransformPoint(tuple(target)))) - target))
            landmarks.append({"patient": pid, "trial": trial, "fre_rms": fre, "tre_target": tre,
                              "predicted_tre_rms": np.sqrt(fitzpatrick_tre2(fixed_pts, target, 2 * 3 * FLE_MM ** 2))})
    return {"rows": rows, "quant": quant, "landmarks": landmarks}


# ---------------------------------------------------------------------------------------------- summary
def q(s: pd.Series) -> str:
    return f"{s.median():.2f} [{s.quantile(0.25):.2f}-{s.quantile(0.75):.2f}]"


def summarise(rows: pd.DataFrame, quant: pd.DataFrame, lm: pd.DataFrame) -> dict:
    import platform
    import subprocess
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001 - git is optional
        commit = ""
    out: dict = {"settings": SETTINGS, "n_patients": int(rows.patient.nunique()), "pysegment_commit": commit,
                 "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
                 "sitk_version": sitk.Version_VersionString(), "known_transform": {}, "different_modalities": {},
                 "quantification": {}, "landmarks": {}}
    k = rows[rows.experiment == "known transform"]
    for (pair, mode), g in k.groupby(["pair", "mode"], sort=False):
        out["known_transform"][f"{pair} {mode}"] = {
            "n": len(g), "initial_tre_mean_mm_median_iqr": q(g.initial_tre_mean), "tre_mean_mm_median_iqr": q(g.tre_mean),
            "tre_p95_mm_median": round(float(g.tre_p95.median()), 2), "tre_max_mm_worst_case": round(float(g.tre_max.max()), 2),
            "pct_cases_tre_mean_below_1mm": round(100 * float((g.tre_mean < 1).mean()), 1),
            "pct_cases_tre_mean_below_2mm": round(100 * float((g.tre_mean < 2).mean()), 1),
            "dice_vs_true_median": round(float(g.dice_vs_true.median()), 3) if "dice_vs_true" in g and g.dice_vs_true.notna().any() else None,
            "folding_cases": int((g.folding_fraction > 0).sum()), "kept_start_cases": int(g.kept_start.sum()),
            "seconds_median": round(float(g.seconds.median()), 1)}
    m = rows[rows.experiment == "different modalities"]
    for pair, g in m.groupby("pair", sort=False):
        out["different_modalities"][pair] = {
            "n": len(g), "initial_misalignment_mm_median_iqr": q(g.initial_tre_mean),
            "consistency_mm_median_iqr": q(g.consistency_mean),
            "pct_cases_consistency_below_2mm": round(100 * float((g.consistency_mean < 2).mean()), 1),
            "pct_cases_consistency_below_5mm": round(100 * float((g.consistency_mean < 5).mean()), 1),
            "reference_distance_mm_median_iqr": q(g.reference_distance_mean),
            "pct_cases_reference_distance_above_10mm": round(100 * float((g.reference_distance_mean > 10).mean()), 1),
            "kept_start_cases": int(g.kept_start.sum()), "seconds_median": round(float(g.seconds.median()), 1)}
    for (path, mode), g in quant.groupby(["path", "mode"], sort=False):
        out["quantification"][f"{path} {mode}"] = {
            c: q(g[c].abs()) for c in g.columns if c.endswith("_pct")}
    if len(lm):
        per = lm.groupby("patient").agg(tre_rms=("tre_target", lambda x: float(np.sqrt(np.mean(np.square(x))))),
                                        predicted=("predicted_tre_rms", "first"), fre=("fre_rms", "mean"))
        out["landmarks"] = {"fle_sd_per_axis_mm": FLE_MM, "pairs": 6, "trials_per_patient": N_LANDMARK_TRIALS,
                            "observed_tre_rms_mm_median_iqr": q(per.tre_rms), "predicted_tre_rms_mm_median_iqr": q(per.predicted),
                            "observed_over_predicted_median": round(float((per.tre_rms / per.predicted).median()), 3),
                            "fre_rms_mm_median": round(float(per.fre.median()), 2)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="ibsi_validation/nifti folder of github.com/theibsi/data_sets")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "validation" / "registration"))
    ap.add_argument("--patients", type=int, default=0, help="first N patients only (0 = all)")
    ap.add_argument("--bspline-ct", type=int, default=10, help="B-spline CT->CT for the first N patients (slow)")
    ap.add_argument("--only", default="", help="comma list, e.g. CT-rigid,PET-bspline,PET->CT,landmarks")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--summary-only", action="store_true", help="only rebuild the tables from the per-patient files")
    a = ap.parse_args(argv)
    folders = sorted(p for p in Path(a.data).iterdir() if p.is_dir() and (p / "CT_image.nii.gz").exists())
    if a.patients:
        folders = folders[: a.patients]
    only = set(filter(None, a.only.split(",")))
    out = Path(a.out)
    raw = out / "raw"                      # one file per patient: an interrupted run continues where it stopped
    raw.mkdir(parents=True, exist_ok=True)
    jobs = [(f, i, a.seed, i < a.bspline_ct, only) for i, f in enumerate(folders)
            if a.summary_only is False and not (raw / f"{f.name}.json").exists()]
    t0 = time.time()
    if jobs:
        with Pool(a.jobs) as pool:
            for n, r in enumerate(pool.imap_unordered(run_case, jobs), 1):
                pid = (r["rows"] or r["landmarks"])[0]["patient"]
                (raw / f"{pid}.json").write_text(json.dumps(r))
                print(f"{pid}: {n}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
    results = [json.loads((raw / f"{f.name}.json").read_text()) for f in folders if (raw / f"{f.name}.json").exists()]
    rows = pd.DataFrame([x for r in results for x in r["rows"]]).sort_values(["experiment", "pair", "mode", "patient"])
    quant = pd.DataFrame([x for r in results for x in r["quant"]])
    lm = pd.DataFrame([x for r in results for x in r["landmarks"]])
    rows.round(4).to_csv(out / "cases.csv", index=False)
    if len(quant):
        quant.sort_values(["path", "mode", "patient"]).round(4).to_csv(out / "quantification.csv", index=False)
    if len(lm):
        lm.sort_values(["patient", "trial"]).round(4).to_csv(out / "landmarks.csv", index=False)
    summary = summarise(rows, quant, lm)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
