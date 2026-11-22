"""Validation of PySegment SUV computation and ROI statistics with the QIBA FDG-PET/CT Digital Reference Objects.

DRO (20130529 female, 20130528 male; Pierce LA et al., Radiology 2015;277:538-545): the PET DICOM files are read by
PySegment (SUVbw from the DICOM header), the six reference ROIs of the DRO user guide are placed automatically on
slice 40 with PySegment's own ROI shapes (25 mm circles; ROI 6 a 25 mm sphere) and the statistics (max, min, mean,
SD) are compared with the acceptance bounds for software that reports two or more decimals (user guide, tables 2
and 4).

DROe (20140714): (1) SUVpeak of the two blobs in PET slice 49, compared with an independent brute-force reference
(mean of every 1 mL sphere centred on a voxel of the blob, PERCIST definition); (2) PET/CT fusion: the CT alignment
stripes and the PET alignment stripes of the PET volume brought onto the CT grid by PySegment must coincide.

Reproduce: python tools/validate_qiba_dro.py --dro <DRO_20130529 folder> --droe <DROe_20140714 folder>
(packages from https://depts.washington.edu/petctdro/; images are not stored in the repository).
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsw import measure, seg  # noqa: E402
from rsw.seg import RegionTooLarge, region_grow  # noqa: E402
from rsw.io import load_volume, resample_to  # noqa: E402
from rsw.roi import circle_region  # noqa: E402

RADIUS_MM = 12.5                     # 25 mm diameter circles and sphere (DRO report)
# Acceptance bounds for software with two or more decimals (DRO user guide tables 2 and 4; same for both DROs)
BOUNDS = {1: {"max": (3.995, 4.005), "min": (0.635, 0.645), "mean": (1.295, 1.505), "sd": (0.875, 1.095)},
          2: {"max": (3.995, 4.005), "min": (3.995, 4.005), "mean": (3.995, 4.005), "sd": (-0.005, 0.005)},
          3: {"max": (4.105, 4.115), "min": (0.995, 1.005), "mean": (1.015, 1.035), "sd": (0.235, 0.325)},
          4: {"max": (0.995, 1.005), "min": (-0.115, -0.105), "mean": (0.985, 0.995), "sd": (0.085, 0.115)},
          5: {"max": (0.895, 0.905), "min": (0.095, 0.105), "mean": (0.465, 0.535), "sd": (0.395, 0.405)},
          6: {"max": (0.895, 0.905), "min": (0.095, 0.105), "mean": (0.475, 0.525), "sd": (0.395, 0.405)}}
ROI_TEXT = {1: "circle, smallest hot sphere", 2: "circle, largest hot sphere", 3: "circle, hot test voxel",
            4: "circle, cold test voxel", 5: "circle, 2D test pattern", 6: "sphere, 3D test pattern"}


# ------------------------------------------------------------------------------------------------- DRO
def find_objects(a: np.ndarray, z: int) -> dict:
    """Centres (y, x) on slice z of the hot/cold test voxels, the smallest and largest hot sphere and the two
    checkerboard patterns (the 3D one continues on the neighbouring slices)."""
    s = a[z]
    hot = np.unravel_index(np.argmax(np.where(np.isclose(s, 4.11, atol=1e-3), 1, 0)), s.shape)
    cold = np.unravel_index(np.argmax(np.where(np.isclose(s, -0.11, atol=1e-3), 1, 0)), s.shape)
    spheres = s >= 3.5
    spheres[hot] = False
    lab, n = ndimage.label(spheres)
    sizes = ndimage.sum(spheres, lab, range(1, n + 1))
    cents = ndimage.center_of_mass(spheres, lab, range(1, n + 1))
    order = np.argsort(sizes)
    checker = np.isclose(s, 0.9, atol=1e-4)
    lab2, n2 = ndimage.label(checker, structure=np.ones((3, 3)))
    sz2 = ndimage.sum(checker, lab2, range(1, n2 + 1))
    pats = [i + 1 for i in np.argsort(-sz2)[:2]]
    out = {"hot_voxel": tuple(map(int, hot)), "cold_voxel": tuple(map(int, cold)), "n_spheres_on_slice": int(n),
           "smallest_sphere": tuple(int(round(v)) for v in cents[order[0]]),
           "largest_sphere": tuple(int(round(v)) for v in cents[order[-1]])}
    for p in pats:
        yx = ndimage.center_of_mass(checker, lab2, p)
        yy, xx = np.nonzero(lab2 == p)
        depth = sum(bool(np.isclose(a[zz, yy, xx], 0.9, atol=1e-4).any() or np.isclose(a[zz, yy, xx], 0.1, atol=1e-4).all())
                    for zz in (z - 3, z + 3))
        key = "pattern_3d" if depth == 2 else "pattern_2d"
        out[key] = tuple(int(round(v)) for v in yx)
    return out


def dro_case(folder: Path, label: str) -> list[dict]:
    pet_dir = next(p for p in folder.iterdir() if p.is_dir() and "PET" in p.name)
    vol = load_volume(pet_dir)
    a = vol.array
    sp_zyx = (vol.geometry.spacing[2], vol.geometry.spacing[1], vol.geometry.spacing[0])
    z = int(np.argwhere(np.isclose(a, 4.11, atol=1e-3))[0][0])            # slice 40 holds the test voxels
    obj = find_objects(a, z)
    centres = {1: obj["smallest_sphere"], 2: obj["largest_sphere"], 3: obj["hot_voxel"], 4: obj["cold_voxel"],
               5: obj["pattern_2d"], 6: obj["pattern_3d"]}
    rows = []
    for roi, (y, x) in centres.items():
        if roi == 6:
            region = np.zeros(a.shape, bool)
            r, box = seg.brush_region(a.shape, (z, y, x), RADIUS_MM, vol.geometry.spacing, "axial", True)
            region[box] = r
        else:
            region = np.zeros(a.shape, bool)
            region[z] = circle_region(a.shape[1:], y, x, RADIUS_MM, (sp_zyx[1], sp_zyx[2]))
        st = measure.roi_stats(a, region, sp_zyx)
        row = {"dro": label, "roi": roi, "roi_definition": ROI_TEXT[roi], "slice": z + 1, "centre_yx": f"{y},{x}",
               "voxels": st["voxels"],
               "size": f"{st['voxels'] * sp_zyx[1] * sp_zyx[2]:.0f} mm2" if roi < 6 else f"{st['volume_ml'] * 1000:.0f} mm3"}
        ok_all = True
        for k in ("max", "min", "mean", "sd"):
            lo, hi = BOUNDS[roi][k]
            v = round(st[k], 3)
            ok = lo <= v <= hi
            ok_all &= ok
            row[k] = v
            row[f"{k}_bounds"] = f"{lo}-{hi}"
            row[f"{k}_pass"] = ok
        row["all_pass"] = ok_all
        rows.append(row)
    rows.append({"dro": label, "roi": "SUVbw", "roi_definition": "body background (expected 1.00), hot spheres (4.00)",
                 "mean": float(np.median(a[a > 0.95][np.isclose(a[a > 0.95], 1.0, atol=0.02)])),
                 "max": float(np.median(a[np.isclose(a, 4.0, atol=0.02)])),
                 "size": f"weight {vol.info.get('weight_kg')} kg, {vol.info.get('injected_MBq')} MBq, "
                         f"{vol.info.get('uptake_min')} min"})
    return rows


# ------------------------------------------------------------------------------------------------- DROe
def peak_reference(a: np.ndarray, region: np.ndarray, sp_zyx) -> float:
    """Independent SUVpeak: for every voxel of the region, the plain mean over all voxels whose centres lie within
    the 1 mL sphere around it (voxels outside the image count as 0). Brute force, no convolution."""
    r = (3 * 1000.0 / (4 * np.pi)) ** (1 / 3)
    n = [int(np.ceil(r / s)) for s in sp_zyx]
    offs = [(dz, dy, dx) for dz in range(-n[0], n[0] + 1) for dy in range(-n[1], n[1] + 1) for dx in range(-n[2], n[2] + 1)
            if (dz * sp_zyx[0]) ** 2 + (dy * sp_zyx[1]) ** 2 + (dx * sp_zyx[2]) ** 2 <= r * r]
    best = -np.inf
    for c in np.argwhere(region):
        tot = 0.0
        for o in offs:
            p = c + o
            if all(0 <= p[i] < a.shape[i] for i in range(3)):
                tot += float(a[tuple(p)])
        best = max(best, tot / len(offs))
    return best


def stripes(mask: np.ndarray, min_len: int) -> dict:
    """Long thin components of a 2D mask (alignment stripes): {'vertical': [x centres], 'horizontal': [y centres]}
    in voxels; dots and corner markers are ignored."""
    lab, n = ndimage.label(mask)
    out = {"vertical": [], "horizontal": []}
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        c = ndimage.center_of_mass(lab == i)
        if h >= min_len and h > 3 * w:
            out["vertical"].append(float(c[1]))
        elif w >= min_len and w > 3 * h:
            out["horizontal"].append(float(c[0]))
    return {k: sorted(v) for k, v in out.items()}


def droe_case(folder: Path) -> dict:
    em = load_volume(next(p for p in folder.iterdir() if p.is_dir() and "emission" in p.name))
    tr = load_volume(next(p for p in folder.iterdir() if p.is_dir() and "transmission" in p.name), kind="CT")
    a = em.array
    sp = (em.geometry.spacing[2], em.geometry.spacing[1], em.geometry.spacing[0])
    out = {"pet_unit": em.unit, "pet_shape_zyx": list(a.shape), "ct_shape_zyx": list(tr.array.shape),
           "pet_spacing_xyz": list(em.geometry.spacing), "ct_spacing_xyz": list(tr.geometry.spacing)}
    # SUVpeak of the two blobs on axial slice 49 (background SUV 1.0)
    z = 48
    above = a > 1.0 + 1e-3
    lab_s, ns = ndimage.label(above[z], structure=np.ones((3, 3)))
    sizes = ndimage.sum(above[z], lab_s, range(1, ns + 1))
    blobs = [i + 1 for i in np.argsort(-sizes)[:2]]
    blobs.sort(key=lambda i: ndimage.center_of_mass(above[z], lab_s, i)[1])          # left, right
    lab3, _ = ndimage.label(above)
    peaks = []
    for side, i in zip(("left", "right"), blobs):
        yy, xx = np.nonzero(lab_s == i)
        ids = np.unique(lab3[z, yy, xx])
        region = np.isin(lab3, ids[ids > 0])
        p_pys = measure.suv_peak(a, region, sp)
        p_ref = peak_reference(a, region, sp)
        k = measure._kernel(sp, 1.0)
        mz = np.unravel_index(np.argmax(np.where(region, a, -np.inf)), a.shape)
        at_max = float(ndimage.convolve(a.astype(np.float64), k / k.sum(), mode="constant")[mz])
        peaks.append({"blob": side, "voxels_3d": int(region.sum()), "max": round(float(a[region].max()), 4),
                      "sphere_voxels": int(k.sum()), "sphere_volume_ml": round(float(k.sum() * np.prod(sp)) / 1000, 3),
                      "sphere_at_hottest_voxel": round(at_max, 4),
                      "suv_peak_pysegment": round(p_pys, 4), "suv_peak_reference": round(p_ref, 4),
                      "difference": round(p_pys - p_ref, 6)})
    out["suv_peak_1ml"] = peaks
    # PET/CT fusion: alignment stripes of CT slice 64 against the PET brought onto the CT grid by PySegment
    pet_on_ct = resample_to(em, tr.geometry, linear=True).array
    zc = 63
    ct_m = tr.array[zc] >= 999.0
    pet_s = pet_on_ct[zc]
    pet_m = pet_s >= 0.5 * (pet_s.max() + 1.0)
    px = tr.geometry.spacing[0]
    c, p = stripes(ct_m, 20), stripes(pet_m, 10)
    res = {}
    for k in ("vertical", "horizontal"):
        d = [round((pi - ci) * px, 3) for ci, pi in zip(c[k], p[k])] if len(c[k]) == len(p[k]) else None
        res[f"{k} stripes"] = {"ct": len(c[k]), "pet": len(p[k]), "offset_mm": d}
    out["fusion_ct_slice_64"] = res
    return out


# ------------------------------------------------------------------------------------------------- DROmtv
def mtv_objects(a: np.ndarray, bkg: float = 1.0) -> list[dict]:
    """Objects of the noise-free DROmtv (above background) with their true volume. The noise-free voxels hold
    background + (level - background) x occupied fraction, so the true volume is the sum of the fractions times the
    voxel volume. Only for objects with a uniform level (most voxels exactly at the maximum); the smooth blob and
    the thin lobed object have no single level, so no true volume is given for them."""
    lab, _ = ndimage.label(ndimage.binary_dilation(a > bkg + 0.05, iterations=2))
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        m = lab == i
        v = a[m].astype(np.float64)
        level = float(v.max())
        n_level = int(np.isclose(v, level, atol=1e-3).sum())
        if (v > bkg + 0.05).sum() < 20:
            continue                                            # fragments of the thin lobed object
        uniform = n_level >= 0.2 * (v > bkg + 0.05).sum()
        frac = np.clip((v - bkg) / (level - bkg), 0, 1)
        centre = tuple(int(round(c)) for c in ndimage.center_of_mass(m))
        out.append({"centre_zyx": centre, "level": round(level, 3), "uniform": bool(uniform), "_frac": float(frac.sum())})
    return out


def dromtv_case(folder: Path) -> list[dict]:
    clean_dir = next(p for p in folder.iterdir() if p.is_dir() and "Free" in p.name)
    noisy_dir = next(p for p in folder.iterdir() if p.is_dir() and "Noise_n" in p.name)
    clean, noisy = load_volume(clean_dir), load_volume(noisy_dir)
    vox_ml = float(np.prod(clean.geometry.spacing)) / 1000.0
    objs = mtv_objects(clean.array)
    # names by content: cube = uniform level 4 with no partial voxels; spheres sorted by volume per contrast
    for o in objs:
        o["true_volume_ml"] = round(o.pop("_frac") * vox_ml, 3) if o["uniform"] else None
    hi = sorted([o for o in objs if o["uniform"] and o["level"] > 3 and o["true_volume_ml"] < 30
                 and o["centre_zyx"][0] == objs[-1]["centre_zyx"][0]], key=lambda o: o["true_volume_ml"])
    lo = sorted([o for o in objs if o["uniform"] and o["level"] < 3], key=lambda o: o["true_volume_ml"])
    for k, o in enumerate(hi, 1):
        o["object"] = f"sphere {k}, 4:1"
    for k, o in enumerate(lo, 1):
        o["object"] = f"sphere {k}, 2:1"
    rest = [o for o in objs if "object" not in o]
    for o in rest:
        if o["uniform"] and o["true_volume_ml"] and o["true_volume_ml"] > 40:
            o["object"] = "cube"
        elif o["uniform"]:
            o["object"] = "irregular"
        else:
            o["object"] = "blob" if max(o["centre_zyx"][1], 0) > 80 else "lobed"
    methods = {"region grow 41 % of max": ("percent", 41.0), "region grow 50 % of max": ("percent", 50.0),
               "region grow SUV 2.5": ("absolute", 2.5),
               "region grow, half-way between background and level": ("absolute", None)}
    rows = []
    for vol_name, vol in (("noise-free", clean), ("noisy, 6 mm FWHM", noisy)):
        for o in objs:
            for mname, (mode, val) in methods.items():
                v = (1.0 + o["level"]) / 2 if val is None else val
                try:
                    reg, _, thr = region_grow(vol.array, o["centre_zyx"], vol.geometry.spacing, mode, v)
                    mtv = float(reg.sum()) * vox_ml
                    note = "" if reg.any() else "nothing above the threshold"
                except RegionTooLarge:
                    mtv, thr, note = float("nan"), float("nan"), "threshold below background: region leaks"
                row = {"image": vol_name, "object": o["object"], "level": o["level"], "true_volume_ml": o["true_volume_ml"],
                       "method": mname, "threshold": round(float(thr), 3), "mtv_ml": round(mtv, 3), "note": note}
                if o["true_volume_ml"] and mtv == mtv and mtv > 0:
                    row["error_pct"] = round(100 * (mtv / o["true_volume_ml"] - 1), 1)
                rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dro", required=True, help="DRO_20130529 folder (with the female and male subfolders)")
    ap.add_argument("--droe", default="", help="DROe_20140714 folder")
    ap.add_argument("--dromtv", default="", help="DROmtv_20160428 folder")
    ap.add_argument("--out", default=str(ROOT / "validation" / "qiba_dro"))
    a = ap.parse_args(argv)
    dro = Path(a.dro)
    rows = []
    for sub in sorted(p for p in dro.iterdir() if p.is_dir() and "DICOM" in p.name):
        rows += dro_case(sub, "female 20130529" if "female" in sub.name else "male 20130528")
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "dro_roi_results.csv", index=False)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = ""
    rois = df[df.roi.astype(str).str.isdigit()]
    summary = {"pysegment_commit": commit, "python": platform.python_version(), "numpy": np.__version__,
               "dro_values_passing": f"{int(sum(rois[f'{k}_pass'].sum() for k in ('max', 'min', 'mean', 'sd')))}/"
                                     f"{4 * len(rois)}",
               "dro_failed": [f"{r.dro} ROI {r.roi} {k}={getattr(r, k)} (bounds {getattr(r, k + '_bounds')})"
                              for r in rois.itertuples() for k in ("max", "min", "mean", "sd") if not getattr(r, k + "_pass")]}
    if a.droe:
        summary["droe"] = droe_case(Path(a.droe))
    if a.dromtv:
        mtv = pd.DataFrame(dromtv_case(Path(a.dromtv)))
        mtv.to_csv(out / "dromtv_results.csv", index=False)
        ok = mtv[mtv.error_pct.notna()]
        summary["dromtv_abs_error_pct_median_by_method"] = {
            f"{img} | {m}": round(float(g.error_pct.abs().median()), 1) for (img, m), g in ok.groupby(["image", "method"])}
        bad = mtv[mtv.note.fillna("") != ""]
        summary["dromtv_not_measured"] = {f"{img} | {m} | {n}": int(c) for (img, m, n), c in
                                          bad.groupby(["image", "method", "note"]).size().items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(df.drop(columns=[c for c in df.columns if c.endswith("_bounds")]).to_string())
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
