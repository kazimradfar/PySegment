"""Measurements on the open image: ROI statistics (SUVmean / max / SD / peak), line profiles with FWHM, the PERCIST
background threshold. Pure numpy/scipy, no Qt. Arrays are the canonical (z, y, x) arrays of the session, spacing in mm."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

PEAK_ML = 1.0        # SUVpeak: mean in the 1 mL sphere centred on the hottest voxel (PERCIST uses about 1.2 cm diameter)


def _kernel(spacing_zyx, volume_ml: float) -> np.ndarray:
    r_mm = (3.0 * volume_ml * 1000.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
    n = [int(np.ceil(r_mm / s)) for s in spacing_zyx]
    z, y, x = np.ogrid[-n[0]:n[0] + 1, -n[1]:n[1] + 1, -n[2]:n[2] + 1]
    d = (z * spacing_zyx[0]) ** 2 + (y * spacing_zyx[1]) ** 2 + (x * spacing_zyx[2]) ** 2
    return d <= r_mm ** 2


def suv_peak(image: np.ndarray, region: np.ndarray, spacing_zyx, volume_ml: float = PEAK_ML) -> float:
    """Highest mean over a sphere of ``volume_ml`` whose centre lies inside the region (the sphere may extend beyond
    the region, as in PERCIST). Edge voxels outside the image count as zero activity."""
    if not region.any():
        return float("nan")
    k = _kernel(spacing_zyx, volume_ml)
    idx = np.argwhere(region)
    lo = np.maximum(idx.min(axis=0) - np.array(k.shape) // 2, 0)
    hi = np.minimum(idx.max(axis=0) + np.array(k.shape) // 2 + 1, image.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    means = ndimage.convolve(image[sl].astype(np.float64), k.astype(np.float64) / k.sum(), mode="constant", cval=0.0)
    return float(means[region[sl]].max())


def roi_stats(image: np.ndarray, region: np.ndarray, spacing_zyx) -> dict:
    """Statistics of the voxels in ``region``; volume from the voxel size."""
    vox_ml = float(np.prod(spacing_zyx)) / 1000.0
    v = image[region].astype(np.float64)
    if v.size == 0:
        return {"voxels": 0, "volume_ml": 0.0, "mean": float("nan"), "sd": float("nan"), "min": float("nan"),
                "max": float("nan"), "median": float("nan"), "peak": float("nan"), "total": float("nan")}
    return {"voxels": int(v.size), "volume_ml": v.size * vox_ml, "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0, "min": float(v.min()), "max": float(v.max()),
            "median": float(np.median(v)), "peak": suv_peak(image, region, spacing_zyx),
            "total": float(v.sum() * vox_ml)}          # sum of value x ml (SUV x ml = "SUV-volume")


def percist_threshold(ref_mean: float, ref_sd: float) -> float:
    """PERCIST (Wahl 2009) lesion threshold: 1.5 x mean of the 3 cm liver sphere + 2 SD."""
    return 1.5 * ref_mean + 2.0 * ref_sd


def line_profile(image: np.ndarray, p0, p1, spacing_zyx, step_mm: float | None = None):
    """Values along the straight line between two voxel positions (z, y, x), trilinear interpolation.
    Returns (distance_mm, values)."""
    p0, p1, sp = np.asarray(p0, float), np.asarray(p1, float), np.asarray(spacing_zyx, float)
    length = float(np.linalg.norm((p1 - p0) * sp))
    step = step_mm or float(min(spacing_zyx)) / 2.0
    n = max(int(np.ceil(length / step)) + 1, 2)
    t = np.linspace(0.0, 1.0, n)
    pts = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
    vals = ndimage.map_coordinates(image.astype(np.float64), pts.T, order=1, mode="nearest")
    return t * length, vals


def fwhm(dist: np.ndarray, vals: np.ndarray) -> float:
    """Full width at half maximum of the main peak of a profile, above the lower of the two end values (linear
    interpolation of the half-maximum crossings). NaN if the peak does not fall below half on both sides."""
    vals = np.asarray(vals, float)
    k = int(np.argmax(vals))
    base = min(vals[0], vals[-1])
    half = base + 0.5 * (vals[k] - base)
    if vals[k] <= base:
        return float("nan")

    def cross(i_from, step):
        i = i_from
        while 0 <= i + step < len(vals):
            if vals[i + step] <= half:
                a, b = vals[i], vals[i + step]
                f = (a - half) / (a - b) if a != b else 0.0
                return dist[i] + f * (dist[i + step] - dist[i])
            i += step
        return None
    left, right = cross(k, -1), cross(k, +1)
    if left is None or right is None:
        return float("nan")
    return float(right - left)


# ---------------------------------------------------------------------------------- counts -> activity, tables
def decay_factor(half_life_h: float, elapsed_h: float) -> float:
    """Factor that corrects a measurement made ``elapsed_h`` after the reference time back to the reference time."""
    return float(2.0 ** (elapsed_h / half_life_h))


def counts_to_activity_mbq(counts: float, sensitivity_cps_per_mbq: float, time_s: float, decay: float = 1.0) -> float:
    """Activity (MBq) from counts collected in ``time_s`` by a system of the given sensitivity (cps per MBq)."""
    return float(counts) / (float(sensitivity_cps_per_mbq) * float(time_s)) * decay


def is_counts_unit(unit: str) -> bool:
    return str(unit).strip().upper() in ("CNTS", "COUNTS", "CPS")


def ct_density(hu) -> np.ndarray:
    """Approximate mass density (g/mL) from CT numbers: 1 + HU/1000 up to 0 HU (air to water), then
    1 + 0.6 x HU/1000 (soft tissue to cortical bone, about 1.6 g/mL at 1000 HU). A scanner-specific calibration
    (density phantom) is more accurate; the result is used only for the mass of an ROI."""
    h = np.asarray(hu, np.float64)
    return np.clip(np.where(h <= 0, 1.0 + h / 1000.0, 1.0 + 0.6 * h / 1000.0), 0.0, None)


def roi_table(rows: list[dict], calib: dict | None = None) -> "pd.DataFrame":
    """Result table of the ROI rows. Always: image, unit, volume, voxels, mean, SD, min, max, median, peak, sum.
    SPECT in counts and a calibration (sensitivity cps/MBq, time s, optional decay factor): activity (MBq) and
    concentration (MBq/mL). PET in SUV with dose and weight known: activity concentration (kBq/mL) and activity (MBq).
    Image already in Bq/mL: activity concentration from the values themselves."""
    import pandas as pd
    calib = calib or {}
    out = []
    for r in rows:
        if r.get("kind") != "roi":
            continue
        unit, vol = r.get("unit", ""), r.get("volume_ml")
        d = {"roi": r.get("roi", ""), "image": r["image"], "unit": unit, "radius_mm": r.get("radius_mm"), "volume_ml": vol, "voxels": r.get("voxels"),
             "mean_intensity": r.get("mean"), "sd_intensity": r.get("sd"), "min_intensity": r.get("min"),
             "max_intensity": r.get("max"), "median_intensity": r.get("median"), "peak_1ml_intensity": r.get("peak")}
        if is_counts_unit(unit) and calib.get("sensitivity") and calib.get("time_s"):
            vox_ml = vol / r["voxels"] if r.get("voxels") else float("nan")
            tot_counts = r["total"] / vox_ml if vox_ml else float("nan")            # total/ml-sum -> sum of voxel counts
            act = counts_to_activity_mbq(tot_counts, calib["sensitivity"], calib["time_s"], calib.get("decay", 1.0))
            d.update({"total_counts": tot_counts, "activity_MBq": act, "concentration_MBq_per_ml": act / vol if vol else float("nan")})
        elif is_counts_unit(unit):
            vox_ml = vol / r["voxels"] if r.get("voxels") else float("nan")
            d["total_counts"] = r["total"] / vox_ml if vox_ml else float("nan")
        if r.get("density_g_ml") is not None:
            d.update({"density_g_ml": r["density_g_ml"], "mass_g": r.get("mass_g")})
        if str(unit).upper() == "SUV" and r.get("dose_MBq") and r.get("weight_kg"):
            f = r["dose_MBq"] / r["weight_kg"]       # SUV x f = kBq/mL at the scan start (dose decayed to that time)
            d.update({"concentration_kBq_per_ml": r["mean"] * f, "activity_MBq": r["mean"] * f * vol / 1000.0})
        if str(unit).upper() in ("BQML", "BQ/ML"):
            d.update({"concentration_kBq_per_ml": r["mean"] / 1000.0, "activity_MBq": r["mean"] * vol / 1e6})
        out.append(d)
    return pd.DataFrame(out)
