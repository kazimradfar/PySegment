"""Agreement between masks (Dice, Jaccard, volume difference, 95th-percentile Hausdorff distance) and
between readers' feature values (ICC(A,1) with the McGraw & Wong confidence interval)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage, stats


def dice(a, b) -> float:
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    s = int(a.sum()) + int(b.sum())
    return 1.0 if s == 0 else 2.0 * int(np.logical_and(a, b).sum()) / s


def jaccard(a, b) -> float:
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    u = int(np.logical_or(a, b).sum())
    return 1.0 if u == 0 else int(np.logical_and(a, b).sum()) / u


def _surface(m):
    return m & ~ndimage.binary_erosion(m, structure=np.ones((3, 3, 3), bool), border_value=0)


def hd95(a, b, spacing_xyz) -> float:
    """95th percentile of the symmetric surface distances (mm); nan if a mask is empty."""
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    if not a.any() or not b.any():
        return float("nan")
    sp = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    sa, sb = _surface(a), _surface(b)
    da = ndimage.distance_transform_edt(~sb, sampling=sp)[sa]
    db = ndimage.distance_transform_edt(~sa, sampling=sp)[sb]
    return float(np.percentile(np.concatenate([da, db]), 95))


def agreement(a, b, spacing_xyz) -> dict:
    vox = float(np.prod(spacing_xyz)) / 1000.0
    va, vb = float((np.asarray(a) > 0).sum()) * vox, float((np.asarray(b) > 0).sum()) * vox
    return {"dice": dice(a, b), "jaccard": jaccard(a, b), "hd95_mm": hd95(a, b, spacing_xyz),
            "volume_a_ml": va, "volume_b_ml": vb, "volume_diff_pct": (100.0 * (vb - va) / va) if va else float("nan")}


def icc_a1(y: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """ICC(A,1) (two-way, absolute agreement, single measure; = ICC(2,1)) with the McGraw & Wong (1996)
    confidence interval. y: (n subjects, k raters)."""
    y = np.asarray(y, float)
    n, k = y.shape
    grand = y.mean()
    ss_r = k * ((y.mean(axis=1) - grand) ** 2).sum()
    ss_c = n * ((y.mean(axis=0) - grand) ** 2).sum()
    ss_e = ((y - grand) ** 2).sum() - ss_r - ss_c
    msr, msc, mse = ss_r / (n - 1), ss_c / (k - 1), ss_e / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    if denom <= 0:
        return float("nan"), float("nan"), float("nan")
    icc = (msr - mse) / denom
    if icc >= 1 or mse == 0:
        return float(icc), float("nan"), float("nan")
    a = k * icc / (n * (1 - icc))
    b = 1 + k * icc * (n - 1) / (n * (1 - icc))
    v = (a * msc + b * mse) ** 2 / ((a * msc) ** 2 / (k - 1) + (b * mse) ** 2 / ((n - 1) * (k - 1)))
    f_s = stats.f.ppf(1 - alpha / 2, n - 1, v)
    f_i = stats.f.ppf(1 - alpha / 2, v, n - 1)
    lower = n * (msr - f_s * mse) / (f_s * (k * msc + (k * n - k - n) * mse) + n * msr)
    upper = n * (f_i * msr - mse) / (k * msc + (k * n - k - n) * mse + n * f_i * msr)
    return float(icc), float(lower), float(upper)


def icc_category(icc: float) -> str:
    """Koo & Li (2016) categories."""
    if not np.isfinite(icc):
        return "not defined"
    return ("excellent (>0.90)" if icc > 0.9 else "good (0.75–0.90)" if icc >= 0.75
            else "moderate (0.50–0.75)" if icc >= 0.5 else "poor (<0.50)")


def icc_table(a, b, id_col: str = "case"):
    """ICC per feature between two feature tables (one row per case, same feature columns)."""
    import pandas as pd
    a, b = a.set_index(id_col), b.set_index(id_col)
    common = a.index.intersection(b.index)
    feats = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c])]
    rows = []
    for f in feats:
        y = np.column_stack([a.loc[common, f].to_numpy(float), b.loc[common, f].to_numpy(float)])
        y = y[np.isfinite(y).all(axis=1)]
        icc, lo, hi = icc_a1(y) if len(y) >= 3 and np.ptp(y) > 0 else (float("nan"),) * 3
        rows.append({"feature": f, "n": len(y), "icc": icc, "ci_low": lo, "ci_high": hi, "category": icc_category(icc)})
    return pd.DataFrame(rows)
