"""IBSI radiomic features with PyRadiomics, for a whole mask and for each connected lesion.

Settings are stored with every result table (settings.json next to the CSV), so an extraction can be
repeated exactly. Texture features are computed only for regions with at least `min_texture_voxels`
voxels after resampling (IBSI recommendation for small regions); smaller lesions get shape and first-order
features only.
"""
from __future__ import annotations

import json
import logging
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

from rsw.io import Geometry

TEXTURE = ["glcm", "glrlm", "glszm", "gldm", "ngtdm"]


@dataclass
class Settings:
    discretisation: str = "width"        # "width" (fixed bin width, e.g. SUV 0.5, HU 25) or "count"
    bin_value: float = 0.5
    resample_mm: float | None = 4.0      # isotropic; None = original voxels
    interpolator: str = "sitkBSpline"    # sitkLinear (trilinear), sitkBSpline (tricubic), sitkNearestNeighbor
    # "pyradiomics" (default, as before): PyRadiomics resamples, grid aligned with the bounding box, mask by
    # nearest neighbour. "ibsi": grid centres aligned with the image, mask interpolated (trilinear, >= 0.5),
    # as the IBSI reference manual specifies (validated: validation/ibsi1/).
    resampling: str = "pyradiomics"     # "ibsi" also computes LoG as IBSI 2 defines it (see _ibsi_log)
    log_boundary: str = "reflect"        # IBSI LoG boundary condition: reflect, nearest, wrap or constant
    round_intensities: bool = False      # IBSI: round to whole numbers after interpolation (CT in HU)
    resegment_low: float | None = 0.0    # e.g. 0 for SUV; None = off
    resegment_high: float | None = None  # upper bound of range re-segmentation, e.g. 400 HU
    outlier_sigma: float | None = None   # IBSI outlier filtering: keep mean +/- n SD (after the range)
    classes: list = field(default_factory=lambda: ["shape", "firstorder", *TEXTURE])
    image_types: list = field(default_factory=lambda: ["Original"])   # "LoG", "Wavelet" optional
    log_sigma_mm: list = field(default_factory=lambda: [2.0, 4.0, 6.0])
    min_texture_voxels: int = 64
    per_lesion: bool = True
    min_lesion_ml: float = 0.0

    @classmethod
    def for_kind(cls, kind: str) -> "Settings":
        if kind == "CT":
            return cls(bin_value=25.0, resample_mm=1.0, resegment_low=None)
        if kind == "MR":
            return cls(discretisation="count", bin_value=32, resample_mm=1.0, resegment_low=None)
        return cls()

    def params(self) -> dict:
        s = {"label": 1, "normalize": False, "force2D": False, "interpolator": self.interpolator,
             "minimumROIDimensions": 1, "minimumROISize": 1, "correctMask": True}
        s["binWidth" if self.discretisation == "width" else "binCount"] = (
            float(self.bin_value) if self.discretisation == "width" else int(self.bin_value))
        if self.resample_mm:
            s["resampledPixelSpacing"] = [float(self.resample_mm)] * 3
        if self.resegment_low is not None or self.resegment_high is not None:
            if self.resegment_high is None:
                s["resegmentRange"] = [float(self.resegment_low)]
            else:
                low = -1e12 if self.resegment_low is None else float(self.resegment_low)
                s["resegmentRange"] = [low, float(self.resegment_high)]
            s["resegmentMode"] = "absolute"
        if self.outlier_sigma:
            if "resegmentRange" in s:
                raise ValueError("Outlier filtering together with a value range needs IBSI resampling.")
            s["resegmentRange"] = [-float(self.outlier_sigma), float(self.outlier_sigma)]
            s["resegmentMode"] = "sigma"
        return s


def _extractor(settings: Settings, texture: bool):
    from radiomics import featureextractor
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    ex = featureextractor.RadiomicsFeatureExtractor(**settings.params())
    ex.disableAllImageTypes()
    for t in settings.image_types:
        if t == "LoG":
            ex.enableImageTypeByName("LoG", customArgs={"sigma": list(settings.log_sigma_mm)})
        else:
            ex.enableImageTypeByName(t)
    ex.disableAllFeatures()
    for c in settings.classes:
        if c in TEXTURE and not texture:
            continue
        ex.enableFeatureClassByName(c)
    return ex


def _voxels_after_resampling(mask: np.ndarray, geo: Geometry, settings: Settings) -> float:
    n = float((mask > 0).sum())
    if not settings.resample_mm:
        return n
    return n * float(np.prod(geo.spacing)) / settings.resample_mm ** 3


_ORDER = {"sitkNearestNeighbor": 0, "sitkLinear": 1, "sitkBSpline": 3}


def ibsi_resample(image_array: np.ndarray, mask: np.ndarray, geo: Geometry, settings: Settings,
                  margin_mm: float = 0.0):
    """IBSI interpolation of one region: image (order from settings.interpolator) and mask (trilinear,
    kept where >= 0.5) on the grid that IBSI defines for the whole image (n' = ceil(n s / s'), grid centres
    aligned). Only the grid points around the mask (plus margin_mm, e.g. the LoG kernel radius) are
    evaluated, so whole-body images stay fast; the
    values are the same as when the whole image is resampled (up to the spline prefilter at the crop
    border, which lies 8 voxels away from the mask). Arrays are (z, y, x). Returns image, mask, geometry."""
    from scipy import ndimage
    order = _ORDER.get(settings.interpolator, 1)
    sp = np.asarray(geo.spacing[::-1], float)
    n = np.asarray(image_array.shape)
    new = float(settings.resample_mm)
    step = new / sp
    n_new = np.ceil(n * sp / new).astype(int)
    start = 0.5 * (n - 1) - 0.5 * (n_new - 1) * step
    idx = np.argwhere(mask > 0)
    lo, hi = idx.min(0), idx.max(0)
    r = int(np.ceil(margin_mm / new))
    j0 = np.clip(np.ceil((lo - 1 - start) / step) - r, 0, n_new - 1).astype(int)
    j1 = np.clip(np.floor((hi + 1 - start) / step) + r, 0, n_new - 1).astype(int)
    pos = [start[i] + np.arange(j0[i], j1[i] + 1) * step[i] for i in range(3)]
    margin = 8
    a0 = np.maximum(np.floor([p.min() for p in pos]).astype(int) - margin, 0)
    a1 = np.minimum(np.ceil([p.max() for p in pos]).astype(int) + margin, n - 1)
    crop = tuple(slice(a0[i], a1[i] + 1) for i in range(3))
    grid = np.meshgrid(*[pos[i] - a0[i] for i in range(3)], indexing="ij")
    img = ndimage.map_coordinates(image_array[crop].astype(np.float64), grid, order=order, mode="nearest",
                                  prefilter=order > 1)
    if settings.round_intensities:
        img = np.round(img)
    msk = ndimage.map_coordinates((mask[crop] > 0).astype(np.float64), grid, order=1, mode="nearest") >= 0.5
    first = np.array([p[0] for p in pos])[::-1] * np.asarray(geo.spacing)        # (x, y, z) mm from voxel 0
    origin = tuple(np.asarray(geo.origin) + np.asarray(geo.direction).reshape(3, 3) @ first)
    g = Geometry(size=img.shape[::-1], spacing=(new, new, new), origin=origin, direction=geo.direction)
    return img.astype(np.float32), msk.astype(np.uint8), g


def resegment_interval(values: np.ndarray, settings: Settings) -> tuple[float | None, float | None]:
    """IBSI re-segmentation as one interval: the value range first, then mean +/- n SD of what remains."""
    lo, hi = settings.resegment_low, settings.resegment_high
    if settings.outlier_sigma:
        v = values[(values >= (-np.inf if lo is None else lo)) & (values <= (np.inf if hi is None else hi))]
        m, sd, k = float(v.mean()), float(v.std()), float(settings.outlier_sigma)
        lo = m - k * sd if lo is None else max(lo, m - k * sd)
        hi = m + k * sd if hi is None else min(hi, m + k * sd)
    return lo, hi


def _run(ex, image: sitk.Image, mask: np.ndarray, geo: Geometry) -> dict:
    m = geo.apply(sitk.GetImageFromArray((mask > 0).astype(np.uint8)))
    res = ex.execute(image, m)
    return {k: float(np.real(v)) for k, v in res.items() if not k.startswith("diagnostics_") and np.ndim(v) == 0}


_BIN_LOCK = threading.Lock()
_PCA_AXES = ("MajorAxisLength", "MinorAxisLength", "LeastAxisLength")


@contextmanager
def _ibsi_bin_edges(low: float | None):
    """Fixed bin size as IBSI defines it: the first bin starts at the lower bound of the re-segmentation range
    (or at the lowest value in the ROI when there is none), X_d = floor((X - X_min) / w) + 1. PyRadiomics starts
    at the multiple of w below the ROI minimum, which numbers the grey levels differently (joint average, sum
    average, grey-level emphases ... change). Only used inside the IBSI method; fixed bin count is unchanged."""
    from radiomics import imageoperations as io
    original = io.getBinEdges

    def edges(values, **kw):
        if kw.get("binCount") is not None:
            return original(values, **kw)
        w = float(kw.get("binWidth", 25))
        start = float(np.min(values)) if low is None else float(low)
        e = np.arange(start, float(np.max(values)) + 2 * w, w)
        return e if len(e) > 1 else np.array([start - 0.5, start + 0.5])

    with _BIN_LOCK:
        io.getBinEdges = edges
        try:
            yield
        finally:
            io.getBinEdges = original


def _ibsi_pca(out: dict, n_voxels: int) -> None:
    """IBSI computes the PCA axes from the covariance with N - 1, PyRadiomics with N: scale the three axis
    lengths by sqrt(N / (N - 1)) (elongation and flatness, ratios, are unchanged)."""
    if n_voxels > 1:
        f = float(np.sqrt(n_voxels / (n_voxels - 1)))
        for k in list(out):
            if k.startswith("original_shape_") and k.endswith(_PCA_AXES):
                out[k] *= f


def _ibsi_log(arr: np.ndarray, spacing_xyz, sigma_mm: float, boundary: str) -> np.ndarray:
    """Laplacian of Gaussian as IBSI 2 defines it: exact Gaussian kernel truncated at 4 sigma, sigma in mm,
    no scale normalisation (PyRadiomics multiplies by sigma^2 and uses a recursive approximation)."""
    from scipy import ndimage
    sig = [sigma_mm / sp for sp in spacing_xyz[::-1]]
    return ndimage.gaussian_laplace(arr.astype(np.float64), sigma=sig, mode=boundary, truncate=4.0).astype(np.float32)


def _features(image: sitk.Image, mask: np.ndarray, geo: Geometry, settings: Settings, cache: dict,
              image_array: np.ndarray | None = None) -> dict:
    texture = _voxels_after_resampling(mask, geo, settings) >= settings.min_texture_voxels
    if settings.resampling != "ibsi":
        ex = cache.setdefault(texture, _extractor(settings, texture))
        return _run(ex, image, mask, geo)
    use_log = "LoG" in settings.image_types
    margin = 4.0 * max(settings.log_sigma_mm) + 2 * (settings.resample_mm or 0) if use_log else 0.0
    arr, msk, g = (ibsi_resample(image_array, mask, geo, settings, margin) if settings.resample_mm
                   else (image_array.astype(np.float32), (mask > 0).astype(np.uint8), geo))
    lo, hi = resegment_interval(arr[msk > 0], settings)
    s = replace(settings, resampling="pyradiomics", resample_mm=None, resegment_low=lo, resegment_high=hi,
                outlier_sigma=None, image_types=[t for t in settings.image_types if t != "LoG"])
    with _ibsi_bin_edges(lo):
        out = _run(_extractor(s, texture), g.apply(sitk.GetImageFromArray(arr)), msk, g)
    _ibsi_pca(out, int((msk > 0).sum()))
    if use_log:
        # IBSI: the ROI is re-segmented on the original intensities, features come from the response map
        keep = (msk > 0) & (arr >= (-np.inf if lo is None else lo)) & (arr <= (np.inf if hi is None else hi))
        s_log = replace(s, resegment_low=None, resegment_high=None, image_types=["Original"],
                        classes=[c for c in settings.classes if c != "shape"])
        ex = _extractor(s_log, texture)
        for sigma in settings.log_sigma_mm:
            resp = _ibsi_log(arr, g.spacing, float(sigma), settings.log_boundary)
            tag = "log-sigma-" + str(float(sigma)).replace(".", "-") + "-mm-3D_"
            with _ibsi_bin_edges(None):
                res = _run(ex, g.apply(sitk.GetImageFromArray(resp)), keep.astype(np.uint8), g)
            for k, v in res.items():
                if k.startswith("original_"):
                    out[tag + k[len("original_"):]] = v
    return out


def extract(image_array: np.ndarray, geo: Geometry, mask: np.ndarray, case_id: str, mask_name: str,
            settings: Settings) -> pd.DataFrame:
    """One row for the whole mask ('all') and, if settings.per_lesion, one row per connected lesion."""
    image = geo.apply(sitk.GetImageFromArray(image_array.astype(np.float32)))
    cache: dict = {}
    rows = []
    if (mask > 0).any():
        rows.append({"case": case_id, "mask": mask_name, "lesion": "all", "voxels": int((mask > 0).sum()),
                     **_features(image, mask, geo, settings, cache, image_array)})
    if settings.per_lesion:
        lab, n = ndimage.label(mask > 0, structure=np.ones((3, 3, 3), bool))
        for i in range(1, n + 1):
            les = lab == i
            if les.sum() * geo.voxel_ml < settings.min_lesion_ml:
                continue
            rows.append({"case": case_id, "mask": mask_name, "lesion": str(i), "voxels": int(les.sum()),
                         **_features(image, les, geo, settings, cache, image_array)})
    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, settings: Settings, path: str | Path) -> Path:
    """CSV plus <name>_settings.json with every setting and the PyRadiomics version."""
    import radiomics
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    meta = {"settings": asdict(settings), "pyradiomics_params": settings.params(),
            "pyradiomics_version": radiomics.__version__, "rows": len(df)}
    side = path.with_name(path.stem + "_settings.json")
    side.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return side
