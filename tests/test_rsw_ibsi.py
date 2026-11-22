"""IBSI options of the PySegment radiomics module (rsw/radiomics.py).

Reference values: IBSI 1 digital phantom (Zwanenburg et al., Radiology 2020). The full validation against
the digital phantom and the chest CT configurations C-E is tools/validate_ibsi1.py (validation/ibsi1/).
"""
import numpy as np
import pytest
from scipy import ndimage

pytest.importorskip("radiomics")
from rsw import radiomics as rr  # noqa: E402
from rsw.io import Geometry  # noqa: E402

# IBSI 1 digital phantom, (z, y, x), 2 mm voxels.
PHANTOM = np.array([
    [[4, 4, 6, 4, 1], [4, 4, 6, 1, 1], [1, 1, 6, 1, 1], [1, 1, 6, 1, 1]],
    [[4, 1, 6, 4, 1], [1, 1, 3, 1, 1], [1, 1, 9, 1, 1], [1, 1, 1, 1, 1]],
    [[1, 4, 6, 1, 1], [1, 1, 6, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
    [[1, 4, 4, 1, 1], [1, 4, 4, 1, 1], [1, 4, 4, 1, 1], [1, 4, 4, 1, 1]]], np.float32)
MASK = np.ones_like(PHANTOM, np.uint8)
MASK[1, 1, 0] = MASK[1, 2, 2] = 0
MASK[3, 2:, 3:] = 0
IDENTITY = (1, 0, 0, 0, 1, 0, 0, 0, 1)

# PyRadiomics name -> (IBSI reference value, tolerance); kurtosis is converted to excess kurtosis below.
REFERENCE = {
    "original_shape_MeshVolume": (556, 4), "original_shape_SurfaceArea": (388, 3),
    "original_shape_Sphericity": (0.843, 0.005), "original_firstorder_Mean": (2.15, 0.005),
    "original_firstorder_Kurtosis": (-0.355, 0.0005), "original_firstorder_Entropy": (1.27, 0.005),
    "original_glcm_MaximumProbability": (0.503, 0.0005), "original_glcm_Contrast": (5.32, 0.005),
    "original_glrlm_ShortRunEmphasis": (0.705, 0.0005), "original_glszm_LargeAreaEmphasis": (550, 0.5),
    "original_gldm_LargeDependenceEmphasis": (109, 0.5), "original_ngtdm_Coarseness": (0.0296, 0.00005),
    "original_ngtdm_Busyness": (6.54, 0.005),
}


def _phantom_settings(**kw):
    return rr.Settings(discretisation="width", bin_value=1.0, resample_mm=None, resegment_low=None,
                       min_texture_voxels=0, per_lesion=False, **kw)


@pytest.mark.parametrize("mode", ["pyradiomics", "ibsi"])
def test_digital_phantom_matches_ibsi_reference(mode):
    geo = Geometry(PHANTOM.shape[::-1], (2.0, 2.0, 2.0), (0, 0, 0), IDENTITY)
    row = rr.extract(PHANTOM, geo, MASK, "phantom", "roi", _phantom_settings(resampling=mode)).iloc[0]
    for name, (ref, tol) in REFERENCE.items():
        value = row[name] - 3.0 if name.endswith("Kurtosis") else row[name]
        assert abs(value - ref) <= tol, (name, value, ref)


def test_default_is_unchanged():
    s = rr.Settings()
    assert s.resampling == "pyradiomics" and s.resegment_high is None and s.outlier_sigma is None
    p = s.params()
    assert p["resegmentRange"] == [0.0] and p["resampledPixelSpacing"] == [4.0] * 3


def test_range_params_and_unsupported_combination():
    assert rr.Settings(resegment_low=-1000, resegment_high=400).params()["resegmentRange"] == [-1000.0, 400.0]
    p = rr.Settings(resegment_low=None, outlier_sigma=3).params()
    assert p["resegmentMode"] == "sigma" and p["resegmentRange"] == [-3.0, 3.0]
    with pytest.raises(ValueError):
        rr.Settings(resegment_low=-1000, resegment_high=400, outlier_sigma=3).params()


def test_resegment_interval_range_then_outlier():
    v = np.concatenate([np.full(1000, 10.0), np.full(1000, 20.0), [500.0, -2000.0]])
    s = rr.Settings(resegment_low=-1000, resegment_high=400, outlier_sigma=3)
    lo, hi = rr.resegment_interval(v, s)
    assert lo == pytest.approx(15 - 3 * 5) and hi == pytest.approx(15 + 3 * 5)   # outliers removed by the range first
    assert rr.resegment_interval(v, rr.Settings(resegment_low=0, resegment_high=None)) == (0, None)


def _whole_image_ibsi(arr, sp_zyx, new, order):
    """Reference: IBSI grid for the whole image (grid centres aligned)."""
    sp, n = np.asarray(sp_zyx, float), np.asarray(arr.shape)
    n_new = np.ceil(n * sp / new).astype(int)
    axes = [0.5 * (n[i] - 1) + (np.arange(n_new[i]) - 0.5 * (n_new[i] - 1)) * new / sp[i] for i in range(3)]
    return ndimage.map_coordinates(arr.astype(np.float64), np.meshgrid(*axes, indexing="ij"), order=order,
                                   mode="nearest", prefilter=order > 1), n_new


@pytest.mark.parametrize("interp,order", [("sitkLinear", 1), ("sitkBSpline", 3)])
def test_bounding_box_resampling_equals_whole_image(interp, order):
    rng = np.random.default_rng(0)
    arr = ndimage.gaussian_filter(rng.normal(0, 100, (40, 60, 70)), 2).astype(np.float32)
    mask = np.zeros(arr.shape, np.uint8)
    mask[15:25, 20:38, 30:50] = 1
    geo = Geometry(arr.shape[::-1], (0.98, 0.98, 3.0), (-50.0, 20.0, 7.0), IDENTITY)
    s = rr.Settings(resample_mm=2.0, interpolator=interp, resampling="ibsi")
    img, msk, g = rr.ibsi_resample(arr, mask, geo, s)
    full, n_new = _whole_image_ibsi(arr, geo.spacing[::-1], 2.0, order)
    full_mask = _whole_image_ibsi(mask.astype(float), geo.spacing[::-1], 2.0, 1)[0] >= 0.5
    # locate the crop in the whole-image grid from its origin
    sp = np.asarray(geo.spacing)
    start_xyz = (0.5 * (np.asarray(geo.size) - 1) - 0.5 * (n_new[::-1] - 1) * 2.0 / sp) * sp   # mm of grid point 0
    j0 = np.rint(((np.asarray(g.origin) - np.asarray(geo.origin)) - start_xyz) / 2.0).astype(int)[::-1]
    sl = tuple(slice(j0[i], j0[i] + img.shape[i]) for i in range(3))
    assert msk.sum() == full_mask.sum()
    assert np.array_equal(msk.astype(bool), full_mask[sl])
    inside = msk > 0
    assert np.max(np.abs(img[inside] - full[sl][inside])) < 1e-3
    assert g.spacing == (2.0, 2.0, 2.0) and tuple(img.shape[::-1]) == g.size


def test_ibsi_mode_without_resampling_equals_default():
    rng = np.random.default_rng(1)
    arr = rng.gamma(2.0, 1.5, (20, 24, 26)).astype(np.float32)
    mask = np.zeros(arr.shape, np.uint8)
    mask[5:14, 6:18, 7:19] = 1
    geo = Geometry(arr.shape[::-1], (4.0, 4.0, 4.0), (0, 0, 0), IDENTITY)
    kw = dict(resample_mm=None, resegment_low=0.0, per_lesion=False)
    a = rr.extract(arr, geo, mask, "c", "m", rr.Settings(**kw)).iloc[0]
    b = rr.extract(arr, geo, mask, "c", "m", rr.Settings(resampling="ibsi", **kw)).iloc[0]
    feats = [c for c in a.index if c.startswith("original_") and not c.endswith(("MajorAxisLength", "MinorAxisLength", "LeastAxisLength"))]
    assert np.allclose(a[feats].astype(float), b[feats].astype(float), rtol=1e-9, atol=1e-12)
    n = int(mask.sum())                                  # PCA axes: covariance with N - 1 in the IBSI method
    for k in ("MajorAxisLength", "MinorAxisLength", "LeastAxisLength"):
        assert b[f"original_shape_{k}"] == pytest.approx(a[f"original_shape_{k}"] * np.sqrt(n / (n - 1)), rel=1e-12)


def test_ibsi_fixed_bin_size_starts_at_the_resegmentation_bound():
    """IBSI 1 chest CT configuration C: bins of 25 HU from -1000 HU. With the ROI minimum at -950 HU PyRadiomics
    numbers the grey levels from -950, IBSI from -1000: grey levels 2 higher, joint average +2, sum average +4."""
    from radiomics import imageoperations
    original = imageoperations.getBinEdges
    rng = np.random.default_rng(4)
    arr = rng.uniform(-950.0, -700.0, (16, 18, 20)).astype(np.float32)
    arr[0, 0, 0] = -950.0                                # ROI minimum exactly on a PyRadiomics bin edge
    mask = np.ones(arr.shape, np.uint8)
    geo = Geometry(arr.shape[::-1], (2.0, 2.0, 2.0), (0, 0, 0), IDENTITY)
    kw = dict(discretisation="width", bin_value=25.0, resample_mm=None, resegment_low=-1000.0, resegment_high=400.0,
              classes=["glcm"], min_texture_voxels=0, per_lesion=False)
    a = rr.extract(arr, geo, mask, "c", "m", rr.Settings(**kw)).iloc[0]
    b = rr.extract(arr, geo, mask, "c", "m", rr.Settings(resampling="ibsi", **kw)).iloc[0]
    assert b["original_glcm_JointAverage"] == pytest.approx(a["original_glcm_JointAverage"] + 2, abs=1e-9)
    assert b["original_glcm_SumAverage"] == pytest.approx(a["original_glcm_SumAverage"] + 4, abs=1e-9)
    assert b["original_glcm_Contrast"] == pytest.approx(a["original_glcm_Contrast"], abs=1e-9)   # differences only
    assert imageoperations.getBinEdges is original       # PyRadiomics is left as it was


def test_ibsi_pca_axes_of_the_digital_phantom():
    geo = Geometry(PHANTOM.shape[::-1], (2.0, 2.0, 2.0), (0, 0, 0), IDENTITY)
    row = rr.extract(PHANTOM, geo, MASK, "phantom", "roi", _phantom_settings(resampling="ibsi")).iloc[0]
    for k, ref in (("MajorAxisLength", 11.4), ("MinorAxisLength", 9.31), ("LeastAxisLength", 8.54)):
        assert abs(row[f"original_shape_{k}"] - ref) <= 0.01, (k, row[f"original_shape_{k}"])


def test_ibsi_log_definition():
    # f = x^2 (voxel units): Gaussian smoothing adds a constant, the Laplacian is exactly 2 (no sigma^2 factor)
    x = np.arange(40, dtype=np.float64)
    arr = np.broadcast_to((x - 20.0) ** 2, (30, 30, 40)).astype(np.float32)
    out = rr._ibsi_log(arr, (2.0, 2.0, 2.0), sigma_mm=3.0, boundary="reflect")
    assert np.allclose(out[10:20, 10:20, 16:24], 2.0, atol=0.01)     # 4-sigma truncation: about -0.3 %
    # sigma is given in mm and the response is in voxel units, as in the IBSI 2 reference response maps:
    # on an isotropic grid with twice the voxel size the response of a smooth blob is 2^2 = 4 times larger
    z, y, xx = np.mgrid[0:40, 0:40, 0:40].astype(float)
    blob = np.exp(-((xx - 20) ** 2 + (y - 20) ** 2 + (z - 20) ** 2) / (2 * 4.0 ** 2)).astype(np.float32)
    a = rr._ibsi_log(blob, (1.0, 1.0, 1.0), 2.0, "reflect")
    b = rr._ibsi_log(blob[::2, ::2, ::2], (2.0, 2.0, 2.0), 2.0, "reflect")
    assert b[10, 10, 10] / a[20, 20, 20] == pytest.approx(4.0, rel=0.05)


def test_ibsi_log_features_are_named_like_pyradiomics():
    rng = np.random.default_rng(2)
    arr = (rng.normal(100, 20, (30, 32, 34))).astype(np.float32)
    mask = np.zeros(arr.shape, np.uint8)
    mask[8:20, 9:22, 10:24] = 1
    geo = Geometry(arr.shape[::-1], (2.0, 2.0, 2.0), (0, 0, 0), IDENTITY)
    s = rr.Settings(resample_mm=2.0, resampling="ibsi", resegment_low=None, image_types=["Original", "LoG"],
                    log_sigma_mm=[2.0, 4.0], classes=["shape", "firstorder", "glcm"], per_lesion=False)
    row = rr.extract(arr, geo, mask, "c", "m", s).iloc[0]
    assert "log-sigma-2-0-mm-3D_firstorder_Mean" in row and "log-sigma-4-0-mm-3D_glcm_Contrast" in row
    assert not any(k.startswith("log-sigma") and "_shape_" in k for k in row.index)
    assert "original_shape_MeshVolume" in row
