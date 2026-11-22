"""Image registration between studies of one patient (PET, SPECT, CT, MRI), built on SimpleITK.

Conventions (the same as SimpleITK / ITK):
  * ``fixed`` is the image whose grid is kept; ``moving`` is the image that is brought onto it.
  * The transform maps points of the FIXED space to points of the MOVING space, so
    ``sitk.Resample(moving, fixed_grid, transform)`` gives the moving image on the fixed grid.
    Masks drawn on the moving image are brought to the fixed grid in the same way (nearest neighbour).

Start ('init'): 'auto' (default) keeps the scanner coordinates when the two images overlap in space (same session,
hybrid scanner, serial studies on one table) and aligns the image centres only when they do not; 'identity',
'geometry' (image centres) and 'landmarks' force a start. The result is never worse than the start: when the
optimised match (metric over all voxels, or a fixed regular sample of a large image) ends worse than at the start, the start transform is kept and a warning
is given.

Steps: 'rigid' (6 parameters), 'affine' (12), 'bspline' (affine first, then a free-form B-spline on top).
The metric is Mattes mutual information when the two images are of different kinds (PET to CT, SPECT to CT,
PET to SPECT ...), mean squares for two images of the same functional kind (PET to PET, SPECT to SPECT)
and normalised correlation for CT to CT or MR to MR.

For a hybrid pair (PET/CT, SPECT/CT) register the two CTs and apply that transform to the functional image:
see ``register_hybrid``. Activity is kept: after resampling use ``quantification_check`` to see how much the
total activity (sum of values times voxel volume) changed.

Research use only. Always check the result visually (fusion, checkerboard) before using it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import SimpleITK as sitk

from rsw.io import Geometry, Volume

MODES = ("rigid", "affine", "bspline")
FUNCTIONAL = ("PET", "SPECT")


# ---------------------------------------------------------------------------------- helpers
def to_image(vol: Volume) -> sitk.Image:
    """The volume as a float32 SimpleITK image with its geometry."""
    return vol.geometry.apply(sitk.GetImageFromArray(vol.array.astype(np.float32)))


def choose_metric(fixed_kind: str, moving_kind: str) -> str:
    """'mattes' for different kinds, 'correlation' for two images of the same kind. Two PET or two SPECT scans of
    a dosimetry series differ in overall level (decay, wash-out), which mean squares would treat as misalignment;
    normalised correlation ignores a global scale."""
    if fixed_kind != moving_kind:
        return "mattes"
    return "correlation"


def landmark_transform(fixed_points: Sequence[Sequence[float]], moving_points: Sequence[Sequence[float]]) -> sitk.Transform:
    """Rigid transform from at least three pairs of points given in PHYSICAL coordinates (mm, LPS):
    fixed_points[i] in the fixed space corresponds to moving_points[i] in the moving space."""
    fp, mp = np.asarray(fixed_points, float), np.asarray(moving_points, float)
    if fp.shape != mp.shape or fp.ndim != 2 or fp.shape[1] != 3:
        raise ValueError("landmarks must be N x 3 arrays of equal length")
    if len(fp) < 3:
        raise ValueError("at least three landmark pairs are needed")
    return sitk.LandmarkBasedTransformInitializer(sitk.VersorRigid3DTransform(), fp.ravel().tolist(), mp.ravel().tolist())


def overlap_fraction(fixed: sitk.Image, moving: sitk.Image) -> float:
    """Volume of the intersection of the two images' physical bounding boxes divided by the smaller box (0 to 1)."""
    def box(img):
        sz = np.array(img.GetSize()) - 1
        corners = [img.TransformContinuousIndexToPhysicalPoint([float(c) for c in (i * sz[0], j * sz[1], k * sz[2])])
                   for i in (0, 1) for j in (0, 1) for k in (0, 1)]
        c = np.array(corners)
        half = np.abs(np.array(img.GetDirection()).reshape(3, 3) * np.array(img.GetSpacing())).sum(axis=1) / 2
        return c.min(0) - half, c.max(0) + half
    (a0, a1), (b0, b1) = box(fixed), box(moving)
    inter = np.prod(np.clip(np.minimum(a1, b1) - np.maximum(a0, b0), 0, None))
    small = min(np.prod(a1 - a0), np.prod(b1 - b0))
    return float(inter / small) if small > 0 else 0.0


def resolve_init(fixed: sitk.Image, moving: sitk.Image, init: str) -> str:
    """'auto' -> 'identity' when the images overlap by at least half of the smaller one, else 'geometry'."""
    if init != "auto":
        return init
    return "identity" if overlap_fraction(fixed, moving) >= 0.5 else "geometry"


METRIC_CHECK_VOXELS = 2_000_000


def metric_value(fixed: sitk.Image, moving: sitk.Image, metric: str, transform: sitk.Transform,
                 fixed_mask: sitk.Image | None = None) -> float:
    """The metric of ``transform`` on every voxel of the fixed image, or on a regular grid of about
    METRIC_CHECK_VOXELS voxels for large images (whole-body CT). The same voxels are used for every transform,
    so values from two calls can be compared."""
    reg = sitk.ImageRegistrationMethod()
    _set_metric(reg, metric, 1.0, 0)
    n = int(np.prod(fixed.GetSize()))
    if n > METRIC_CHECK_VOXELS:
        reg.SetMetricSamplingStrategy(reg.REGULAR)
        reg.SetMetricSamplingPercentage(METRIC_CHECK_VOXELS / n, 0)
    else:
        reg.SetMetricSamplingStrategy(reg.NONE)
    reg.SetInterpolator(sitk.sitkLinear)
    if fixed_mask is not None:
        reg.SetMetricFixedMask(fixed_mask)
    reg.SetInitialTransform(transform)
    return float(reg.MetricEvaluate(fixed, moving))


LARGE_CORRECTION_MM = 20.0


def correction_mm(start: sitk.Transform, final: sitk.Transform, fixed: sitk.Image) -> float:
    """How far the optimisation moved the centre of the fixed image away from where the start transform put it (mm)."""
    sz = np.array(fixed.GetSize(), float)
    c = fixed.TransformContinuousIndexToPhysicalPoint(((sz - 1) / 2).tolist())
    return float(np.linalg.norm(np.array(final.TransformPoint(c)) - np.array(start.TransformPoint(c))))


def _worse(after: float, before: float) -> bool:
    """Lower is better for every metric used here; a change below 0.1 % counts as equal."""
    return after > before + 1e-3 * abs(before)


@dataclass
class RegResult:
    transform: sitk.Transform
    mode: str
    metric: str
    metric_value: float                       # final value of the optimised metric (lower is better)
    iterations: int = 0
    stop_reason: str = ""
    warnings: list = field(default_factory=list)
    jacobian: dict | None = None              # for 'bspline': min, fraction <= 0 (folding)
    init: str = ""                            # start actually used ('auto' resolved)
    start_metric_value: float | None = None   # metric at the start (same voxels as metric_value)
    kept_start: bool = False                  # True when the optimisation ended worse than the start

    @property
    def folding(self) -> bool:
        return bool(self.jacobian and self.jacobian["fraction_nonpositive"] > 0)


# ---------------------------------------------------------------------------------- registration
def _initial_linear(fixed: sitk.Image, moving: sitk.Image, mode: str, init: str,
                    landmarks: tuple | None) -> sitk.Transform:
    if init == "landmarks":
        if landmarks is None:
            raise ValueError("init='landmarks' needs landmarks=(fixed_points, moving_points)")
        rigid = landmark_transform(*landmarks)
        if mode == "rigid":
            return rigid
        aff = sitk.AffineTransform(3)
        aff.SetMatrix(rigid.GetMatrix())
        aff.SetTranslation(rigid.GetTranslation())
        aff.SetCenter(rigid.GetCenter())
        return aff
    base = sitk.VersorRigid3DTransform() if mode == "rigid" else sitk.AffineTransform(3)
    if init == "identity":
        return base
    if init == "geometry":
        return sitk.CenteredTransformInitializer(fixed, moving, base, sitk.CenteredTransformInitializerFilter.GEOMETRY)
    raise ValueError(f"unknown init '{init}'")


def _corners(img: sitk.Image) -> list:
    size = img.GetSize()
    return [img.TransformContinuousIndexToPhysicalPoint([float(i * (size[0] - 1)), float(j * (size[1] - 1)),
                                                          float(k * (size[2] - 1))])
            for i in (0, 1) for j in (0, 1) for k in (0, 1)]


def still_moving_mm(track: list, window: int | None = None) -> float:
    """Largest movement of the fixed-image corners over the last `window` iterations of the finest level (over
    the whole level if it ran fewer). The corners bound the movement of every point of the image."""
    window = window or DRIFT_WINDOW
    if not track:
        return 0.0
    last = [p for lv, p in track if lv == track[-1][0]]
    ref = last[0] if len(last) <= window else last[-1 - window]
    return float(np.linalg.norm(last[-1] - ref, axis=1).max())


def linear_as_affine(t: sitk.Transform, img: sitk.Image) -> sitk.AffineTransform:
    """Any linear transform (also a composite of linear transforms) as one AffineTransform centred on the image:
    least-squares fit to the mapped corners and centre of the image, exact for linear transforms."""
    size = img.GetSize()
    idx = [(i * (size[0] - 1), j * (size[1] - 1), k * (size[2] - 1)) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    p = np.array([img.TransformContinuousIndexToPhysicalPoint([float(v) for v in c]) for c in idx])
    c = p.mean(axis=0)
    q = np.array([t.TransformPoint(tuple(map(float, x))) for x in p])
    a, *_ = np.linalg.lstsq(np.c_[p - c, np.ones(len(p))], q, rcond=None)
    out = sitk.AffineTransform(3)
    out.SetCenter(tuple(map(float, c)))
    out.SetMatrix(a[:3].T.ravel().tolist())
    out.SetTranslation(tuple(map(float, a[3] - c)))
    return out


def _set_metric(reg: sitk.ImageRegistrationMethod, metric: str, sampling: float, seed: int):
    if metric == "mattes":
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    elif metric == "mean_squares":
        reg.SetMetricAsMeanSquares()
    elif metric == "correlation":
        reg.SetMetricAsCorrelation()
    else:
        raise ValueError(f"unknown metric '{metric}'")
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(sampling, seed)


MAX_SAMPLES = 1_500_000     # metric samples per resolution level; 20 % of a whole-body CT would be 8 million or more


def _cap_samples(reg: sitk.ImageRegistrationMethod, fixed: sitk.Image, sampling: float, shrinks, seed: int):
    """Random sampling of ``sampling`` of the fixed voxels on each level, but never more than MAX_SAMPLES points:
    a 512 x 512 x 161 CT then registers in about a minute instead of seven, with the same result (thousands of
    points are enough for the metric; elastix uses a few thousand per iteration). Images up to 7.5 million voxels
    (all images of validation/registration/) are not affected."""
    n = float(np.prod(fixed.GetSize()))
    per = [min(float(sampling), MAX_SAMPLES / max(n / float(s) ** 3, 1.0)) for s in shrinks]
    reg.SetMetricSamplingPercentagePerLevel(per, seed)


def _levels(reg: sitk.ImageRegistrationMethod, levels: int):
    shrink = [2 ** i for i in range(levels - 1, -1, -1)]
    reg.SetShrinkFactorsPerLevel(shrink)
    reg.SetSmoothingSigmasPerLevel([float(s) / 2 for s in shrink[:-1]] + [0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()


def _normalise(img: sitk.Image, metric: str, kind: str) -> sitk.Image:
    """CT: limit the range to soft tissue and bone so that air and metal do not dominate the metric.
    Functional images are used as they are (activity is not rescaled)."""
    if kind == "CT":
        img = sitk.Clamp(img, sitk.sitkFloat32, -300.0, 600.0)
    return sitk.Cast(img, sitk.sitkFloat32)


def jacobian_stats(transform: sitk.Transform, grid: Geometry) -> dict:
    """Determinant of the Jacobian of a transform over a grid: min, max and the fraction of voxels with
    determinant <= 0, where the deformation folds the image onto itself."""
    ref = grid.reference()
    field_img = sitk.TransformToDisplacementField(transform, sitk.sitkVectorFloat64, ref.GetSize(), ref.GetOrigin(),
                                                  ref.GetSpacing(), ref.GetDirection())
    det = sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(field_img))
    return {"min": float(det.min()), "max": float(det.max()), "mean": float(det.mean()),
            "fraction_nonpositive": float((det <= 0).mean()), "n_nonpositive": int((det <= 0).sum())}


def folding_text(jac: dict) -> str:
    """'12 voxels (0.0021 %)': a fraction printed with two significant digits, never rounded to 0.00 %."""
    return f"{jac['n_nonpositive']} voxels ({100 * jac['fraction_nonpositive']:.2g} %)"


ENGINES = ("simpleitk", "elastix")
RIGID_FIRST = True
CONVERGED_MM = 0.05         # at the iteration limit, still 'converged' if the corners moved less than this ...
DRIFT_WINDOW = 20           # ... over the last DRIFT_WINDOW iterations of the finest level          # affine (and the affine stage of 'bspline') starts from a rigid registration


def register(fixed: Volume, moving: Volume, mode: str = "rigid", metric: str = "auto", init: str = "auto",
             landmarks: tuple | None = None, levels: int = 3, iterations: int = 500, sampling: float = 0.2,
             bspline_nodes: int = 8, seed: int = 1234, progress: Callable[[int, float], None] | None = None,
             engine: str = "simpleitk", profile: str = "general") -> RegResult:
    """Register ``moving`` onto ``fixed``. Deterministic for a given seed. engine 'elastix' (optional package
    itk-elastix) with profile 'general' or 'lung' (respiratory motion of the lungs between two CTs)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if engine == "elastix":
        return _register_elastix(fixed, moving, mode, init, landmarks, seed, profile)
    if engine != "simpleitk":
        raise ValueError(f"engine must be one of {ENGINES}")
    metric = choose_metric(fixed.kind, moving.kind) if metric == "auto" else metric
    fimg = _normalise(to_image(fixed), metric, fixed.kind)
    mimg = _normalise(to_image(moving), metric, moving.kind)
    warns: list = []

    lin_mode = "rigid" if mode == "rigid" else "affine"
    init = resolve_init(fimg, mimg, init)
    n_rigid = 0
    if lin_mode == "affine" and RIGID_FIRST:
        # Rigid first, then affine from the rigid result (as elastix and ANTs do): the 12 affine parameters
        # started far from the answer wander (scale/shear absorb part of a rotation) and need many more
        # iterations; the start guard below then also keeps the affine result at least as good as the rigid one.
        rig = register(fixed, moving, "rigid", metric, init, landmarks, levels, iterations, sampling, seed=seed)
        n_rigid = rig.iterations
        tfm0 = linear_as_affine(rig.transform, fimg)
        warns.extend(w for w in rig.warnings if not w.startswith("Large correction"))
    else:
        tfm0 = _initial_linear(fimg, mimg, lin_mode, init, landmarks)
    m_start = metric_value(fimg, mimg, metric, tfm0)
    kept = False

    reg = sitk.ImageRegistrationMethod()
    _set_metric(reg, metric, sampling, seed)
    reg.SetInterpolator(sitk.sitkLinear)
    # Regular-step descent: steps shrink when the metric stops improving; robust for small phantoms and real scans
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-3, numberOfIterations=iterations,
                                                 relaxationFactor=0.6, gradientMagnitudeTolerance=1e-8)
    reg.SetOptimizerScalesFromPhysicalShift()
    _levels(reg, levels)
    _cap_samples(reg, fimg, sampling, [2 ** i for i in range(levels - 1, -1, -1)], seed)
    reg.SetInitialTransform(tfm0, inPlace=False)
    corners = _corners(fimg)
    track: list = []                                   # (level, corner positions) per iteration

    def on_iteration():
        t = sitk.Transform(tfm0)
        t.SetParameters(reg.GetOptimizerPosition())
        track.append((reg.GetCurrentLevel(), np.array([t.TransformPoint(c) for c in corners])))
        if progress:
            progress(reg.GetOptimizerIteration(), reg.GetMetricValue())

    reg.AddCommand(sitk.sitkIterationEvent, on_iteration)
    lin = reg.Execute(fimg, mimg)
    n_it, stop = n_rigid + reg.GetOptimizerIteration(), reg.GetOptimizerStopConditionDescription()
    drift = still_moving_mm(track)
    if "Maximum number of iterations" in stop and drift > CONVERGED_MM:
        warns.append(f"Not converged: the {lin_mode} optimisation stopped at the limit of {iterations} iterations "
                     f"per level and the image corners still moved {drift:.2f} mm over the last {DRIFT_WINDOW} "
                     "iterations; the result may be inaccurate. Check the fusion or increase the iterations.")
    value = metric_value(fimg, mimg, metric, lin)
    if _worse(value, m_start):
        warns.append(f"The optimisation ended with a worse match than the start ({value:.4f} vs {m_start:.4f}); "
                     "the start position is kept. Check the fusion; try another start (landmarks) or the hybrid option.")
        lin, value, kept = tfm0, m_start, True
    final: sitk.Transform = lin
    jac = None

    if mode == "bspline":
        mesh = [max(2, bspline_nodes)] * 3
        bs = sitk.BSplineTransformInitializer(fimg, mesh, order=3)
        reg2 = sitk.ImageRegistrationMethod()
        _set_metric(reg2, metric, sampling, seed)
        reg2.SetInterpolator(sitk.sitkLinear)
        reg2.SetMovingInitialTransform(lin)
        reg2.SetInitialTransformAsBSpline(bs, inPlace=True, scaleFactors=[1] * min(levels, 3))
        reg2.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=max(20, iterations // 2),
                                  maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=1000,
                                  costFunctionConvergenceFactor=1e7)
        reg2.SetShrinkFactorsPerLevel([4, 2, 1][-min(levels, 3):])
        _cap_samples(reg2, fimg, sampling, [4, 2, 1][-min(levels, 3):], seed)
        reg2.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0][-min(levels, 3):])
        reg2.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
        reg2.Execute(fimg, mimg)
        n_it += reg2.GetOptimizerIteration()
        stop = reg2.GetOptimizerStopConditionDescription()
        comp = sitk.CompositeTransform(3)
        comp.AddTransform(lin)       # applied second
        comp.AddTransform(bs)        # applied first
        comp.FlattenTransform()          # a nested composite cannot be written to a file
        value_bs = metric_value(fimg, mimg, metric, comp)
        if _worse(value_bs, value):
            warns.append(f"The B-spline step made the match worse ({value_bs:.4f} vs {value:.4f}); "
                         f"the {lin_mode} result is kept without deformation.")
        else:
            final, value = comp, value_bs
            jac = jacobian_stats(final, fixed.geometry)
        if jac and jac["fraction_nonpositive"] > 0:
            warns.append(f"The deformation folds the image in {folding_text(jac)} "
                         "(Jacobian determinant <= 0). Do not use it for quantification; try fewer B-spline nodes or 'affine'.")
    shift = correction_mm(tfm0, final, fimg)
    if shift > LARGE_CORRECTION_MM:
        warns.append(f"Large correction: the optimisation moved the image by {shift:.0f} mm from the start (at the centre "
                     "of the fixed image). This can be right (different positioning), but check the fusion carefully; for "
                     "PET/CT or SPECT/CT studies the CT-to-CT (hybrid) option is more reliable.")
    if value > -1e-12 and metric == "mattes":
        warns.append("Mutual information did not improve; check that the two images overlap and look at the fusion.")
    return RegResult(final, mode, metric, float(value), int(n_it), str(stop), warns, jac, init, float(m_start), kept)


def _register_elastix(fixed: Volume, moving: Volume, mode: str, init: str, landmarks, seed: int, profile: str) -> RegResult:
    from rsw import registration_elastix as rx
    if not rx.available():
        raise ValueError("the elastix engine needs the package itk-elastix (pip install itk-elastix)")
    if init == "landmarks" or landmarks is not None:
        raise ValueError("a landmark start is available with the SimpleITK engine only")
    fimg, mimg = to_image(fixed), to_image(moving)
    init = resolve_init(fimg, mimg, init)
    transform, f, m, mask = rx.estimate(fimg, mimg, fixed.kind, moving.kind, mode, profile, init, seed=seed)
    metric = "mattes" if (profile == "general" and fixed.kind != moving.kind) else "correlation"
    start = sitk.Transform() if init == "identity" else _initial_linear(f, m, "rigid", "geometry", None)
    m_start = metric_value(f, m, metric, start, mask)
    value = metric_value(f, m, metric, transform, mask)
    warns, kept, jac = [], False, None
    if _worse(value, m_start):
        warns.append(f"The elastix result matches worse than the start ({value:.4f} vs {m_start:.4f}); the start "
                     "position is kept. Check the fusion.")
        transform, value, kept = start, m_start, True
    elif mode == "bspline":
        jac = jacobian_stats(transform, fixed.geometry)
        if jac["fraction_nonpositive"] > 0:
            warns.append(f"The deformation folds the image in {folding_text(jac)} "
                         "(Jacobian determinant <= 0). Do not use it for quantification.")
    shift = correction_mm(start, transform, fimg)
    if shift > LARGE_CORRECTION_MM and profile != "lung":
        warns.append(f"Large correction: the image moved by {shift:.0f} mm from the start (at the centre of the fixed "
                     "image). Check the fusion carefully.")
    return RegResult(transform, mode, f"elastix-{profile}-{metric}", float(value), 0, "elastix", warns, jac, init,
                     float(m_start), kept)


def register_hybrid(fixed_ct: Volume, moving_ct: Volume, moving_functional: Volume, **kw) -> tuple[RegResult, Volume]:
    """Hybrid studies (PET/CT, SPECT/CT): register moving CT to fixed CT and apply the same transform to the
    functional image of the moving study. Returns the result and the functional image on the fixed CT grid.
    The functional image must be in the same frame of reference as its own CT (true for hybrid scanners)."""
    res = register(fixed_ct, moving_ct, **kw)
    return res, resample_volume(moving_functional, fixed_ct.geometry, res.transform)


# ---------------------------------------------------------------------------------- applying
def resample_volume(vol: Volume, ref: Geometry, transform: sitk.Transform | None = None, linear: bool = True,
                    default: float | None = None) -> Volume:
    """``vol`` on the grid ``ref`` through ``transform`` (fixed to moving; identity if None)."""
    src = to_image(vol)
    fill = float(vol.array.min()) if default is None else default
    out = sitk.Resample(src, ref.reference(), transform or sitk.Transform(),
                        sitk.sitkLinear if linear else sitk.sitkNearestNeighbor, fill, sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(out)
    from rsw.io import is_counts
    if is_counts(vol.unit):
        arr = arr * np.float32(ref.voxel_ml / vol.geometry.voxel_ml)      # counts per voxel: conserve the total
        if transform is not None:
            from rsw.io import jacobian_on
            arr = arr * jacobian_on(ref, transform)
    info = dict(vol.info)
    if transform is not None:
        info["registered"] = True
    return Volume(vol.name, arr, ref, vol.kind, vol.unit, vol.path, info)


def propagate_mask(mask: np.ndarray, mask_geometry: Geometry, ref: Geometry, transform: sitk.Transform | None = None) -> np.ndarray:
    """A mask (array z, y, x) drawn on the moving image, on the fixed grid; nearest neighbour keeps the labels."""
    img = mask_geometry.apply(sitk.GetImageFromArray(mask.astype(np.uint8)))
    out = sitk.Resample(img, ref.reference(), transform or sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(out).astype(np.uint8)


def quantification_check(before: Volume, after: Volume, roi_before: np.ndarray | None = None,
                         roi_after: np.ndarray | None = None) -> dict:
    """How much did the resampling change the numbers? Total activity = sum(values) * voxel volume (mL), for the
    whole image or for an ROI (also bring the ROI through ``propagate_mask``). Also mean value in the ROI.
    A loss of total activity usually means that part of the object left the field of view."""
    def total(v: Volume, roi):
        a = v.array if roi is None else v.array[roi.astype(bool)]
        from rsw.io import is_counts
        factor = 1.0 if is_counts(v.unit) else v.geometry.voxel_ml          # counts are per voxel already
        return float(a.sum(dtype=np.float64)) * factor, float(a.mean()) if a.size else float("nan")
    from rsw.io import is_counts
    t0, m0 = total(before, roi_before)
    t1, m1 = total(after, roi_after)
    rel = (t1 - t0) / t0 if t0 else float("nan")
    out = {"total_before": t0, "total_after": t1, "total_change_pct": 100 * rel,
           "mean_before": m0, "mean_after": m1,          # mean over different grids/fields of view: not shown to the user
           "mean_change_pct": 100 * (m1 - m0) / m0 if m0 else float("nan"), "max_before": None, "max_after": None}
    if not is_counts(before.unit):                       # concentration: the maximum must stay about the same
        out["max_before"], out["max_after"] = float(before.array.max()), float(after.array.max())
    return out


# ---------------------------------------------------------------------------------- transform files
def save_transform(transform: sitk.Transform, path: str | Path) -> Path:
    path = Path(path)
    if path.suffix.lower() not in (".tfm", ".h5", ".hdf5", ".txt"):
        path = path.with_suffix(".tfm")
    sitk.WriteTransform(transform, str(path))
    return path


def load_transform(path: str | Path) -> sitk.Transform:
    return sitk.ReadTransform(str(path))


def transform_matrix(transform: sitk.Transform) -> np.ndarray | None:
    """4 x 4 matrix of a linear transform (also a composite of linear ones), else None."""
    m = np.eye(4)
    ts = [transform.GetNthTransform(i) for i in range(transform.GetNumberOfTransforms())] \
        if isinstance(transform, sitk.CompositeTransform) or transform.GetName() == "CompositeTransform" else [transform]
    for t in ts:
        a = np.eye(4)
        if t.GetName() == "IdentityTransform":
            continue
        if hasattr(t, "GetOffset") and not hasattr(t, "GetMatrix"):          # pure translation
            a[:3, 3] = t.GetOffset()
            m = m @ a
            continue
        if not hasattr(t, "GetMatrix"):
            return None
        a[:3, :3] = np.array(t.GetMatrix()).reshape(3, 3)
        c = np.array(t.GetCenter()) if hasattr(t, "GetCenter") else np.zeros(3)
        tr = np.array(t.GetTranslation())
        a[:3, 3] = tr + c - a[:3, :3] @ c
        m = m @ a
    return m


def describe(transform: sitk.Transform) -> str:
    """One line for the log: translation (mm) and rotation (deg) for a rigid transform."""
    m = transform_matrix(transform)
    if m is None:
        return "deformable transform"
    r = m[:3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))) if np.allclose(r @ r.T, np.eye(3), atol=1e-4) else float("nan")
    return f"translation {np.round(m[:3, 3], 2).tolist()} mm, rotation {ang:.2f} deg"
