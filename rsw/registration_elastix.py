"""Optional elastix engine for PySegment registration (package ``itk-elastix``, Apache 2.0).

elastix optimises with adaptive stochastic gradient descent on a few thousand new random samples per iteration, so a
fine B-spline grid on a whole CT takes one or two minutes. The result is converted into SimpleITK transforms (exact:
the same point mapping as transformix), so saving, re-applying, mask propagation, the Jacobian check and the
quantification check of rsw.registration work unchanged.

Profiles:
  general : the image kinds decide (normalised correlation for the same kind, Mattes mutual information otherwise;
            CT limited to -300..600 HU as in the SimpleITK engine); native grids.
  lung    : respiratory motion of the lungs (4D CT, inspiration/expiration CT): both CTs on a 2 mm isotropic working
            grid, CT window -1000..200 HU (lung tissue and vessels), metric only inside a lung mask of the fixed CT,
            B-spline control points every 10 mm at the finest level. Validated on the POPI 4D CT data with expert
            landmarks (validation/registration_landmarks/).
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk

PROFILES = ("general", "lung")
LUNG_WINDOW = (-1000.0, 200.0)


def available() -> bool:
    try:
        import itk  # noqa: F401
        return hasattr(itk, "elastix_registration_method")
    except Exception:  # noqa: BLE001 - any import problem means: not available
        return False


def lung_mask(ct: sitk.Image, dilate_mm: float = 8.0) -> sitk.Image:
    """Lungs (and airways) of a CT: air-like voxels (< -400 HU) inside the body, where the body is the largest
    region above -500 HU with its holes filled slice by slice (axial slices, in which the chest wall encloses the
    lungs). Independent of whether the airways reach the edge of the image. Parts smaller than 5 % of the largest
    are dropped; the result is dilated by 8 mm."""
    from scipy import ndimage
    a = sitk.GetArrayFromImage(ct)
    lab, n = ndimage.label(a > -500)
    if n == 0:
        raise ValueError("no body found in the CT (nothing above -500 HU)")
    body = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    body = np.stack([ndimage.binary_fill_holes(s) for s in body])
    lab, n = ndimage.label((a < -400) & body)
    if n == 0:
        raise ValueError("no lungs found in the CT (no air inside the body)")
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    lungs = np.isin(lab, np.nonzero(sizes >= 0.05 * sizes.max())[0])
    m = sitk.GetImageFromArray(lungs.astype(np.uint8))
    m.CopyInformation(ct)
    r = max(1, int(round(dilate_mm / min(ct.GetSpacing()))))
    return sitk.BinaryDilate(m, [r] * 3)


def isotropic(img: sitk.Image, mm: float, fill: float) -> sitk.Image:
    sp, sz = np.array(img.GetSpacing()), np.array(img.GetSize())
    n = [int(v) for v in np.ceil(sz * sp / mm)]
    return sitk.Resample(img, n, sitk.Transform(), sitk.sitkLinear, img.GetOrigin(), [mm] * 3, img.GetDirection(),
                         fill, sitk.sitkFloat32)


def _to_itk(img: sitk.Image, dtype=np.float32):
    import itk
    out = itk.image_from_array(np.ascontiguousarray(sitk.GetArrayFromImage(img).astype(dtype)))
    out.SetSpacing(img.GetSpacing())
    out.SetOrigin(img.GetOrigin())
    out.SetDirection(itk.matrix_from_array(np.array(img.GetDirection(), float).reshape(3, 3)))
    return out


def parameter_maps(mode: str, metric: str, grid_mm: float, resolutions: int, iterations: int, samples: int,
                   centre_start: bool, seed: int):
    import itk
    po = itk.ParameterObject.New()
    kinds = {"rigid": ["rigid"], "affine": ["affine"], "bspline": ["affine", "bspline"]}[mode]
    for i, kind in enumerate(kinds):
        p = (po.GetDefaultParameterMap(kind, resolutions, grid_mm) if kind == "bspline"
             else po.GetDefaultParameterMap(kind, resolutions))
        p["Metric"] = [metric]
        p["MaximumNumberOfIterations"] = [str(iterations if kind == "bspline" else 300)]
        p["NumberOfSpatialSamples"] = [str(samples)]
        p["NewSamplesEveryIteration"] = ["true"]
        p["ImageSampler"] = ["RandomCoordinate"]
        p["RandomSeed"] = [str(seed + i)]
        p["ErodeMask"] = ["false"]
        p["WriteResultImage"] = ["false"]
        if i == 0 and centre_start:
            p["AutomaticTransformInitialization"] = ["true"]
            p["AutomaticTransformInitializationMethod"] = ["GeometricalCenter"]
        po.AddParameterMap(p)
    return po


def _f(pm, key):
    return [float(v) for v in pm[key]]


def to_sitk(result_parameters) -> sitk.Transform:
    """elastix result (rigid / affine, optionally followed by a B-spline; 'Compose') as one SimpleITK transform that
    maps fixed to moving points like transformix."""
    parts = []
    for k in range(result_parameters.GetNumberOfParameterMaps()):
        pm = result_parameters.GetParameterMap(k)
        name = pm["Transform"][0]
        p = _f(pm, "TransformParameters")
        if name == "EulerTransform":
            t = sitk.Euler3DTransform()
            t.SetCenter(_f(pm, "CenterOfRotationPoint"))
            if pm.get("ComputeZYX", ("false",))[0] == "true":
                t.SetComputeZYX(True)
            t.SetRotation(*p[:3])
            t.SetTranslation(p[3:6])
        elif name == "AffineTransform":
            t = sitk.AffineTransform(3)
            t.SetCenter(_f(pm, "CenterOfRotationPoint"))
            t.SetMatrix(p[:9])
            t.SetTranslation(p[9:12])
        elif name == "BSplineTransform":
            t = sitk.BSplineTransform(3, int(float(pm.get("BSplineTransformSplineOrder", ("3",))[0])))
            t.SetFixedParameters(_f(pm, "GridSize") + _f(pm, "GridOrigin") + _f(pm, "GridSpacing") + _f(pm, "GridDirection"))
            t.SetParameters(p)
        else:
            raise ValueError(f"elastix transform {name} is not supported")
        parts.append(t)
    if len(parts) == 1:
        return parts[0]
    comp = sitk.CompositeTransform(3)
    for t in reversed(parts):          # elastix 'Compose': the first map is applied first; SimpleITK applies the last added first
        comp.AddTransform(t)
    return comp


def estimate(fixed: sitk.Image, moving: sitk.Image, fixed_kind: str, moving_kind: str, mode: str, profile: str,
             init: str, grid_mm: float = 10.0, iterations: int = 500, samples: int = 3000, seed: int = 1234):
    """Run elastix on two SimpleITK images (already in physical space). Returns (transform, fixed image used,
    moving image used, fixed mask or None) so that the caller can evaluate the metric on the same images."""
    import itk
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    if profile == "lung":
        if fixed_kind != "CT" or moving_kind != "CT":
            raise ValueError("the lung profile registers two CTs (4D CT phases or inspiration / expiration CT)")
        f = sitk.Clamp(isotropic(fixed, 2.0, -1000.0), sitk.sitkFloat32, *LUNG_WINDOW)
        m = sitk.Clamp(isotropic(moving, 2.0, -1000.0), sitk.sitkFloat32, *LUNG_WINDOW)
        mask = lung_mask(f)
        metric = "AdvancedNormalizedCorrelation"
    else:
        def norm(img, kind):
            return sitk.Clamp(img, sitk.sitkFloat32, -300.0, 600.0) if kind == "CT" else sitk.Cast(img, sitk.sitkFloat32)
        f, m, mask = norm(fixed, fixed_kind), norm(moving, moving_kind), None
        metric = "AdvancedNormalizedCorrelation" if fixed_kind == moving_kind else "AdvancedMattesMutualInformation"
    po = parameter_maps(mode, metric, grid_mm, 4, iterations, samples, init == "geometry", seed)
    itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(max(1, sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()))
    kw = {"fixed_mask": _to_itk(mask, np.uint8)} if mask is not None else {}
    _, result = itk.elastix_registration_method(_to_itk(f), _to_itk(m), parameter_object=po, log_to_console=False, **kw)
    return to_sitk(result), f, m, mask
