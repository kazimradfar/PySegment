"""Registration (rsw/registration.py): synthetic phantoms with a KNOWN transform; the recovered transform is
compared with the true one through the distance (mm) between the two at test points inside the body."""
import numpy as np
import pytest
import SimpleITK as sitk

from rsw import registration as rg
from rsw.io import Geometry, Volume

SPACING = (3.0, 3.0, 3.0)
SHAPE = (44, 48, 48)          # (z, y, x)


def labels() -> np.ndarray:
    z, y, x = np.ogrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    lab = np.zeros(SHAPE, np.uint8)
    lab[((z - 22) / 17) ** 2 + ((y - 24) / 18) ** 2 + ((x - 24) / 20) ** 2 <= 1] = 1          # body
    lab[(((y - 20) ** 2 + (x - 14) ** 2) <= 36) & (z > 8) & (z < 36)] = 2                     # "spine"
    for cz, cy, cx, r in [(14, 30, 32, 4.5), (28, 18, 34, 3.5), (22, 34, 16, 3.0), (30, 26, 24, 2.5)]:
        lab[(z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= r * r] = 3                        # hot lesions
    return lab


def grid() -> Geometry:
    return Geometry((SHAPE[2], SHAPE[1], SHAPE[0]), SPACING, (-70.0, -72.0, -66.0), (1, 0, 0, 0, 1, 0, 0, 0, 1))


def blur(a, s):
    return sitk.GetArrayFromImage(sitk.SmoothingRecursiveGaussian(sitk.GetImageFromArray(a.astype(np.float32)), s))


def make(kind: str, lab: np.ndarray) -> np.ndarray:
    """Different contrast per kind: CT (HU), PET and SPECT (activity, blurred; SPECT coarser)."""
    if kind == "CT":
        lut = np.array([-1000, 40, 700, 60], np.float32)
        return lut[lab] + np.random.default_rng(0).normal(0, 8, lab.shape).astype(np.float32)
    lut = np.array([0.05, 1.0, 0.3, 9.0], np.float32)
    return blur(lut[lab], 0.7 if kind == "PET" else 1.4)


def volume(kind: str, arr: np.ndarray, name: str = "") -> Volume:
    return Volume(name or kind, arr.astype(np.float32), grid(), kind, "HU" if kind == "CT" else "SUV")


def moving_from(kind: str, true_tfm: sitk.Transform):
    """Moving image such that ``true_tfm`` (fixed to moving) is the correct registration."""
    lab = labels()
    arr = make(kind, lab)
    img = grid().apply(sitk.GetImageFromArray(arr))
    inv = true_tfm.GetInverse()
    mov = sitk.Resample(img, grid().reference(), inv, sitk.sitkLinear, float(arr.min()), sitk.sitkFloat32)
    return volume(kind, sitk.GetArrayFromImage(mov)), lab


def rigid(rx=3.0, ry=-2.0, rz=4.0, t=(5.0, -4.0, 6.0)) -> sitk.Transform:
    r = sitk.Euler3DTransform()
    r.SetCenter((0.0, 0.0, 0.0))
    r.SetRotation(*np.radians([rx, ry, rz]))
    r.SetTranslation(t)
    return r


def probe_points() -> np.ndarray:
    z, y, x = np.nonzero(labels() > 0)
    sel = np.random.default_rng(1).choice(len(z), 400, replace=False)
    g = grid()
    return np.array([[g.origin[0] + x[i] * SPACING[0], g.origin[1] + y[i] * SPACING[1], g.origin[2] + z[i] * SPACING[2]]
                     for i in sel])


def tre(a: sitk.Transform, b: sitk.Transform) -> float:
    pts = probe_points()
    return float(np.mean([np.linalg.norm(np.array(a.TransformPoint(tuple(p))) - np.array(b.TransformPoint(tuple(p)))) for p in pts]))


# (fixed kind, moving kind): every pair of modalities, between and within
PAIRS = [("CT", "CT"), ("PET", "PET"), ("SPECT", "SPECT"), ("CT", "PET"), ("CT", "SPECT"), ("PET", "SPECT"), ("SPECT", "PET")]


@pytest.mark.parametrize("fk,mk", PAIRS)
def test_rigid_recovers_known_transform(fk, mk):
    t = rigid()
    mov, _ = moving_from(mk, t)
    fixed = volume(fk, make(fk, labels()))
    before = tre(sitk.Transform(3, sitk.sitkIdentity), t)
    res = rg.register(fixed, mov, mode="rigid", init="identity", levels=3, iterations=150)
    err = tre(res.transform, t)
    assert before > 6.0                      # the starting error was large
    assert err < 3.0, f"{fk}<-{mk}: mean error {err:.2f} mm (start {before:.1f} mm)"
    assert res.metric == rg.choose_metric(fk, mk)
    assert not res.folding


def test_metric_choice():
    assert rg.choose_metric("CT", "PET") == "mattes" and rg.choose_metric("SPECT", "CT") == "mattes"
    assert rg.choose_metric("PET", "PET") == "correlation" and rg.choose_metric("SPECT", "SPECT") == "correlation"
    assert rg.choose_metric("CT", "CT") == "correlation" and rg.choose_metric("MR", "MR") == "correlation"


def test_affine_recovers_scaling():
    a = sitk.AffineTransform(3)
    r = rigid(2.0, 1.0, -3.0, (4.0, 3.0, -5.0))
    a.SetMatrix(np.array(r.GetMatrix()) * 1.06)
    a.SetTranslation(r.GetTranslation())
    mov, _ = moving_from("CT", a)
    res = rg.register(volume("CT", make("CT", labels())), mov, mode="affine", init="identity", levels=3, iterations=200)
    assert tre(res.transform, a) < 3.0


def test_landmarks_initialisation_gives_exact_rigid():
    t = rigid()
    pts_f = np.array([[0, 0, 0], [40, 0, 0], [0, 40, 0], [0, 0, 40], [30, 30, 30]], float)
    pts_m = np.array([t.TransformPoint(tuple(p)) for p in pts_f])
    est = rg.landmark_transform(pts_f, pts_m)
    assert tre(est, t) < 0.01
    with pytest.raises(ValueError):
        rg.landmark_transform(pts_f[:2], pts_m[:2])


def test_bspline_improves_and_does_not_fold():
    t = rigid(1.0, 0.0, 2.0, (3.0, 2.0, -2.0))
    mov, _ = moving_from("CT", t)
    # add a smooth local deformation to the moving image
    img = grid().apply(sitk.GetImageFromArray(mov.array))
    bs = sitk.BSplineTransformInitializer(img, [4, 4, 4], order=3)
    p = np.array(bs.GetParameters())
    p[:] = np.random.default_rng(3).uniform(-3.0, 3.0, p.shape)
    bs.SetParameters(p.tolist())
    deformed = sitk.Resample(img, img, bs, sitk.sitkLinear, float(mov.array.min()))
    mov2 = volume("CT", sitk.GetArrayFromImage(deformed))
    fixed = volume("CT", make("CT", labels()))
    aff = rg.register(fixed, mov2, mode="affine", init="identity", levels=3, iterations=150)
    deform = rg.register(fixed, mov2, mode="bspline", init="identity", levels=3, iterations=150, bspline_nodes=6)
    assert deform.metric_value < aff.metric_value             # lower is better
    assert deform.jacobian is not None and not deform.folding
    # the returned transform itself must give a better image than affine, and survive a save / load round trip
    ref = make("CT", labels())
    inside = labels() > 0
    err = lambda r: np.abs(rg.resample_volume(mov2, fixed.geometry, r.transform).array - ref)[inside].mean()
    assert err(deform) < err(aff)
    import tempfile, pathlib
    out = rg.save_transform(deform.transform, pathlib.Path(tempfile.mkdtemp()) / "bs.tfm")
    back = rg.load_transform(out)
    assert np.allclose(rg.resample_volume(mov2, fixed.geometry, back).array,
                       rg.resample_volume(mov2, fixed.geometry, deform.transform).array, atol=1e-3)


def test_jacobian_detects_folding():
    ref = grid()
    img = ref.reference()
    bs = sitk.BSplineTransformInitializer(img, [4, 4, 4], order=3)
    p = np.zeros(len(bs.GetParameters()))
    p[::3] = np.where(np.arange(len(p[::3])) % 2 == 0, 60.0, -60.0)
    bs.SetParameters(p.tolist())
    assert rg.jacobian_stats(bs, ref)["fraction_nonpositive"] > 0


def test_identity_transform_leaves_grid_unchanged():
    v = volume("PET", make("PET", labels()))
    out = rg.resample_volume(v, v.geometry, None)
    assert np.allclose(out.array, v.array, atol=1e-4)


def test_hybrid_applies_ct_transform_to_functional_image():
    t = rigid()
    ct_m, lab = moving_from("CT", t)
    pet_m, _ = moving_from("PET", t)                             # PET of the same study moved the same way
    res, pet_on_fixed = rg.register_hybrid(volume("CT", make("CT", lab)), ct_m, pet_m, mode="rigid", init="identity",
                                           levels=3, iterations=150)
    ref_pet = make("PET", lab)
    inside = lab > 0
    err_after = np.abs(pet_on_fixed.array - ref_pet)[inside].mean()
    err_before = np.abs(pet_m.array - ref_pet)[inside].mean()
    assert err_after < 0.5 * err_before
    assert pet_on_fixed.info.get("registered") is True


def test_mask_propagation_keeps_labels_and_position():
    t = rigid()
    lab = labels()
    mask_fixed = (lab == 3).astype(np.uint8)
    # mask on the moving image: the fixed mask moved by the inverse of the registration
    g = grid()
    mask_img = g.apply(sitk.GetImageFromArray(mask_fixed))
    mov_mask = sitk.GetArrayFromImage(sitk.Resample(mask_img, g.reference(), t.GetInverse(), sitk.sitkNearestNeighbor, 0))
    back = rg.propagate_mask(mov_mask, g, g, t)
    inter = np.logical_and(back > 0, mask_fixed > 0).sum()
    dice = 2 * inter / ((back > 0).sum() + mask_fixed.sum())
    assert set(np.unique(back)) <= {0, 1} and dice > 0.8


def test_quantification_check_reports_losses():
    v = volume("PET", make("PET", labels()))
    same = rg.quantification_check(v, rg.resample_volume(v, v.geometry, None))
    assert abs(same["total_change_pct"]) < 0.01
    moved = rg.resample_volume(v, v.geometry, rigid(0, 0, 0, (5, 4, -6)))
    q = rg.quantification_check(v, moved)
    assert abs(q["total_change_pct"]) < 2.0                      # object stays inside the field of view
    far = rg.resample_volume(v, v.geometry, rigid(0, 0, 0, (150, 0, 0)))
    assert rg.quantification_check(v, far)["total_change_pct"] < -20     # object left the field of view


def test_transform_files_roundtrip_and_description(tmp_path):
    t = rigid()
    p = rg.save_transform(t, tmp_path / "t")
    assert p.suffix == ".tfm"
    back = rg.load_transform(p)
    assert tre(back, t) < 1e-6
    m = rg.transform_matrix(t)
    assert m.shape == (4, 4) and np.allclose(m[:3, 3], t.GetTranslation())
    assert "rotation" in rg.describe(t)


def test_counts_are_conserved_when_voxel_size_changes():
    from rsw.io import Geometry, Volume as V
    lab = labels()
    src = V("sp", make("SPECT", lab), grid(), "SPECT", "CNTS")
    coarse = Geometry((24, 22, 22), (6.0, 6.0, 6.0), (-70.0, -72.0, -66.0), (1, 0, 0, 0, 1, 0, 0, 0, 1))
    out = rg.resample_volume(src, coarse, None)
    assert abs(out.array.sum() / src.array.sum() - 1) < 0.1          # total counts kept (field of view covers the object)
    q = rg.quantification_check(src, out)
    assert abs(q["total_change_pct"]) < 10


def test_translation_composite_is_described_as_linear():
    comp = sitk.CompositeTransform(3)
    comp.AddTransform(sitk.TranslationTransform(3, (1.0, 2.0, 3.0)))
    m = rg.transform_matrix(comp)
    assert m is not None and np.allclose(m[:3, 3], (1, 2, 3))


# ---------------------------------------------------------------------------------- start position and safeguards
def test_auto_start_keeps_scanner_coordinates_when_images_overlap():
    fixed = rg.to_image(volume("CT", make("CT", labels())))
    other = sitk.Image(fixed)
    assert rg.overlap_fraction(fixed, other) == pytest.approx(1.0)
    assert rg.resolve_init(fixed, other, "auto") == "identity"
    far = sitk.Image(fixed)
    far.SetOrigin(tuple(np.array(fixed.GetOrigin()) + (0.0, 0.0, 500.0)))       # another coordinate system
    assert rg.overlap_fraction(fixed, far) == 0.0
    assert rg.resolve_init(fixed, far, "auto") == "geometry"
    assert rg.resolve_init(fixed, far, "identity") == "identity"               # a forced start is kept


def test_result_is_never_worse_than_the_start(monkeypatch):
    """When the optimiser ends at a worse match than the start (seen on real cropped PET/CT), the start is kept."""
    lab = labels()
    fixed, mov = volume("CT", make("CT", lab)), volume("PET", make("PET", lab))   # already aligned

    class Diverging(sitk.ImageRegistrationMethod):
        def Execute(self, f, m):
            return sitk.TranslationTransform(3, (25.0, 0.0, 0.0))                 # a wrong end point

    monkeypatch.setattr(rg.sitk, "ImageRegistrationMethod", Diverging)
    res = rg.register(fixed, mov, mode="rigid", init="identity")
    assert res.kept_start and res.init == "identity"
    assert np.allclose(rg.transform_matrix(res.transform), np.eye(4))
    assert any("worse match than the start" in w for w in res.warnings)
    assert res.metric_value == pytest.approx(res.start_metric_value)


def test_metric_does_not_get_worse_on_a_normal_run():
    lab = labels()
    fixed = volume("CT", make("CT", lab))
    mov, _ = moving_from("PET", rigid())
    res = rg.register(fixed, mov, mode="rigid", iterations=150)
    assert res.init == "identity" and not res.kept_start
    assert res.metric_value <= res.start_metric_value


def test_large_correction_is_measured_at_the_image_centre():
    img = rg.to_image(volume("CT", make("CT", labels())))
    start = sitk.Euler3DTransform()
    assert rg.correction_mm(start, sitk.TranslationTransform(3, (0.0, 24.0, 7.0)), img) == pytest.approx(25.0)
    assert rg.correction_mm(start, start, img) == 0.0


def test_landmark_fit_follows_the_fitzpatrick_prediction():
    """Rigid landmark fit with localisation noise: observed TRE matches the theory used in the validation."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("vr", Path(__file__).resolve().parents[1] / "tools" / "validate_registration.py")
    vr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vr)
    rng = np.random.default_rng(3)
    pts = np.array([[0, 0, 0], [60, 0, 0], [0, 50, 0], [0, 0, 40], [-40, -30, 10], [30, 30, -30]], float)
    target = np.array([20.0, 10.0, 5.0])
    true = rigid()
    moving_pts = vr.tp(true, pts)
    errs = []
    for _ in range(400):
        t = rg.landmark_transform(pts + rng.normal(0, 1, pts.shape), moving_pts + rng.normal(0, 1, pts.shape))
        errs.append(np.linalg.norm(np.array(t.TransformPoint(tuple(target))) - np.array(true.TransformPoint(tuple(target)))))
    observed = np.sqrt(np.mean(np.square(errs)))
    predicted = np.sqrt(vr.fitzpatrick_tre2(pts, target, 2 * 3 * 1.0 ** 2))
    assert 0.85 < observed / predicted < 1.15


def test_metric_check_uses_the_same_voxels_for_large_images(monkeypatch):
    """Whole-body CT: the start/end comparison uses a fixed regular sample, so it is repeatable and fast."""
    img = rg.to_image(volume("CT", make("CT", labels())))
    monkeypatch.setattr(rg, "METRIC_CHECK_VOXELS", 20_000)               # the phantom has about 93 000 voxels
    t = sitk.TranslationTransform(3, (3.0, 0.0, 0.0))
    a = rg.metric_value(img, img, "correlation", t)
    assert a == rg.metric_value(img, img, "correlation", t)               # identical voxels each call
    assert rg.metric_value(img, img, "correlation", sitk.Transform()) < a  # aligned is better


def test_metric_samples_are_capped_for_large_images():
    """Whole-body CT: 20 % of 40 million voxels would be 8 million metric samples per iteration (7 min for a rigid
    registration of a 512 x 512 x 161 CT); the number is capped. Small images keep the 20 %."""
    reg = sitk.ImageRegistrationMethod()
    big = sitk.Image([512, 512, 161], sitk.sitkUInt8)
    rg._cap_samples(reg, big, 0.2, [4, 2, 1], 1)
    per = reg.GetMetricSamplingPercentagePerLevel()
    n = 512 * 512 * 161
    assert per[0] == pytest.approx(0.2)                                  # coarse level: 20 % is below the cap
    assert per[2] * n == pytest.approx(rg.MAX_SAMPLES)                   # full resolution: capped
    small = sitk.Image([188, 233, 73], sitk.sitkUInt8)                   # cropped CT of validation/registration
    rg._cap_samples(reg, small, 0.2, [4, 2, 1], 1)
    assert list(reg.GetMetricSamplingPercentagePerLevel()) == pytest.approx([0.2, 0.2, 0.2])


# ---------------------------------------------------------------------------------- optional elastix engine
def test_elastix_result_converts_exactly_to_simpleitk(tmp_path):
    """The elastix result (affine + B-spline) as SimpleITK transforms maps points exactly like transformix."""
    itk = pytest.importorskip("itk")
    from rsw import registration_elastix as rx
    if not rx.available():
        pytest.skip("itk-elastix not installed")
    fixed = rg.to_image(volume("CT", make("CT", labels())))
    mov, _ = moving_from("CT", rigid())
    m = rg.to_image(mov)
    po = rx.parameter_maps("bspline", "AdvancedNormalizedCorrelation", 15.0, 2, 50, 2000, False, 7)
    _, result = itk.elastix_registration_method(rx._to_itk(fixed), rx._to_itk(m), parameter_object=po, log_to_console=False)
    t = rx.to_sitk(result)
    pts = np.array([[-20.0, -10.0, 5.0], [15.0, 20.0, -12.0], [0.0, 0.0, 0.0], [30.0, -25.0, 18.0]])
    f = tmp_path / "p.txt"
    f.write_text("point\n4\n" + "\n".join(" ".join(f"{v:.4f}" for v in p) for p in pts))
    itk.transformix_pointset(rx._to_itk(m), result, fixed_point_set_file_name=str(f), output_directory=str(tmp_path))
    ref = np.array([[float(v) for v in line.split("OutputPoint = [")[1].split("]")[0].split()]
                    for line in (tmp_path / "outputpoints.txt").read_text().splitlines()])
    got = np.array([t.TransformPoint(tuple(p)) for p in pts])
    assert np.abs(got - ref).max() < 1e-4


def test_elastix_engine_recovers_a_known_transform():
    pytest.importorskip("itk")
    from rsw import registration_elastix as rx
    if not rx.available():
        pytest.skip("itk-elastix not installed")
    lab = labels()
    fixed = volume("CT", make("CT", lab))
    true = rigid()
    mov, _ = moving_from("CT", true)
    res = rg.register(fixed, mov, mode="affine", engine="elastix", profile="general")
    body = np.argwhere(lab > 0)[::50]
    g = grid().reference()
    p = np.array([g.TransformContinuousIndexToPhysicalPoint([float(i[2]), float(i[1]), float(i[0])]) for i in body])
    err = np.linalg.norm(np.array([res.transform.TransformPoint(tuple(q)) for q in p])
                         - np.array([true.TransformPoint(tuple(q)) for q in p]), axis=1)
    assert err.mean() < 1.5 and not res.kept_start and res.metric.startswith("elastix")
    with pytest.raises(ValueError, match="two CTs"):
        rg.register(fixed, volume("PET", make("PET", lab)), mode="rigid", engine="elastix", profile="lung")


def test_lung_mask_when_the_airways_reach_the_edge_of_the_image():
    """POPI patient dx: the lungs are connected to the air outside the body (airways at the image edge); the first
    lung mask (air not touching the sides) then held no lung at all. The body-based mask finds both lungs."""
    from rsw import registration_elastix as rx
    z, y, x = np.mgrid[:40, :80, :80]
    ct = np.full((40, 80, 80), -1000.0, np.float32)                                     # air around the body
    body = ((y - 40) / 34.0) ** 2 + ((x - 40) / 36.0) ** 2 <= 1
    ct[body] = 40.0
    lung_l = ((y - 40) / 20.0) ** 2 + ((x - 24) / 11.0) ** 2 + ((z - 20) / 15.0) ** 2 <= 1
    lung_r = ((y - 40) / 20.0) ** 2 + ((x - 56) / 11.0) ** 2 + ((z - 20) / 15.0) ** 2 <= 1
    ct[lung_l | lung_r] = -850.0
    ct[:22, 38:42, 39:42] = -1000.0                                                     # airway up to the top edge
    ct[5:8, 38:42, 0:42] = -1000.0                                                      # ... and out to the side
    img = sitk.GetImageFromArray(ct)
    img.SetSpacing((2.0, 2.0, 2.0))
    m = sitk.GetArrayFromImage(rx.lung_mask(img, dilate_mm=2.0)).astype(bool)
    assert m[lung_l].mean() > 0.99 and m[lung_r].mean() > 0.99
    assert not m[0, 0, 0] and m.sum() < 1.5 * (lung_l | lung_r).sum()                   # no outside air


def test_folding_warning_never_reads_zero_percent():
    """POPI: 2e-6 of the voxels folded and the warning said '0.00 %'. The count and two significant digits are shown."""
    from rsw.registration import folding_text
    assert folding_text({"n_nonpositive": 3, "fraction_nonpositive": 2.04e-6}) == "3 voxels (0.0002 %)"


def test_stopping_at_the_iteration_limit_gives_a_warning():
    """51 cropped STS CTs: with 150 iterations rigid CT->CT often stopped at the limit (TRE 1.4-2 mm instead of
    0.2 mm with 300-400 iterations on a POPI CT slab of the same thickness). Stopping at the limit is now reported."""
    lab = labels()
    mov, _ = moving_from("CT", rigid())
    fixed = volume("CT", make("CT", lab))
    res = rg.register(fixed, mov, mode="rigid", init="identity", iterations=2)
    assert any(w.startswith("Not converged") for w in res.warnings)
    assert not any(w.startswith("Not converged")
                   for w in rg.register(fixed, mov, mode="rigid", init="identity").warnings)


def test_affine_starts_from_the_rigid_result_and_is_not_worse():
    """Affine (and the affine stage of B-spline) starts from a rigid registration, so it can never end at a worse
    match than rigid registration of the same pair (51 STS CTs, affine from identity: median TRE 4.03 mm; rigid-first 0.30 mm)."""
    lab = labels()
    mov, _ = moving_from("CT", rigid())
    fixed = volume("CT", make("CT", lab))
    r = rg.register(fixed, mov, mode="rigid", init="identity")
    a = rg.register(fixed, mov, mode="affine", init="identity")
    assert a.metric_value <= r.metric_value * (1 + 1e-3)
    assert tre(a.transform, rigid()) < 1.0
    assert isinstance(rg.linear_as_affine(r.transform, rg.to_image(fixed)), sitk.AffineTransform)


def test_iteration_limit_with_a_stable_result_is_not_reported_as_not_converged():
    """Re-run of the 51 STS patients: with 500 iterations the limit was reached in 33/51 CT and 45/51 PET rigid
    registrations although the TRE was 0.4-0.8 mm, so the warning said nothing. Now it is given only if the image
    corners still move (> 0.05 mm over the last 20 iterations)."""
    from rsw.registration import still_moving_mm
    still = [(2, np.zeros((8, 3)) + 1e-4 * i) for i in range(60)]
    moving = [(2, np.zeros((8, 3)) + 0.01 * i) for i in range(60)]
    assert still_moving_mm(still) < 0.05 < still_moving_mm(moving)
    assert still_moving_mm([(0, np.zeros((8, 3)) + i) for i in range(60)] + [(1, np.zeros((8, 3)))] * 5) == 0.0
    assert still_moving_mm([(1, np.zeros((8, 3))), (1, np.ones((8, 3)))]) > 1      # short level: whole level
