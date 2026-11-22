"""Projects, radiomics and reader agreement of the desktop workstation (synthetic data only)."""
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from rsw import analysis, io, project
from rsw.radiomics import Settings, extract, save_table


def _write(arr, path, spacing=(4.0, 4.0, 4.0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    im = sitk.GetImageFromArray(arr)
    im.SetSpacing(spacing)
    sitk.WriteImage(im, str(path))


def _pet(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(1.0, 0.1, (24, 32, 32)).astype(np.float32)
    a[8:14, 8:16, 8:16] += rng.normal(8, 1.5, (6, 8, 8))     # lesion 1 (heterogeneous)
    a[16:19, 20:24, 20:24] += 4                                # lesion 2 (small)
    return a


def _generic(root: Path, n=4):
    for i in range(n):
        c = root / f"case{i:02d}"
        pet = _pet(i)
        _write(pet, c / "PET_SUV.nii.gz")
        _write(np.zeros_like(pet), c / "CT.nii.gz")
        auto = (pet > 4).astype(np.uint8)
        _write(auto, c / "lesion_auto.nii.gz")
        r1 = auto.copy(); r1[8:14, 8:16, 15] = 0                # reader 1 trims one edge
        r2 = auto.copy(); r2[8:14, 16, 8:16] = 1                # reader 2 adds one edge
        _write(r1, c / "masks" / "lesions_reader1.nii.gz")
        _write(r2, c / "masks" / "lesions_reader2.nii.gz")
    return root


def test_generic_layout_scan_save_load(tmp_path):
    p = project.scan(_generic(tmp_path / "study"))
    assert p.layout == "generic" and [c.id for c in p.cases] == ["case00", "case01", "case02", "case03"]
    c = p.cases[0]
    assert set(c.images) == {"PET_SUV", "CT"} and c.reference == "PET_SUV"
    assert set(c.masks) == {"lesion_auto", "lesions_reader1", "lesions_reader2"}
    p.save()
    q = project.open_project(tmp_path / "study")
    assert [x.id for x in q.cases] == [x.id for x in p.cases]


def test_reader_start_and_blinding(tmp_path):
    p = project.scan(_generic(tmp_path / "s", n=1))
    c = p.cases[0]
    assert p.start_mask(c, "reader1") == ("continue", "case00/masks/lesions_reader1.nii.gz")
    (tmp_path / "s" / "case00" / "masks" / "lesions_reader2.nii.gz").unlink()
    p = p.rescan(); c = p.cases[0]
    assert p.start_mask(c, "reader2") == ("auto", "case00/lesion_auto.nii.gz")      # never reader 1's mask
    vis = p.visible_masks(c, "reader2", blind=True)
    assert "lesions_reader1" not in vis and "lesion_auto" in vis
    assert "lesions_reader1" in p.visible_masks(c, "reader2", blind=False)


def test_prosper_layout(tmp_path):
    root = tmp_path / "ws"
    _write(_pet(), root / "data" / "nifti" / "SYN001" / "PET_SUV.nii.gz")
    _write(np.zeros((24, 32, 32), np.uint8), root / "data" / "masks" / "SYN001" / "tumor_mask_auto.nii.gz")
    p = project.scan(root)
    assert p.layout == "prosper" and p.cases[0].reference == "PET_SUV"
    assert p.reader_mask_rel(p.cases[0], "reader1") == "data/masks/SYN001/tumor_mask_reader1.nii.gz"
    assert p.start_mask(p.cases[0], "reader1")[0] == "auto"


def test_single_folder_is_one_case(tmp_path):
    _write(_pet(), tmp_path / "one" / "pet.nii.gz")
    p = project.scan(tmp_path / "one")
    assert len(p.cases) == 1 and p.cases[0].folder == "."
    assert p.reader_mask_rel(p.cases[0], "reader1") == "./masks/lesions_reader1.nii.gz"


def test_radiomics_whole_and_per_lesion(tmp_path):
    pet = _pet()
    geo = io.Geometry((32, 32, 24), (4.0, 4.0, 4.0), (0.0, 0.0, 0.0), (1, 0, 0, 0, 1, 0, 0, 0, 1))
    mask = (pet > 4).astype(np.uint8)
    s = Settings(resample_mm=None)
    df = extract(pet, geo, mask, "c1", "auto", s)
    assert list(df["lesion"]) == ["all", "1", "2"]
    assert df.loc[0, "original_firstorder_Maximum"] == pytest.approx(float(pet[mask > 0].max()), rel=1e-4)
    small = df[df["lesion"] == "2"].iloc[0]
    assert small["voxels"] == 48 and pd_isnan(small.get("original_glcm_Contrast"))   # < 64 voxels: no texture
    assert np.isfinite(df.loc[1, "original_glcm_Contrast"])
    side = save_table(df, s, tmp_path / "out" / "features.csv")
    assert side.exists() and "binWidth" in side.read_text()


def pd_isnan(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def test_project_agreement_and_icc(tmp_path):
    p = project.scan(_generic(tmp_path / "study", n=5))
    ag = analysis.agreement_project(p, "reader1", "reader2")
    assert len(ag) == 5 and (ag["dice"] > 0.8).all() and (ag["dice"] < 1).all()
    tab, fa, fb = analysis.icc_project(p, "reader1", "reader2", Settings(resample_mm=None, classes=["shape", "firstorder"]))
    assert len(fa) == len(fb) == 5 and not tab.empty and tab["kept"].any()
    assert "features" in analysis.summary_icc(tab)


def test_session_reader_workflow_saves_on_original_grid(tmp_path):
    """Open a case, start Reader 2 (blinded), paint and grow in canonical views, save: the file is on the
    exact PET grid and contains the edits at the right original voxels."""
    from rsw.session import Session
    root = _generic(tmp_path / "study", n=1)
    (root / "case00" / "masks" / "lesions_reader2.nii.gz").unlink()
    p = project.scan(root)
    s = Session.open_case(p, p.cases[0], reader="reader2", blind=True)
    assert "lesions_reader1" not in s.layers and "lesion_auto" in s.layers
    name, source = s.start_reader("reader2")
    assert source == "auto" and s.active == name == "lesions_reader2"
    auto_vox = s.layer().voxels
    s2 = Session.open_case(p, p.cases[0])
    a0 = s.shape[0] - 1 - 17                                    # identity direction: canonical axis 0 flipped
    n_grow, thr = s.grow("axial", a0, 21, 21)
    assert s.layer().voxels >= auto_vox and not s.layer().saved
    n = s.paint("axial", a0, 2, 2, radius_mm=4.0)
    assert n > 0
    res = s.save_layer()
    saved = sitk.ReadImage(res["path"])
    assert io.Geometry.of(saved).same_grid(s.geo, tol=1e-9)
    arr = sitk.GetArrayFromImage(saved)
    assert arr[17, 2, 2] == 1 and int(arr.sum()) == s.layer().voxels       # edit landed on original voxel
    assert s.layer().saved and s.unsaved() == []
    assert s2.value_at(a0, 21, 21)["PET_SUV"] > 4
    from rsw.seg import RegionTooLarge
    with pytest.raises(RegionTooLarge):                         # a click in the background must not flood
        s.grow("axial", a0, 2, 30, "absolute", 0.5)


def test_region_grow_refuses_a_threshold_near_the_local_background():
    """QIBA DROmtv: with 2:1 contrast and noise, 50 % of the maximum lies at background level and the region leaked
    into the background (254 mL for an 11.5 mL sphere) without any error. Now refused; a 4:1 lesion still works."""
    import numpy as np
    import pytest
    from rsw.seg import RegionTooLarge, region_grow
    rng = np.random.default_rng(1)
    z, y, x = np.ogrid[:40, :40, :40]
    ball = (z - 20) ** 2 + (y - 20) ** 2 + (x - 20) ** 2 <= 36
    for contrast, ok in ((4.0, True), (2.0, False)):
        img = (1.0 + (contrast - 1.0) * ball + rng.normal(0, 0.08, ball.shape)).astype(np.float32)
        if ok:
            reg, _, _ = region_grow(img, (20, 20, 20), (2.0, 2.0, 2.0), "percent", 50.0)
            assert 0.8 < reg.sum() / ball.sum() < 1.3
        else:
            with pytest.raises(RegionTooLarge, match="too close to the background"):
                region_grow(img, (20, 20, 20), (2.0, 2.0, 2.0), "percent", 50.0)
