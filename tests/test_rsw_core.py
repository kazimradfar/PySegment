"""Core library of the desktop workstation (rsw/): synthetic data only."""
import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from rsw import io, metrics, seg


def _img(arr, spacing=(2.0, 2.0, 3.0), origin=(-10.0, 5.0, 20.0)):
    im = sitk.GetImageFromArray(arr)
    im.SetSpacing(spacing)
    im.SetOrigin(origin)
    return im


@pytest.fixture
def pet(tmp_path):
    a = np.full((20, 30, 30), 1.0, np.float32)
    a[8:12, 10:16, 10:16] = 8.0          # hot lesion
    a[9:11, 12:14, 12:14] = 12.0         # its peak
    a[3:5, 24:27, 24:27] = 5.0           # second lesion
    p = tmp_path / "PET_SUV.nii.gz"
    sitk.WriteImage(_img(a), str(p))
    return p


def test_load_kind_unit_and_geometry(pet, tmp_path):
    v = io.load_volume(pet)
    assert v.kind == "PET" and v.unit == "SUV" and v.shape == (20, 30, 30)
    assert v.geometry.spacing == (2.0, 2.0, 3.0) and v.geometry.voxel_ml == pytest.approx(0.012)
    ct = tmp_path / "CT.nii.gz"
    sitk.WriteImage(_img(np.zeros((40, 60, 60), np.float32), spacing=(1.0, 1.0, 1.5), origin=(-10.5, 4.5, 19.0)), str(ct))
    c = io.load_volume(ct)
    assert c.kind == "CT" and c.unit == "HU"
    r = io.resample_to(c, v.geometry)
    assert r.shape == v.shape and r.geometry.same_grid(v.geometry)


def test_save_mask_geometry_backup_history(pet, tmp_path):
    v = io.load_volume(pet)
    m = np.zeros(v.shape, np.uint8)
    m[8:12, 10:16, 10:16] = 1
    out = tmp_path / "case" / "lesions_reader1.nii.gz"
    r1 = io.save_mask(m, v.geometry, out, reader="A")
    assert r1["voxels"] == 4 * 6 * 6 and r1["backup"] == ""
    back = sitk.ReadImage(str(out))
    assert back.GetPixelID() == sitk.sitkUInt8 and io.Geometry.of(back).same_grid(v.geometry, tol=1e-9)
    r2 = io.save_mask(m, v.geometry, out, reader="A")
    assert Path(r2["backup"]).exists() and Path(r2["backup"]).parent.name == "backup"
    hist = [json.loads(l) for l in (out.parent / "mask_history.jsonl").read_text().splitlines()]
    assert len(hist) == 2 and hist[0]["reader"] == "A"
    loaded, resampled = io.load_mask(out, v.geometry)
    assert not resampled and int(loaded.sum()) == r1["voxels"]
    with pytest.raises(ValueError):
        io.save_mask(m[:-1], v.geometry, tmp_path / "bad.nii.gz")


def test_brush_disk_sphere_threshold_and_undo():
    shape, sp = (20, 30, 30), (2.0, 2.0, 3.0)
    layer = seg.MaskLayer("m", np.zeros(shape, np.uint8))
    reg, box = seg.brush_region(shape, (10, 15, 15), 4.0, sp, view="axial")
    assert reg.shape[0] == 1                                  # one axial slice
    assert layer.apply(reg, box, 1) == int(reg.sum()) > 0
    reg3, box3 = seg.brush_region(shape, (10, 15, 15), 6.0, sp, sphere=True)
    assert reg3.shape[0] > 1                                  # several slices
    img = np.zeros(shape, np.float32); img[10, 15, 15:] = 5
    regt, _ = seg.brush_region(shape, (10, 15, 15), 4.0, sp, image=img, low=2.5)
    assert int(regt.sum()) == 3                               # only the hot voxels inside the disk
    n = layer.voxels
    layer.apply(reg3, box3, 1)
    assert layer.voxels > n and layer.undo() and layer.voxels == n
    assert layer.redo() and layer.voxels > n
    layer.locked = True
    with pytest.raises(PermissionError):
        layer.apply(reg, box, 0)


def test_region_grow_percent_and_absolute(pet):
    v = io.load_volume(pet)
    region, box, thr = seg.region_grow(v.array, (10, 11, 11), v.geometry.spacing, "percent", 41.0)
    assert thr == pytest.approx(12.0 * 0.41)
    full = np.zeros(v.shape, bool); full[box] = region
    assert full.sum() == 4 * 6 * 6 and not full[3:5, 24:27, 24:27].any()   # only the clicked lesion
    region, box, thr = seg.region_grow(v.array, (4, 25, 25), v.geometry.spacing, "absolute", 2.5)
    full = np.zeros(v.shape, bool); full[box] = region
    assert full.sum() == 2 * 3 * 3


def test_islands_and_lesions(pet):
    v = io.load_volume(pet)
    layer = seg.MaskLayer("m", (v.array > 4).astype(np.uint8))
    layer.array[0, 0, 0] = 1                                     # tiny island
    les = seg.lesions(layer.array, v.array, v.geometry.spacing)
    assert len(les) == 3 and max(l["max"] for l in les) == 12.0
    assert seg.remove_islands(layer, 5) == 1 and len(seg.lesions(layer.array, None, v.geometry.spacing)) == 2
    assert layer.undo() and layer.array[0, 0, 0] == 1


def test_agreement_metrics():
    a = np.zeros((10, 10, 10), np.uint8); a[2:6, 2:6, 2:6] = 1
    b = np.zeros_like(a); b[3:7, 2:6, 2:6] = 1
    r = metrics.agreement(a, b, (1.0, 1.0, 1.0))
    assert r["dice"] == pytest.approx(0.75) and r["jaccard"] == pytest.approx(0.6)
    assert r["hd95_mm"] == pytest.approx(1.0) and r["volume_diff_pct"] == pytest.approx(0.0)
    assert np.isnan(metrics.hd95(a, a * 0, (1, 1, 1)))


def test_icc_matches_pipeline_and_reference_values():
    """Same numbers as the analysis pipeline (step 6) and as the Shrout & Fleiss (1979) example."""
    pipeline_icc = pytest.importorskip("pipeline.step6_reproducibility").icc_a1   # not in the stand-alone repository
    rng = np.random.default_rng(1)
    y = rng.normal(size=(30, 1)) + rng.normal(scale=0.3, size=(30, 2))
    assert metrics.icc_a1(y) == pytest.approx(pipeline_icc(y), nan_ok=True)
    sf = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], float)
    assert metrics.icc_a1(sf)[0] == pytest.approx(0.29, abs=0.005)       # ICC(2,1) in Shrout & Fleiss
    assert metrics.icc_category(0.8) == "good (0.75–0.90)"


def test_canonical_orientation_views_write_through():
    from rsw.orient import Orientation, plane_to_index, view_slice
    a = np.arange(4 * 5 * 6).reshape(4, 5, 6).astype(np.float32)          # (z, y, x)
    # identity LPS: z grows superior -> canonical axis 0 (inferior) is flipped
    o = Orientation.from_direction((1, 0, 0, 0, 1, 0, 0, 0, 1), (1.0, 2.0, 3.0))
    c = o.canonical(a)
    assert c.shape == (4, 5, 6) and c[0, 0, 0] == a[3, 0, 0] and o.spacing_zyx == (3.0, 2.0, 1.0)
    assert np.shares_memory(c, a)
    c[0, 1, 2] = -1                                                          # edit through the view
    assert a[o.to_original((0, 1, 2), a.shape)] == -1 and a[3, 1, 2] == -1
    # RAS-stored image (x grows to the right, y anterior): left/posterior axes flipped
    o2 = Orientation.from_direction((-1, 0, 0, 0, -1, 0, 0, 0, 1), (1.0, 1.0, 1.0))
    assert o2.flips == (True, True, True)
    # sagittal-acquired image: index x along S, y along P, z along L
    o3 = Orientation.from_direction((0, 0, 1, 0, 1, 0, 1, 0, 0), (1.0, 2.0, 3.0))
    assert o3.order == (2, 1, 0) and o3.spacing_zyx == (1.0, 2.0, 3.0)
    assert view_slice(c, "coronal", 2).shape == (4, 6) and plane_to_index("sagittal", 3, 1, 2) == (1, 2, 3)


def test_brush_stroke_is_one_undo_step():
    layer = seg.MaskLayer("m", np.zeros((5, 20, 20), np.uint8))
    layer.begin_group((slice(2, 3), slice(None), slice(None)))
    for c in range(3, 15, 2):
        reg, box = seg.brush_region(layer.array.shape, (2, 10, c), 2.0, (1, 1, 1))
        layer.apply(reg, box, 1)
    assert layer.end_group() and layer.voxels > 20 and not layer.saved
    assert layer.undo() and layer.voxels == 0 and not layer.undo()


def test_export_images_as_nifti_own_grid_and_pet_grid(tmp_path):
    """DICOM -> NIfTI export: PET (SUV) and CT keep their values and geometry, or are put on the PET grid."""
    import SimpleITK as sitk
    from rsw import io
    from rsw.session import Session
    pet_geo = io.Geometry((16, 16, 10), (4.0, 4.0, 4.0), (0.0, 0.0, 0.0), (1, 0, 0, 0, 1, 0, 0, 0, 1))
    ct_geo = io.Geometry((32, 32, 20), (2.0, 2.0, 2.0), (0.0, 0.0, 0.0), (1, 0, 0, 0, 1, 0, 0, 0, 1))
    pet = np.zeros((10, 16, 16), np.float32)
    pet[4:6, 6:10, 6:10] = 7.5
    ct = np.full((20, 32, 32), -1000.0, np.float32)
    ct[:, 8:24, 8:24] = 40.0
    s = Session(None, None, [io.Volume("PET", pet, pet_geo, "PET", "SUV"), io.Volume("CT", ct, ct_geo, "CT", "HU")])
    own = s.export_images(tmp_path / "own")
    assert sorted(p.name for p in own) == ["CT.nii.gz", "PET_SUV.nii.gz"]
    p_img, c_img = sitk.ReadImage(str(tmp_path / "own" / "PET_SUV.nii.gz")), sitk.ReadImage(str(tmp_path / "own" / "CT.nii.gz"))
    assert np.allclose(sitk.GetArrayFromImage(p_img), pet) and p_img.GetSpacing() == (4.0, 4.0, 4.0)
    assert c_img.GetPixelID() == sitk.sitkInt16 and c_img.GetSize() == (32, 32, 20)
    assert np.array_equal(sitk.GetArrayFromImage(c_img), ct.astype(np.int16))
    s.export_images(tmp_path / "grid", on_reference_grid=True)
    c2 = sitk.ReadImage(str(tmp_path / "grid" / "CT.nii.gz"))
    assert c2.GetSize() == (16, 16, 10) and c2.GetSpacing() == (4.0, 4.0, 4.0)
