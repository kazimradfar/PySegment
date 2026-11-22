"""Registration plugged into io / session / GUI, and SPECT (NM) reading. Synthetic data only."""
import numpy as np
import pytest
import SimpleITK as sitk

from rsw import io
from rsw import registration as rg
from rsw.session import Session
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from tests.test_rsw_registration import grid, labels, make, rigid, volume, moving_from, tre


@pytest.fixture
def app():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def write_nm(folder, arr, spacing=(4.5, 4.5, 4.5), origin=(-100.0, -90.0, -80.0), nested=True):
    """Multi-frame NM file (reconstructed SPECT) with the geometry in the DetectorInformationSequence."""
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    folder.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.20"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(folder / "nm.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Modality, ds.SOPClassUID, ds.SOPInstanceUID = "NM", meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = generate_uid()
    ds.Rows, ds.Columns, ds.NumberOfFrames = arr.shape[1], arr.shape[2], arr.shape[0]
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit, ds.PixelRepresentation = 15, 0
    ds.PixelSpacing = [spacing[1], spacing[0]]
    ds.SpacingBetweenSlices = spacing[2]
    ds.Units = "CNTS"
    det = Dataset()
    det.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    det.ImagePositionPatient = list(origin)
    ds.DetectorInformationSequence = [det]
    ds.PixelData = arr.astype("<u2").tobytes()
    ds.is_little_endian, ds.is_implicit_VR = True, False
    ds.save_as(str(folder / "nm.dcm"), enforce_file_format=True) if hasattr(ds, "save_as") else None
    return folder


def test_guess_kind_spect():
    assert io.guess_kind("SPECT_recon") == "SPECT" and io.guess_kind("nm_lu177") == "SPECT"
    assert io.guess_kind("PET_SUV") == "PET" and io.guess_kind("CT") == "CT"


def test_read_nm_multiframe_keeps_counts_and_geometry(tmp_path):
    arr = (np.random.default_rng(0).random((10, 12, 14)) * 1000).astype(np.uint16)
    v = io.load_volume(write_nm(tmp_path / "spect", arr))
    assert v.kind == "SPECT" and v.unit == "CNTS" and v.array.shape == (10, 12, 14)
    assert np.allclose(v.array, arr)                                   # counts untouched, no SUV
    assert v.geometry.spacing == (4.5, 4.5, 4.5) and v.geometry.origin == (-100.0, -90.0, -80.0)
    assert not v.info                                                  # nothing like injected dose invented


def test_resample_to_identity_transform_and_none_agree(tmp_path):
    v = volume("PET", make("PET", labels()))
    small = grid()
    ref = io.Geometry((20, 20, 20), (6.0, 6.0, 6.0), (-60.0, -60.0, -60.0), (1, 0, 0, 0, 1, 0, 0, 0, 1))
    a = io.resample_to(v, ref)
    b = io.resample_to(v, ref, transform=sitk.Transform(3, sitk.sitkIdentity))
    assert np.allclose(a.array, b.array) and b.info.get("registered") is True
    assert io.resample_to(v, small).array is v.array                   # no transform, same grid: unchanged


def test_load_mask_with_transform_keeps_labels(tmp_path):
    t = rigid(0, 0, 0, (6.0, 0.0, 0.0))
    lab = (labels() == 3).astype(np.uint8)
    g = grid()
    im = g.apply(sitk.GetImageFromArray(lab))
    sitk.WriteImage(im, str(tmp_path / "m.nii.gz"))
    arr, resampled = io.load_mask(tmp_path / "m.nii.gz", g, transform=t)
    assert resampled and set(np.unique(arr)) <= {0, 1}
    assert arr.sum() == lab.sum() or abs(int(arr.sum()) - int(lab.sum())) < 0.1 * lab.sum()
    # the mask moved by the transform: centre of mass shifts by about -6 mm / 3 mm = 2 voxels along x
    dx = np.argwhere(arr)[:, 2].mean() - np.argwhere(lab)[:, 2].mean()
    assert abs(dx + 2.0) < 0.5            # sign matters: the mask moves with the transform's inverse direction


def test_session_transform_changes_view_not_native_and_spect_is_functional():
    t = rigid()
    ct_m, lab = moving_from("CT", t)
    pet = volume("PET", make("PET", lab))
    sp = io.Volume("SPECT_x", make("SPECT", lab), grid(), "SPECT", "CNTS")
    s = Session(None, None, [pet, io.Volume("CT_moving", ct_m.array, grid(), "CT", "HU"), sp])
    assert s.ref.name == "PET" and s.pet_name == "PET"
    native_before = s.native["CT_moving"].array.copy()
    s.set_transform("CT_moving", t)
    assert np.array_equal(s.native["CT_moving"].array, native_before)
    ref_ct = make("CT", lab)
    inside = lab > 0
    assert np.abs(s.volumes["CT_moving"].array - ref_ct)[inside].mean() < 0.5 * np.abs(native_before - ref_ct)[inside].mean()
    with pytest.raises(ValueError):
        s.set_transform("PET", t)
    s.clear_transform("CT_moving")
    assert np.allclose(s.volumes["CT_moving"].array, native_before)


def test_spect_alone_is_the_functional_image():
    lab = labels()
    s = Session(None, None, [io.Volume("ct", make("CT", lab), grid(), "CT", "HU"),
                             io.Volume("spect", make("SPECT", lab), grid(), "SPECT", "CNTS")])
    assert s.pet_name == "spect" and s.ct_name == "ct"


def test_registration_dialog_end_to_end(app, tmp_path, monkeypatch):
    from rsw.gui.main_window import MainWindow
    t = rigid(2.0, -1.0, 3.0, (6.0, -5.0, 4.0))
    lab = labels()
    ct_m, _ = moving_from("CT", t)
    pet = volume("PET", make("PET", lab))
    ct_fixed = volume("CT", make("CT", lab))
    w = MainWindow()
    w.session = Session(None, None, [pet, ct_fixed, io.Volume("CT_moving", ct_m.array, grid(), "CT", "HU")])
    w._after_open("syn")
    d = w.open_registration()
    assert d is not None
    d.fixed.setCurrentText("CT")
    d.moving.setCurrentText("CT_moving")
    d.init.setCurrentIndex(d.init.findData("identity"))
    res = d.run(sync=True)
    assert tre(res.transform, t) < 3.0
    assert "WARNING" not in d.report.toPlainText()
    assert d.pic.pixmap() is not None and not d.pic.pixmap().isNull()
    out = d.save_result(str(tmp_path / "t.tfm"))
    assert out.exists() and tre(rg.load_transform(out), res.transform) < 1e-6
    d.reject_result()
    assert "CT_moving" not in w.session.transforms
    res = d.run(sync=True)
    d.accept_result()
    assert "CT_moving" in w.session.transforms
    w.close()


def test_dialog_reject_second_run_and_close_restore_previous(app, tmp_path, monkeypatch):
    from rsw.gui.main_window import MainWindow
    t = rigid(1.0, 0.0, 2.0, (4.0, -3.0, 2.0))
    lab = labels()
    ct_m, _ = moving_from("CT", t)
    w = MainWindow()
    w.session = Session(None, None, [volume("PET", make("PET", lab)), volume("CT", make("CT", lab)),
                                     io.Volume("CT_moving", ct_m.array, grid(), "CT", "HU")])
    w._after_open("syn")
    d = w.open_registration()
    d.fixed.setCurrentText("CT")
    d.moving.setCurrentText("CT_moving")
    d.init.setCurrentIndex(d.init.findData("identity"))
    d.run(sync=True)
    d.run(sync=True)                                   # second run before deciding
    d.reject_result()
    assert "CT_moving" not in w.session.transforms     # back to the state before the dialog, not the first preview
    d.run(sync=True)
    d.reject()                                         # Esc / window close without Accept
    assert "CT_moving" not in w.session.transforms
    w.close()


def test_brush_size_control_and_outline(app, tmp_path):
    from rsw.gui.main_window import MainWindow
    lab = labels()
    w = MainWindow()
    w.session = Session(None, None, [volume("PET", make("PET", lab)), volume("CT", make("CT", lab))])
    w._after_open("syn")
    assert w.brush.minimum() == 0.5 and w.brush.decimals() == 1
    w.brush.setValue(2.0)
    assert w.brush_slider.value() == 4 and "below half a voxel" not in w.brush_note.text()   # 2 mm in a 3 mm voxel
    w.brush.setValue(0.5)
    assert "below half a voxel" in w.brush_note.text()
    w._brush_step(+1)
    assert w.brush.value() == 1.0
    w.set_tool("paint")
    v = w.viewer.views["axial"]
    v._place_ring(10, 10, True)
    assert v.ring.isVisible()
    rect = v.ring.rect()
    assert rect.width() >= v.pixel[1]                       # never smaller than the voxel actually painted
    w.set_tool("navigate")
    assert not v.ring.isVisible()
    w.close()


def _blob_layer():
    from rsw import seg
    z, y, x = np.ogrid[:30, :30, :30]
    arr = (((z - 15) ** 2 + (y - 15) ** 2 + (x - 15) ** 2) <= 36).astype(np.uint8)
    arr[15, 15, 24:28] = 1                      # thin spur
    arr[3, 3, 3] = 1                            # speck
    return seg.MaskLayer("m", arr)


def test_smooth_mask_removes_spur_and_undoes():
    from rsw import seg
    layer = _blob_layer()
    before = layer.array.copy()
    n = seg.smooth_mask(layer, 1.5, (2.0, 2.0, 2.0))
    assert n > 0 and layer.array[15, 15, 27] == 0 and layer.array[3, 3, 3] == 0      # spur and speck gone
    assert abs(int(layer.array.sum()) - int(before.sum())) < 0.25 * before.sum()
    assert layer.undo() and np.array_equal(layer.array, before)                      # one undo step restores all


def test_margin_mask_uses_millimetres_with_anisotropic_voxels():
    from rsw import seg
    arr = np.zeros((20, 20, 20), np.uint8)
    arr[10, 10, 10] = 1
    layer = seg.MaskLayer("m", arr)
    sp_xyz = (2.0, 2.0, 5.0)                    # z voxels are 5 mm thick
    seg.margin_mask(layer, 5.0, sp_xyz)
    assert layer.array[10, 10, 12] == 1 and layer.array[10, 10, 13] == 0     # 5 mm in x: 2 voxels (4 mm) in, 3 (6 mm) out
    assert layer.array[10, 12, 10] == 1                                      # y too
    assert layer.array[11, 10, 10] == 1 and layer.array[12, 10, 10] == 0     # z: 5 mm = 1 voxel in, 10 mm = out
    seg.margin_mask(layer, -5.0, sp_xyz)
    assert layer.array.sum() == 1 and layer.array[10, 10, 10] == 1


def test_paint_with_upper_limit(app):
    from rsw.gui.main_window import MainWindow
    lab = labels()
    w = MainWindow()
    w.session = Session(None, None, [volume("PET", make("PET", lab))])
    w._after_open("syn")
    w.session.add_empty_layer("m")
    arr = w.session.can["PET"]
    w.session.layer().array[:] = 0
    # a voxel with value inside [0.5, 2.0] is painted, a hot one (> 2.0) is not
    hot = np.unravel_index(np.argmax(arr), arr.shape)
    w.session.paint("axial", hot[0], hot[1], hot[2], 0.5, 1, False, 0.5, 2.0)
    assert w.session.layer().array[hot] == 0
    w.close()


def test_remove_layer_asks_and_keeps_file(app, tmp_path, monkeypatch):
    from PySide6 import QtWidgets
    from rsw.gui.main_window import MainWindow
    lab = labels()
    w = MainWindow()
    w.session = Session(None, None, [volume("PET", make("PET", lab))])
    w._after_open("syn")
    w.session.add_empty_layer("a")
    w.session.add_empty_layer("b")
    w.session.active = "b"
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: QtWidgets.QMessageBox.No))
    w.remove_active_layer()
    assert "b" in w.session.layers                       # answer No: nothing removed
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes))
    w.remove_active_layer()
    assert "b" not in w.session.layers and "a" in w.session.layers and w.session.active == "a"
    w.close()


def test_threshold_modes():
    from rsw import seg
    img = np.zeros((20, 20, 20), np.float32)
    img[5:8, 5:8, 5:8] = 10.0
    img[6, 6, 6] = 20.0                       # lesion A, peak 20
    img[13:16, 13:16, 13:16] = 6.0            # lesion B, peak 6
    layer = seg.MaskLayer("m", np.zeros(img.shape, np.uint8))
    seg.threshold_mask(layer, img, "add", low=5.0)                     # whole image
    assert layer.array.sum() == 27 + 27
    seg.threshold_mask(layer, img, "refine_rel", percent=41.0)         # per lesion: A keeps >= 8.2, B keeps >= 2.46
    assert layer.array[6, 6, 6] == 1 and layer.array[5, 5, 5] == 1 and layer.array[14, 14, 14] == 1
    assert layer.array[13:16, 13:16, 13:16].sum() == 27                # B untouched: its own maximum is the reference
    seg.threshold_mask(layer, img, "refine_rel", percent=75.0)         # A keeps >= 15 -> only the peak; B keeps >= 4.5 -> all
    assert layer.array[5:8, 5:8, 5:8].sum() == 1 and layer.array[13:16, 13:16, 13:16].sum() == 27
    assert layer.undo() and layer.array[5:8, 5:8, 5:8].sum() == 27
    seg.threshold_mask(layer, img, "refine_abs", low=7.0)              # B (6.0) disappears
    assert layer.array[13:16, 13:16, 13:16].sum() == 0
    with pytest.raises(ValueError):
        seg.threshold_mask(layer, img, "nope")


def test_ct_fixed_pet_moving_switches_reference_and_aligns(app, tmp_path, monkeypatch):
    from PySide6 import QtWidgets
    from rsw.gui.main_window import MainWindow
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes))
    t = rigid(2.0, -1.0, 3.0, (6.0, -5.0, 4.0))
    lab = labels()
    pet_m, _ = moving_from("PET", t)                       # PET moved away from the CT
    ct = volume("CT", make("CT", lab))
    w = MainWindow()
    w.session = Session(None, None, [io.Volume("PET", pet_m.array, grid(), "PET", "SUV"), ct])
    w._after_open("syn")
    assert w.session.ref.name == "PET"                      # PET is the viewer reference to begin with
    w.session.add_empty_layer("m")
    d = w.open_registration()
    assert d.fixed.currentText() == "CT" and d.moving.currentText() == "PET"      # sensible defaults
    d.init.setCurrentIndex(d.init.findData("identity"))
    res = d.run(sync=True)
    assert w.session.ref.name == "CT" and "m" in w.session.layers and not w.session.layers["m"].saved
    assert tre(res.transform, t) < 3.0
    ref_pet = make("PET", lab)
    inside = lab > 0
    shown = w.session.volumes["PET"].array
    assert np.abs(shown - ref_pet)[inside].mean() < 0.5 * np.abs(pet_m.array - ref_pet)[inside].mean()
    w.close()


def test_set_reference_keeps_mask_position():
    from rsw import seg
    lab = labels()
    s = Session(None, None, [volume("PET", make("PET", lab)), io.Volume("CT", make("CT", lab), io.Geometry(
        (96, 96, 88), (1.5, 1.5, 1.5), (-70.0, -72.0, -66.0), (1, 0, 0, 0, 1, 0, 0, 0, 1)), "CT", "HU")])
    s.add_empty_layer("m")
    s.layer().array[20:26, 20:26, 20:26] = 1
    before = np.argwhere(s.layer().array).mean(axis=0) * np.array([3.0, 3.0, 3.0])      # z,y,x in mm
    s.set_reference("CT")
    after = np.argwhere(s.layers["m"].array).mean(axis=0) * np.array([1.5, 1.5, 1.5])
    assert np.allclose(before, after, atol=2.5)


def test_registered_image_is_displayed_and_toggle_works(app, tmp_path, monkeypatch):
    from rsw.gui.main_window import MainWindow
    t = rigid(2.0, -1.0, 3.0, (6.0, -5.0, 4.0))
    lab = labels()
    sp_m, _ = moving_from("SPECT", t)
    w = MainWindow()
    w.session = Session(None, None, [volume("PET", make("PET", lab)), volume("CT", make("CT", lab)),
                                     io.Volume("SPECT_x", sp_m.array, grid(), "SPECT", "CNTS")])
    w._after_open("syn")
    assert w.display_names() == ("CT", "PET")                      # automatic choice shows the first PET, not the SPECT
    d = w.open_registration()
    d.fixed.setCurrentText("CT")
    d.moving.setCurrentText("SPECT_x")
    d.init.setCurrentIndex(d.init.findData("identity"))
    d.run(sync=True)
    assert w.display_names() == ("CT", "SPECT_x")                 # the moved image is now the visible overlay
    assert w.over_box.currentText() == "SPECT_x"
    d.toggle_before_after()
    assert "SPECT_x" not in w.session.transforms
    d.toggle_before_after()
    assert "SPECT_x" in w.session.transforms
    w.close()


def test_wheel_guard_ignores_unfocused_controls(app):
    from PySide6 import QtCore, QtGui, QtWidgets
    from rsw.gui.widgets import guard_wheel
    box = QtWidgets.QDoubleSpinBox()
    box.setRange(0, 100)
    box.setValue(5)
    holder = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(holder)
    lay.addWidget(box)
    assert guard_wheel(holder) == 1
    ev = QtGui.QWheelEvent(QtCore.QPointF(5, 5), QtCore.QPointF(5, 5), QtCore.QPoint(0, 0), QtCore.QPoint(0, 120),
                           QtCore.Qt.NoButton, QtCore.Qt.NoModifier, QtCore.Qt.NoScrollPhase, False)
    QtWidgets.QApplication.sendEvent(box, ev)
    assert box.value() == 5                                          # not focused: unchanged


def test_mask_operations():
    from rsw import seg
    arr = np.zeros((12, 20, 20), np.uint8)
    arr[2:10, 4:16, 4:16] = 1
    arr[5, 8:12, 8:12] = 0                         # hole in one slice
    arr[0, 0, 0] = 1                               # separate speck
    layer = seg.MaskLayer("m", arr.copy())
    assert seg.fill_holes(layer, True) > 0 and layer.array[5, 9, 9] == 1
    assert seg.keep_largest(layer) == 1 and layer.array[0, 0, 0] == 0
    other = np.zeros_like(arr)
    other[:, :, :10] = 1
    before = layer.array.copy()
    seg.combine(layer, other, "subtract")
    assert layer.array[:, :, :10].sum() == 0 and layer.array[:, :, 10:].sum() > 0
    layer.undo()
    assert np.array_equal(layer.array, before)
    seg.combine(layer, other, "intersect")
    assert layer.array[:, :, 10:].sum() == 0
    seg.clear_mask(layer)
    assert layer.array.sum() == 0


def test_interpolate_between_slices_and_polygon(app):
    from rsw import seg
    from rsw.gui.widgets import polygon_mask
    arr = np.zeros((11, 40, 40), np.uint8)
    yy, xx = np.ogrid[:40, :40]
    arr[0] = (((yy - 20) ** 2 + (xx - 20) ** 2) <= 8 ** 2)
    arr[10] = (((yy - 20) ** 2 + (xx - 20) ** 2) <= 4 ** 2)
    layer = seg.MaskLayer("m", arr)
    seg.interpolate_slices(layer, (1.0, 1.0, 1.0))
    areas = [int(layer.array[k].sum()) for k in range(11)]
    assert all(a > 0 for a in areas) and areas == sorted(areas, reverse=True)       # shrinks smoothly from 8 to 4 voxels radius
    sq = polygon_mask((20, 20), [(5.5, 5.5), (5.5, 14.5), (14.5, 14.5), (14.5, 5.5)])
    assert sq.sum() == 100 and sq[5, 5] and sq[14, 14] and not sq[4, 4] and not sq[15, 15]


def test_polygon_tool_adds_and_erases(app):
    from rsw.gui.main_window import MainWindow
    lab = labels()
    w = MainWindow()
    w.session = Session(None, None, [volume("PET", make("PET", lab))])
    w._after_open("syn")
    w.session.add_empty_layer("m")
    w.set_tool("poly")
    for r, c in [(10, 10), (10, 20), (20, 20)]:
        w.on_click("axial", r, c)
    w.on_click("axial", 20, 10)                              # a double-click gives a single click first ...
    w.on_click("axial", 20, 10, True)                        # ... then the double-click itself: closes, no extra corner
    idx = w.viewer.views["axial"].index
    n = int(w.session.layer().array[idx].sum())
    assert n == 121 and w.poly is None                       # 11 x 11 voxels, corners included
    w.poly_erase.setChecked(True)
    for r, c in [(12, 12), (12, 16), (16, 16), (16, 12)]:
        w.on_click("axial", r, c)
    w.poly_finish()
    assert w.session.layer().array[idx].sum() == n - 25      # 5 x 5 hole
    assert w.session.layer().undo()
    for l in w.session.layers.values():
        l.saved = True
    w.close()



def _chain_session():
    """A (frame), B and C = the same anatomy seen through known transforms tB, tC (fixed A -> moving)."""
    tB, tC = rigid(2.0, -1.0, 3.0, (8.0, -6.0, 5.0)), rigid(-1.0, 2.0, -2.0, (-5.0, 7.0, 4.0))
    b, _ = moving_from("CT", tB)
    c, _ = moving_from("PET", tC)
    a = volume("CT", make("CT", labels()), "A")
    b.name, c.name = "B", "C"
    s = Session(None, None, [a, b, c])
    s.set_reference("A")                                   # a PET would otherwise be the default reference
    return s, tB, tC


def _off(t, truth):
    from tests.test_rsw_registration import probe_points
    return float(np.mean([np.linalg.norm(np.array(t.TransformPoint(tuple(p))) - np.array(truth.TransformPoint(tuple(p))))
                          for p in probe_points()]))


def test_change_reference_keeps_earlier_registrations():
    """Registering A<-B and A<-C, then using the registered B as the fixed image, must not lose or double them."""
    s, tB, tC = _chain_session()
    s.set_transform("B", tB)
    s.set_transform("C", tC)
    s.set_reference("B")                                   # B has a transform: the frame becomes B's own space
    binv = tB.GetInverse()
    assert "B" not in s.transforms
    assert _off(s.transforms["A"], binv) < 1e-4                          # old reference = T_B^-1
    comp = sitk.CompositeTransform(3)
    comp.AddTransform(tC)
    comp.AddTransform(binv)
    assert _off(s.transforms["C"], comp) < 1e-4                          # C(T_B^-1(p))
    a_v, b_v = s.can["A"], s.can["B"]                                    # same CT anatomy, must overlap again
    assert np.corrcoef(a_v.ravel(), b_v.ravel())[0, 1] > 0.9
    s.set_reference("A")                                                 # and back: the original state
    assert _off(s.transforms["B"], tB) < 1e-4 and _off(s.transforms["C"], tC) < 1e-4
    assert "A" not in s.transforms


def test_change_reference_moves_masks_with_the_frame(tmp_path):
    s, tB, _ = _chain_session()
    s.set_transform("B", tB)
    s.add_empty_layer("m")
    lab = labels() > 0
    s.layers["m"].apply(np.asarray(s.orient.canonical(lab), bool), tuple(slice(0, n) for n in s.shape), 1)
    n0 = int(s.layers["m"].array.sum())
    s.set_reference("B")
    assert abs(int(s.layers["m"].array.sum()) - n0) < 0.1 * n0          # same volume, moved (not lost)


def test_dialog_chain_reject_restores_reference_and_transforms(app, monkeypatch):
    from rsw.gui.main_window import MainWindow
    s, tB, tC = _chain_session()
    w = MainWindow()
    w.session = s
    w._after_open("syn")
    s.set_transform("B", tB)
    s.set_transform("C", tC)
    d = w.open_registration()
    d.fixed.setCurrentText("B")
    d.moving.setCurrentText("C")
    d.run(sync=True)
    assert s.ref.name == "B"
    d.reject_result()
    assert s.ref.name == "A"
    assert _off(s.transforms["B"], tB) < 1e-4 and _off(s.transforms["C"], tC) < 1e-4
    w.close()


def test_hybrid_estimate_moves_the_studys_other_image_and_before_shows_previous(app, monkeypatch):
    from rsw.gui.main_window import MainWindow
    s, tB, tC = _chain_session()
    w = MainWindow()
    w.session = s
    w._after_open("syn")
    d = w.open_registration()
    d.fixed.setCurrentText("A")
    d.moving.setCurrentText("C")
    d.moving_est.setCurrentText("B")                       # estimate on B (same study), apply to C
    d.init.setCurrentIndex(d.init.findData("identity"))
    d.run(sync=True)
    assert "B" in s.transforms and "C" in s.transforms     # both images of the study moved together
    d.reject_result()
    assert not s.transforms
    w.close()


def test_landmark_pairs_fit_and_validation(app):
    from rsw.gui.main_window import MainWindow
    from tests.test_rsw_registration import probe_points
    s, tB, _ = _chain_session()
    w = MainWindow()
    w.session = s
    w._after_open("syn")
    d = w.open_registration()
    d.fixed.setCurrentText("A")
    d.moving.setCurrentText("B")
    # cursor -> pair: the same voxel gives the same physical point on both images while B has no transform
    w.set_cursor((10, 12, 14))
    d.lm_fixed()
    d.lm_moving()
    assert len(d.pairs) == 1 and np.allclose(d.pairs[0]["fixed"], d.pairs[0]["moving"])
    d.pairs = []
    pts = probe_points()
    for i in (0, 50, 100, 150, 200, 250):                                  # fitting pairs from the known transform
        d.pairs.append({"fixed": tuple(pts[i]), "moving": tuple(tB.TransformPoint(tuple(pts[i]))), "validation": False})
    for i in (300, 350):                                                   # validation pairs
        d.pairs.append({"fixed": tuple(pts[i]), "moving": tuple(tB.TransformPoint(tuple(pts[i]))), "validation": True})
    d.pairs_ref = "A"
    d._lm_summary()
    assert "6 fitting" in d.lm_text.text() and "2 validation" in d.lm_text.text()
    res = d.run_landmarks()
    assert tre(res.transform, tB) < 1e-3
    assert "Validation landmarks" in d.report.toPlainText() and "TRE mean 0.0" in d.report.toPlainText()
    d.reject_result()
    d.init.setCurrentIndex([i for i in range(d.init.count()) if d.init.itemData(i) == "landmarks"][0])
    res = d.run(sync=True)                                                 # landmarks as the start of the image optimisation
    assert tre(res.transform, tB) < 3.0
    w.close()


def test_landmark_input_checks(app):
    from rsw.gui.main_window import MainWindow
    s, _, _ = _chain_session()
    w = MainWindow()
    w.session = s
    w._after_open("syn")
    d = w.open_registration()
    d.fixed.setCurrentText("A")
    d.moving.setCurrentText("B")
    d.pairs_ref = "A"
    assert d.run_landmarks() is None and "three" in d.report.toPlainText()
    line = [(float(i) * 30.0, 0.0, 0.0) for i in range(4)]                 # all on one line
    d.pairs = [{"fixed": p, "moving": p, "validation": False} for p in line]
    assert d.run_landmarks() is None and "one line" in d.report.toPlainText()
    w.close()
