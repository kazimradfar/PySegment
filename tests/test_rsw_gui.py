"""The desktop window, driven like a user (offscreen Qt, synthetic patients only).
Screenshots are written to $RSW_SHOTS if set (used by CI to check the layout)."""
import os
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def synthetic_case(folder: Path, seed=0):
    rng = np.random.default_rng(seed)
    z, y, x = 60, 64, 64
    zz, yy, xx = np.ogrid[:z, :y, :x]
    body = ((yy - 32) / 26) ** 2 + ((xx - 32) / 30) ** 2 <= 1
    ct = np.where(body, 40.0, -1000.0).astype(np.float32) + rng.normal(0, 15, (z, y, x)).astype(np.float32)
    pet = np.where(body, 0.8, 0.05).astype(np.float32) * rng.gamma(20, 1 / 20, (z, y, x)).astype(np.float32)
    liver = ((zz - 35) / 9) ** 2 + ((yy - 30) / 10) ** 2 + ((xx - 20) / 9) ** 2 <= 1
    pet[liver] = 5.0
    auto = np.zeros((z, y, x), np.uint8)
    for (cz, cy, cx, r, suv) in [(20, 40, 40, 4, 14.0), (45, 25, 44, 3, 9.0), (12, 30, 22, 2.5, 7.0)]:
        m = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        pet[m] = suv
        auto[m] = 1
    auto[20, 40, 43] = 0                                           # automatic mask misses a voxel
    folder.mkdir(parents=True, exist_ok=True)
    for arr, name in ((pet, "PET_SUV.nii.gz"), (ct, "CT.nii.gz"), (auto, "lesions_auto.nii.gz")):
        im = sitk.GetImageFromArray(arr)
        im.SetSpacing((4.0, 4.0, 4.0))
        im.SetOrigin((-128.0, -128.0, -900.0))
        sitk.WriteImage(im, str(folder / name))


@pytest.fixture
def app():
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _shot(win, name):
    out = os.environ.get("RSW_SHOTS")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        win.grab().save(str(Path(out) / name))


def _pump(app, n=5):
    for _ in range(n):
        app.processEvents()


def test_window_reader_workflow(app, tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    for i in range(3):
        synthetic_case(tmp_path / "study" / f"SYN{i + 1:03d}", seed=i)
    from rsw.gui.main_window import MainWindow
    w = MainWindow()
    w.resize(1366, 768)
    w.show()
    w.open_folder(tmp_path / "study")
    assert w.case_tree.topLevelItemCount() == 3 and w.project.layout == "generic"
    w.open_case("SYN001", ask=False)
    _pump(app)
    s = w.session
    assert s.pet_name == "PET_SUV" and s.ct_name == "CT" and "lesions_auto" in s.layers
    _shot(w, "rsw_1_case.png")

    # Reader 1: start from the automatic mask, paint a stroke, grow a lesion, save
    w.reader_box.setCurrentText("reader1")
    w.start_reader()
    assert s.active == "lesions_reader1" and w.tool == "paint"
    before = s.layer().voxels
    a0 = s.shape[0] - 1 - 20                                         # canonical axial index of z = 20
    w.viewer.views["axial"].index = a0
    w.on_paint("axial", 10, 10, True, False)
    w.on_paint("axial", 10, 14, False, False)
    w.on_paint("axial", 10, 18, False, True)
    painted = s.layer().voxels
    assert painted > before
    assert s.layer().undo() and s.layer().voxels == before          # the whole stroke is one undo step
    assert s.layer().redo() and s.layer().voxels == painted
    w.set_tool("grow")
    w.on_click("axial", 40, 41)
    assert s.layer().voxels > painted                              # the missed voxel of lesion 1 is added
    w.viewer.refresh()
    _pump(app)
    _shot(w, "rsw_2_reader1_editing.png")
    w.save_active()
    saved = tmp_path / "study" / "SYN001" / "masks" / "lesions_reader1.nii.gz"
    assert saved.exists() and s.layer().saved
    img = sitk.ReadImage(str(saved))
    pet = sitk.ReadImage(str(tmp_path / "study" / "SYN001" / "PET_SUV.nii.gz"))
    assert img.GetSize() == pet.GetSize() and img.GetOrigin() == pet.GetOrigin()
    assert sitk.GetArrayFromImage(img)[20, 40, 43] == 1             # grown voxel on the original grid

    # Reader 2 (blinded): Reader 1's mask is not shown, start from the automatic mask
    w.open_case("SYN001", ask=False)
    w.reader_box.setCurrentText("reader2")
    w.blind.setChecked(True)
    w.start_reader()
    assert "lesions_reader1" not in w.session.layers
    assert w.session.layer().voxels == w.session.layers["lesions_auto"].voxels
    w.save_active()

    # agreement of the current case and lesion list
    w.ag_a.setCurrentText("lesions_auto")
    w.ag_b.setCurrentText("lesions_reader2")
    w.compare_layers()
    assert "Dice 1.000" in w.ag_label.text()
    w.list_lesions()
    assert len(w.lesion_model.df) == 3

    # radiomics of the current case (background job)
    w.r_res.setValue(0)
    job = w.run_radiomics()
    job.wait(120000)
    _pump(app, 20)
    assert len(w.feat_model.df) == 4 and "original_firstorder_Maximum" in w.feat_model.df.columns
    st = w.radiomics_settings()
    assert st.resampling == "pyradiomics" and st.outlier_sigma is None and not w.r_outlier.isEnabled()
    # IBSI resampling option
    w.r_method.setCurrentIndex(1)
    w.r_interp.setCurrentIndex(1)
    w.r_res.setValue(4.0)
    w.r_outlier.setChecked(True)
    st = w.radiomics_settings()
    assert (st.resampling, st.interpolator, st.outlier_sigma) == ("ibsi", "sitkLinear", 3.0)
    job = w.run_radiomics()
    job.wait(120000)
    _pump(app, 20)
    assert len(w.feat_model.df) == 4
    w.r_method.setCurrentIndex(0)
    w.r_interp.setCurrentIndex(0)
    w.tabs.setCurrentIndex(2)
    _pump(app)
    _shot(w, "rsw_3_radiomics.png")
    w.viewer.set_layout("Axial + MIP")
    _pump(app)
    _shot(w, "rsw_4_axial_mip.png")


def test_labels_and_names_on_the_image(app, tmp_path, monkeypatch):
    """Labels tab: text with an arrow at the clicked point, moved by dragging text and tip, edited, deleted;
    mask/ROI names on the image; both exported by 'Save view as image'."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    synthetic_case(tmp_path / "study" / "SYN001")
    from PySide6 import QtCore, QtWidgets
    from rsw.gui.main_window import MainWindow
    w = MainWindow()
    w.resize(1366, 768)
    w.show()
    w.open_folder(tmp_path / "study")
    w.open_case("SYN001", ask=False)
    _pump(app)
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *a, **k: ("Lesion 1", True))
    v = w.viewer.views["axial"]
    w.set_tool("label")
    w.on_click("axial", 30, 30)
    a = w.session.annotations[-1]
    assert a["text"] == "Lesion 1" and a["index"] == v.index and w.label_list.count() == 1
    assert any(t.textItem.toPlainText() == "Lesion 1" for _i, t, _xy in v._hits) and len(v._labels) == 3
    tip = v.vb.mapViewToScene(QtCore.QPointF(a["x"], a["y"]))
    assert v.label_hit(tip) == (0, "tip")
    assert v.label_hit(v._hits[0][1].sceneBoundingRect().center()) == (0, "text")
    w.move_label(v, 0, "text", a["x"] + 20, a["y"] - 5)             # text to the right: arrow follows
    assert (a["dx"], a["dy"]) == (20, -5)
    tx, ty = a["x"] + a["dx"], a["y"] + a["dy"]
    w.move_label(v, 0, "tip", a["x"] - 3, a["y"] + 4)                # tip moves, text stays
    assert abs(a["x"] + a["dx"] - tx) < 1e-9 and abs(a["y"] + a["dy"] - ty) < 1e-9
    w.label_size.setValue(20)
    assert w.session.annotations[0]["size"] == 20
    w.names_check.setChecked(True)
    _pump(app)
    out = tmp_path / "view.png"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), "PNG (*.png)"))
    w.save_screenshot()
    assert out.exists() and out.stat().st_size > 1000
    from PySide6 import QtGui
    full = tmp_path / "window.png"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (str(full), "PNG (*.png)"))
    w.save_window_image()
    im = QtGui.QImage(str(full))
    assert (im.width(), im.height()) == (3 * w.width(), 3 * w.height())          # whole window, 3x resolution
    assert abs(im.dotsPerMeterX() * 0.0254 - 300) < 1
    dlg = QtWidgets.QDialog(w)                                       # F12 in a dialog saves that dialog
    dlg.resize(300, 200)
    dlg.show()
    part = tmp_path / "dialog.png"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (str(part), "PNG (*.png)"))
    from PySide6.QtTest import QTest
    QTest.keyClick(dlg, QtCore.Qt.Key_F12)
    im = QtGui.QImage(str(part))
    assert (im.width(), im.height()) == (900, 600)
    dlg.close()
    _shot(w, "rsw_labels.png")
    w.delete_label()
    assert not w.session.annotations and w.label_list.count() == 0


def test_open_dicom_folder_with_two_series_in_one_folder(app, tmp_path, monkeypatch):
    """File > Open DICOM folder: a PET and a CT series exported into the same folder are listed separately and
    opened together; PET is converted to SUV."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    from datetime import datetime
    from pydicom.uid import generate_uid
    from tools import make_realistic_demo as m
    when = datetime(2026, 3, 3, 9, 0)
    study, frame = generate_uid(), generate_uid()
    pet = np.full((6, 16, 16), 5000.0, np.float32)
    ct = np.full((6, 16, 16), 40.0, np.float32)
    m.write_dicom_series(pet, (4.0, 4.0, 4.0), tmp_path / "a", "PT", study, frame, "PET", m.pet_extra(80.0, 150.0, 60.0, when), when, "X")
    m.write_dicom_series(ct, (4.0, 4.0, 4.0), tmp_path / "b", "CT", study, frame, "CT", {}, when, "X")
    mixed = tmp_path / "export"
    mixed.mkdir()
    for sub in ("a", "b"):
        for f in (tmp_path / sub).iterdir():
            f.rename(mixed / f"{sub}_{f.name}")
    from rsw.suv import list_series
    series = list_series(mixed)
    assert [e["modality"] for e in series] == ["PT", "CT"] and [e["n"] for e in series] == [6, 6]
    from rsw.gui.main_window import MainWindow
    w = MainWindow()
    w.open_dicom(mixed, choose=lambda s: s)
    assert set(w.session.volumes) == {"PET", "CT"} and w.session.pet_name == "PET"
    assert abs(float(w.session.volumes["CT"].array.mean()) - 40.0) < 1.0
    assert w.session.volumes["PET"].unit == "SUV"


def _planar_wb(path):
    """Synthetic whole-body NM file (GE-like): 2 frames, anterior and posterior, hot spot in the upper left of the
    anterior frame (patient's right shoulder region)."""
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    a = np.ones((2, 64, 16), np.uint16)
    a[0, 5:8, 2:4] = 50                               # anterior: row 5 (head end), column 2 (viewer's left)
    a[1, 5:8, 12:14] = 50                             # same spot seen from behind: mirrored
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.20"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = pydicom.FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Modality, ds.SOPClassUID, ds.SeriesInstanceUID = "NM", meta.MediaStorageSOPClassUID, generate_uid()
    ds.ImageType = ["ORIGINAL", "PRIMARY", "WHOLE BODY", "EMISSION"]
    ds.SeriesDescription, ds.NumberOfFrames, ds.Rows, ds.Columns = "WB", 2, 64, 16
    ds.PixelSpacing, ds.BitsAllocated, ds.BitsStored, ds.HighBit = [2.2, 2.2], 16, 16, 15
    ds.SamplesPerPixel, ds.PhotometricInterpretation, ds.PixelRepresentation = 1, "MONOCHROME2", 0
    ds.ActualFrameDuration = 160000
    dets = []
    for view in ("Anterior projection", "Posterior projection"):
        d, c = Dataset(), Dataset()
        c.CodeMeaning = view
        d.ViewCodeSequence = [c]
        dets.append(d)
    ds.DetectorInformationSequence = dets
    ds.PixelData = a.tobytes()
    ds.save_as(str(path), enforce_file_format=True)


def test_planar_whole_body_ant_post(app, tmp_path, monkeypatch):
    """A whole-body planar NM file opens as two images, NM ANT and NM POST, in one coronal view, head up; the
    posterior view is mirrored, so the same spot lies at the same place in both."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    (tmp_path / "wb").mkdir()
    _planar_wb(tmp_path / "wb" / "ANT001_DS.dcm")
    from rsw.gui.main_window import MainWindow
    w = MainWindow()
    w.resize(1366, 768)
    w.show()
    w.open_dicom(tmp_path / "wb")
    _pump(app)
    s = w.session
    assert set(s.volumes) == {"NM ANT", "NM POST"} and s.shape[1] == 1
    ant, post = s.can["NM ANT"], s.can["NM POST"]
    za, _, xa = np.unravel_index(np.argmax(ant), ant.shape)
    zp, _, xp = np.unravel_index(np.argmax(post), post.shape)
    assert (za, xa) == (zp, xp)                                       # same anatomical place after mirroring
    from rsw.orient import view_slice
    v = w.viewer.views["coronal"]
    sl = view_slice(ant, "coronal", v.index)
    r, c = np.unravel_index(np.argmax(sl), sl.shape)
    assert r < sl.shape[0] // 4 and c < sl.shape[1] // 2              # hot spot near the top and on the viewer's left
    assert w.viewer.views["coronal"].isVisible() and not w.viewer.views["axial"].isVisible()
    monkeypatch.setattr(QtWidgets_dialog(), "getText", lambda *a, **k: ("Shoulder", True))
    w.set_tool("label")
    w.on_click("coronal", int(r), int(c))
    assert w.session.annotations[-1]["view"] == "coronal"
    _shot(w, "rsw_planar.png")


def QtWidgets_dialog():
    from PySide6 import QtWidgets
    return QtWidgets.QInputDialog


def test_registration_dialog_can_be_saved_as_image(app, tmp_path, monkeypatch):
    """The registration dialog had an attribute 'metric' that hid QPaintDevice.metric(), so drawing it into an image
    (F12) failed with 'QComboBox object is not callable'."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    synthetic_case(tmp_path / "study" / "SYN001")
    from PySide6 import QtGui
    from rsw.gui.main_window import MainWindow
    w = MainWindow()
    w.open_folder(tmp_path / "study")
    w.open_case("SYN001", ask=False)
    dlg = w.open_registration()
    out = tmp_path / "reg.png"
    w.render_png(dlg, out)
    assert QtGui.QImage(str(out)).width() == 3 * dlg.width()
    dlg.close()
