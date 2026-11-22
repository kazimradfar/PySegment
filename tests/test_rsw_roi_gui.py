"""ROI workflow in the window: draw (freehand, ellipse, polygon, circle, sphere), click to open the details, save and
load, ROI table with calibration, PERCIST reference, profile, Gaussian filter. Synthetic images only."""
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("pyqtgraph")
from PySide6 import QtWidgets  # noqa: E402

from rsw import io  # noqa: E402
from rsw.session import Session  # noqa: E402


@pytest.fixture
def win(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from rsw.gui import main_window as mw
    z, y, x = np.ogrid[:30, :30, :30]
    pet = np.full((30, 30, 30), 1.0, np.float32)
    pet[((z - 15) ** 2 + (y - 15) ** 2 + (x - 15) ** 2) * 16 <= 32.0 ** 2] = 8.0          # hot ball, radius 32 mm
    geo = io.Geometry((30, 30, 30), (4.0, 4.0, 4.0), (0.0, 0.0, 0.0), (1, 0, 0, 0, 1, 0, 0, 0, 1))
    w = mw.MainWindow()
    w.session = Session(None, None, [io.Volume("PET", pet, geo, "PET", "SUV", info={"injected_MBq": 250.0, "dose_at_scan_MBq": 200.0, "weight_kg": 80.0}),
                                     io.Volume("CT", pet * 10.0, geo, "CT", "HU")])
    w._after_open("syn")
    yield w, app
    for r in w.session.rois.values():
        r.saved = True
    w.close()


def centre(w):
    return tuple(int(v) for v in np.argwhere(w.session.can["PET"] == 8.0).mean(axis=0).round())


def shape_idx(w, key):
    return [i for i in range(w.roi_shape.count()) if w.roi_shape.itemData(i) == key][0]


def test_draw_all_shapes_and_click_opens_details(win):
    w, app = win
    cz, cy, cx = centre(w)
    w.set_cursor((cz, cy, cx))
    w.set_tool("roi")
    w.roi_shape.setCurrentIndex(shape_idx(w, "freehand"))
    pts = [(cy - 3, cx - 3), (cy - 3, cx + 3), (cy + 3, cx + 3), (cy + 3, cx - 3)]
    for i, (r, c) in enumerate(pts):
        w.on_roi_drag("axial", r, c, i == 0, i == len(pts) - 1)
    w.roi_shape.setCurrentIndex(shape_idx(w, "ellipse"))
    w.on_roi_drag("axial", cy - 2, cx - 4, True, False)
    w.on_roi_drag("axial", cy + 2, cx + 4, False, True)
    w.roi_shape.setCurrentIndex(shape_idx(w, "polygon"))
    for r, c in ((cy - 2, cx - 2), (cy - 2, cx + 2), (cy + 2, cx)):
        w.on_click("axial", r, c)
    w.on_click("axial", cy + 2, cx, double=True)
    w.roi_shape.setCurrentIndex(shape_idx(w, "circle"))
    w.on_click("axial", cy, cx)
    w.roi_shape.setCurrentIndex(shape_idx(w, "sphere"))
    w.roi_radius.setValue(8.0)
    w.on_click("axial", cy, cx)
    s = w.session
    assert [r.shape for r in s.rois.values()] == ["freehand", "ellipse", "polygon", "circle", "sphere"]
    assert w.roi_list.count() == 5
    for n in s.rois:
        info, rows = s.roi_details(n)
        assert next(r for r in rows if r["image"] == "PET")["mean"] == pytest.approx(8.0)       # all inside the hot ball
    assert s.roi_details("ROI 1")[0]["voxels"] == 49                                           # 7 x 7 incl. the outline
    assert s.roi_details("ROI 5")[0]["plane"] == "3D"
    w.set_tool("navigate")
    w.on_click("axial", cy, cx)                                                                 # click inside -> details
    d = w.roi_dlg
    assert d is not None and d.isVisible() and d.name == "ROI 5"                                # the newest ROI wins
    assert set(d.df["image"]) == {"PET", "CT"} and "concentration_kBq_per_ml" in d.df
    assert "Volume" in d.info.text()
    shown = d.model.df                                                                          # quantities as rows
    assert list(shown.columns) == ["Quantity", "PET", "CT"] and "Mean" in list(shown["Quantity"])
    scr = app.primaryScreen().availableGeometry()
    assert d.width() <= scr.width() and d.height() <= scr.height()
    w.on_click("axial", 0, 0)                                                                   # outside: no change
    assert d.name == "ROI 5"


def test_save_load_table_percist_profile_filter(win, tmp_path, monkeypatch):
    w, _ = win
    cz, cy, cx = centre(w)
    w.set_cursor((cz, cy, cx))
    w.set_tool("roi")
    w.roi_shape.setCurrentIndex(shape_idx(w, "sphere"))
    w.roi_radius.setValue(8.0)
    w.on_click("axial", cy, cx)
    w.roi_shape.setCurrentIndex(shape_idx(w, "circle"))
    w.on_click("axial", 2, 2)                                                                    # background circle
    out = w.save_all_rois(str(tmp_path))
    assert len(out) == 2 and all(p.exists() for p in out) and len(list(tmp_path.glob("*.roi.json"))) == 2
    s = w.session
    for n in list(s.rois):
        s.remove_roi(n)
    w.load_roi_dialog([str(p) for p in out])
    assert sorted(s.rois) == ["ROI 1", "ROI 2"] and s.rois["ROI 2"].shape == "circle" and all(r.saved for r in s.rois.values())
    s.roi_selected = "ROI 1"
    w.roi_set_reference()
    w.roi_to_threshold()
    assert w.thr_val.value() == pytest.approx(12.0)                                             # 1.5 x 8 + 2 x 0
    w.show_measure_table()
    assert len(w.measure_dlg.frame()) == 2
    f = tmp_path / "t.csv"
    from rsw.gui import main_window as mw
    monkeypatch.setattr(mw.QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(f), "")))
    w.measure_dlg.export()
    df = pd.read_csv(f)
    assert list(df["roi"]) == ["ROI 1", "ROI 2"] and df["mean_intensity"][0] == pytest.approx(8.0)
    w.set_tool("profile")
    w.profile_click((cz, cy, 1))
    w.profile_click((cz, cy, 28))
    assert 50.0 < w.last_roi["fwhm_mm"] < 75.0
    w._profile_dlg.close()
    assert w.add_filtered_image(3.0).startswith("PET_g")


def test_single_click_paints_and_tool_buttons_stay_in_step(win):
    w, _ = win
    s = w.session
    s.add_empty_layer("m")
    s.active = "m"
    w._refresh_layers()
    cz, cy, cx = centre(w)
    w.set_cursor((cz, cy, cx))
    w.brush.setValue(4.0)
    w.set_tool("paint")
    w.on_click("axial", cy, cx)                                   # one click, no drag
    n1 = int(s.layers["m"].array.sum())
    assert n1 > 0
    w.on_click("axial", cy, cx + 6)
    assert int(s.layers["m"].array.sum()) > n1                    # every click adds
    w.set_tool("erase")
    w.on_click("axial", cy, cx)
    assert int(s.layers["m"].array.sum()) < 2 * n1
    w.set_tool("roi")
    assert all(b.isChecked() for b in w.tool_buttons["roi"])
    assert not any(b.isChecked() for b in w.tool_buttons["paint"] + w.tool_buttons["navigate"])
    assert len(w.tool_buttons["navigate"]) == 3 and "roi" not in [k for k, b in w.tool_buttons.items() if len(b) > 1]
    s.layers["m"].saved = True


def test_edit_roi_add_remove_undo_and_fusion_preview(win):
    w, _ = win
    s = w.session
    cz, cy, cx = centre(w)
    w.set_cursor((cz, cy, cx))
    w.set_tool("roi")
    w.roi_shape.setCurrentIndex(shape_idx(w, "circle"))
    w.roi_radius.setValue(8.0)
    w.on_click("axial", cy, cx)                                     # ROI 1
    n0 = s.roi_details("ROI 1")[0]["voxels"]
    w.roi_mode.setCurrentIndex(1)                                   # add to the selected ROI
    w.on_click("axial", cy, cx + 5)
    n1 = s.roi_details("ROI 1")[0]["voxels"]
    assert len(s.rois) == 1 and n1 > n0
    w.roi_mode.setCurrentIndex(2)                                   # remove
    w.roi_shape.setCurrentIndex(shape_idx(w, "ellipse"))
    w.on_roi_drag("axial", cy - 1, cx - 1, True, False)
    w.on_roi_drag("axial", cy + 1, cx + 1, False, True)
    n2 = s.roi_details("ROI 1")[0]["voxels"]
    assert n2 < n1 and len(s.rois) == 1
    w.undo_roi_edit()
    assert s.roi_details("ROI 1")[0]["voxels"] == n1
    w.undo_roi_edit()
    assert s.roi_details("ROI 1")[0]["voxels"] == n0 and not s.rois["ROI 1"].saved
    from rsw.gui.registration_dialog import fusion
    img = fusion(np.random.rand(20, 30), np.random.rand(20, 30))
    assert img.shape == (20, 30, 3) and img.dtype == np.uint8
