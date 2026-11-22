"""Main window: project and case list, viewer, segmentation tools, radiomics, agreement."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

from rsw import APP_LONG_NAME, APP_NAME, analysis, metrics
from rsw import io, seg
from rsw.gui.viewer import Viewer, _lut
from rsw.gui.widgets import Job, PandasModel, about_text, guard_wheel, polygon_mask, table_view
from rsw.orient import VIEW_AXIS
from rsw.project import Project, open_project
from rsw.radiomics import TEXTURE, Settings, extract, save_table
from rsw.seg import RegionTooLarge
from rsw.session import Session

TOOLS = [("navigate", "Navigate", "N"), ("paint", "Paint", "P"), ("erase", "Erase", "E"), ("poly", "Polygon", "L"),
         ("grow", "Grow", "G"), ("roi", "ROI", "R"), ("profile", "Profile", "O"), ("label", "Label", "T")]


def _spin(lo, hi, val, step=1.0, dec=1, suffix=""):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    if suffix:
        s.setSuffix(suffix)
    return s


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} – {APP_LONG_NAME}")
        self.resize(1366, 768)
        self.project: Project | None = None
        self.session: Session | None = None
        self.cursor = (0, 0, 0)
        self.tool = "navigate"
        self.show_cursor = True
        self.overlay_opacity = 0.6
        self.ref_roi, self.profile_pts, self.last_profile, self.last_roi = None, None, None, None
        self.measure_dlg = self.roi_dlg = self.dosimetry_dlg = None
        self.poly = None                       # polygon tool in progress: {view, index, pts[(row, col)]}
        self.mask_fill = 100
        self.win = {"PET": [0.0, 10.0], "CT": [-160.0, 240.0], "MR": None, "SPECT": None, "other": None}
        self._stroke = None
        self.jobs = []
        self.settings = QtCore.QSettings(APP_NAME, APP_NAME)

        self.viewer = Viewer(self)
        self.setCentralWidget(self.viewer)
        self._build_docks()
        self._build_menus()
        QtWidgets.QApplication.instance().installEventFilter(self)     # F12: save whichever window is in front
        self.lic = QtWidgets.QLabel("Open source · MIT licence · research use, not a medical device")
        self.statusBar().addPermanentWidget(self.lic)
        self.status("Open a folder (File → Open project folder) or images (File → Open images).")

    # ================================================================== layout
    def _build_docks(self):
        self.tabs = QtWidgets.QTabWidget()
        # every tab scrolls, so nothing is cut off on small screens or with a large Windows text size
        for build, title in ((self._project_tab, "Project"), (self._segment_tab, "Segment"), (self._roi_tab, "ROI"),
                             (self._labels_tab, "Labels"), (self._radiomics_tab, "Radiomics"), (self._agreement_tab, "Agreement"),
                             (self._display_tab, "Display")):
            self.tabs.addTab(self._scroll(build()), title)
        guard_wheel(self.tabs)      # the wheel scrolls the panel; values change only after a click into the control
        dock = QtWidgets.QDockWidget("Workspace")
        dock.setObjectName("workspace")
        dock.setWidget(self.tabs)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        dock.setMinimumWidth(330)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        self.logbox = QtWidgets.QPlainTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setMaximumBlockCount(2000)
        ld = QtWidgets.QDockWidget("Log")
        ld.setObjectName("log")
        ld.setWidget(self.logbox)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, ld)
        ld.setMaximumHeight(130)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)

    def _build_menus(self):
        m = self.menuBar()
        f = m.addMenu("&File")
        self._act(f, "Open project folder…", self.open_folder_dialog, "Ctrl+O")
        self._act(f, "Open images…", self.open_images_dialog)
        self._act(f, "Open DICOM folder…", self.open_dicom_dialog)
        f.addSeparator()
        self._act(f, "Save active mask", self.save_active, "Ctrl+S")
        self._act(f, "Save active mask as…", self.save_active_as)
        self._act(f, "Load mask file…", self.load_mask_dialog)
        f.addSeparator()
        self._act(f, "Export images as NIfTI…", self.export_images)
        self._act(f, "DICOM header and SUV check…", self.dicom_header_dialog)
        self._act(f, "Anonymise DICOM folder…", self.dicom_anonymise_dialog)
        self._act(f, "Save view as image…", self.save_screenshot)
        self._act(f, "Save window as image… (F12: any open window)", lambda: self.save_window_image())
        f.addSeparator()
        self._act(f, "Exit", self.close)
        e = m.addMenu("&Edit")
        self._act(e, "Undo", self.undo, "Ctrl+Z")
        self._act(e, "Redo", self.redo, "Ctrl+Y")
        v = m.addMenu("&View")
        for name in Viewer.LAYOUTS:
            self._act(v, f"Layout: {name}", lambda _=False, n=name: self.viewer.set_layout(n))
        self._act(v, "Reset zoom", lambda: [w.reset_zoom() for w in self.viewer.all()], "Ctrl+0")
        t = m.addMenu("&Tools")
        for key, label, sc in TOOLS:
            self._act(t, label, lambda _=False, k=key: self.set_tool(k), sc)
        self._act(t, "Close polygon", self.poly_finish, "Return")
        self._act(t, "Cancel polygon", self.poly_cancel, "Escape")
        self._act(t, "Brush larger", lambda: self._brush_step(+1), "]")
        self._act(t, "Brush smaller", lambda: self._brush_step(-1), "[")
        self._act(t, "Dosimetry (activity over time → dose)…", self.open_dosimetry)
        self._act(t, "Gaussian filter (new smoothed image)…", lambda: self.add_filtered_image())
        self._act(t, "ROI table (all ROIs)…", self.show_measure_table)
        mk = m.addMenu("&Mask")
        self._mask_menu_actions(mk)
        r = m.addMenu("&Registration")
        self._act(r, "Register images…", self.open_registration)
        self._act(r, "Load transform file…", self.load_transform_dialog)
        self._act(r, "Remove registration of an image…", self.clear_registration)
        h = m.addMenu("&Help")
        self._act(h, f"About {APP_NAME}", lambda: QtWidgets.QMessageBox.about(self, APP_NAME, about_text()))

    @staticmethod
    def _scroll(widget):
        area = QtWidgets.QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QtWidgets.QFrame.NoFrame)
        area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        area.setWidget(widget)
        return area

    def _act(self, menu, text, fn, shortcut=None):
        a = QtGui.QAction(text, self)
        a.triggered.connect(fn)
        if shortcut:
            a.setShortcut(QtGui.QKeySequence(shortcut))
        menu.addAction(a)
        self.addAction(a)
        return a

    # ------------------------------------------------------------------ tabs
    def _project_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Open folder…")
        b.clicked.connect(self.open_folder_dialog)
        r = QtWidgets.QPushButton("Rescan")
        r.clicked.connect(self.rescan)
        row.addWidget(b)
        row.addWidget(r)
        lay.addLayout(row)
        self.proj_label = QtWidgets.QLabel("No project")
        self.proj_label.setWordWrap(True)
        lay.addWidget(self.proj_label)
        self.case_tree = QtWidgets.QTreeWidget()
        self.case_tree.setHeaderLabels(["Case", "Images", "Masks"])
        self.case_tree.setRootIsDecorated(False)
        self.case_tree.itemDoubleClicked.connect(lambda it, _: self.open_case(it.text(0)))
        lay.addWidget(self.case_tree, 1)
        form = QtWidgets.QFormLayout()
        self.readers_edit = QtWidgets.QLineEdit("reader1, reader2")
        self.structure_edit = QtWidgets.QLineEdit("lesions")
        form.addRow("Readers", self.readers_edit)
        form.addRow("Structure", self.structure_edit)
        lay.addLayout(form)
        sv = QtWidgets.QPushButton("Save project settings")
        sv.clicked.connect(self.save_project)
        lay.addWidget(sv)
        return w

    def _segment_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        rbox = QtWidgets.QGroupBox("Reader")
        rf = QtWidgets.QFormLayout(rbox)
        self.reader_box = QtWidgets.QComboBox()
        self.blind = QtWidgets.QCheckBox("Blind to the other readers' masks")
        self.blind.setChecked(True)
        start = QtWidgets.QPushButton("Start / continue correction")
        start.clicked.connect(self.start_reader)
        rf.addRow("Reader", self.reader_box)
        rf.addRow(self.blind)
        rf.addRow(start)
        lay.addWidget(rbox)
        lbox = QtWidgets.QGroupBox("Mask layers (tick = visible, select = edit)")
        ll = QtWidgets.QVBoxLayout(lbox)
        self.layer_list = QtWidgets.QListWidget()
        self.layer_list.setMinimumHeight(90)
        self.layer_list.setMaximumHeight(150)
        self.layer_list.itemChanged.connect(self._layer_visibility)
        self.layer_list.currentTextChanged.connect(self._layer_selected)
        ll.addWidget(self.layer_list)
        row = QtWidgets.QHBoxLayout()
        nb = QtWidgets.QPushButton("New")
        nb.clicked.connect(self.new_layer)
        lb = QtWidgets.QPushButton("Load…")
        lb.clicked.connect(self.load_mask_dialog)
        sb = QtWidgets.QPushButton("Save")
        sb.clicked.connect(self.save_active)
        rb_ = QtWidgets.QPushButton("Del")
        rb_.setToolTip("Removes the selected mask layer from this view. A file already saved on disk is NOT deleted.")
        rb_.clicked.connect(self.remove_active_layer)
        for x in (nb, lb, sb, rb_):
            x.setMinimumWidth(0)
            x.setStyleSheet("padding: 3px 6px;")
            row.addWidget(x)
        ll.addLayout(row)
        lay.addWidget(lbox)
        tbox = QtWidgets.QGroupBox("Tools")
        tf = QtWidgets.QFormLayout(tbox)
        self.tool_buttons: dict[str, list] = {}
        tf.addRow(self._tool_row(("navigate", "paint", "erase", "poly", "grow")))
        note = QtWidgets.QLabel("Paint / Erase: a single click paints one brush spot, dragging paints a stroke. "
                                "Measurement ROIs are drawn in the ROI tab.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 9pt;")
        tf.addRow(note)
        self.poly_erase = QtWidgets.QCheckBox("Polygon removes (instead of adds)")
        self.poly_erase.setToolTip("Polygon tool: click the corners on a slice, close with double-click or Enter "
                                   "(Esc cancels). The inside is added to the mask, or removed with this box ticked.")
        tf.addRow(self.poly_erase)
        mo = QtWidgets.QToolButton()
        mo.setText("Mask operations ▾")
        mo.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        mo.setToolTip("Fill holes, keep largest part, fill between slices, clear, duplicate, combine two layers")
        mmenu = QtWidgets.QMenu(mo)
        self._mask_menu_actions(mmenu)
        mo.setMenu(mmenu)
        tf.addRow(mo)
        self.brush = _spin(0.5, 100, 5, 0.5, 1, " mm")
        self.brush.setToolTip("Radius of the brush in mm. A yellow circle on the image shows its true size. "
                              "Keys [ and ] change it.")
        self.brush_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.brush_slider.setRange(1, 200)                       # 0.5 .. 100 mm in 0.5 mm steps
        self.brush_slider.setValue(int(round(self.brush.value() * 2)))
        self.brush_slider.valueChanged.connect(lambda v: self.brush.setValue(v / 2))
        self.brush.valueChanged.connect(self._brush_changed)
        self.sphere = QtWidgets.QCheckBox("3D sphere")
        bl = QtWidgets.QHBoxLayout()
        bl.addWidget(self.brush)
        bl.addWidget(self.sphere)
        tf.addRow("Brush radius", bl)
        tf.addRow(self.brush_slider)
        self.brush_note = QtWidgets.QLabel()
        self.brush_note.setWordWrap(True)
        self.brush_note.setMinimumWidth(0)
        self.brush_note.setStyleSheet("color: gray; font-size: 9pt;")
        self.brush_note.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        tf.addRow(self.brush_note)
        self.limit_on = QtWidgets.QCheckBox("only where value ≥")
        self.limit_val = _spin(-2000, 1e5, 2.5, 0.5, 2)
        self.limit_hi_on = QtWidgets.QCheckBox("and ≤")
        self.limit_hi = _spin(-2000, 1e5, 50.0, 0.5, 2)
        self.limit_on.setText("≥")
        self.limit_hi_on.setText("≤")
        ll2 = QtWidgets.QHBoxLayout()
        ll2.addWidget(self.limit_on)
        ll2.addWidget(self.limit_val)
        ll3 = QtWidgets.QHBoxLayout()
        ll3.addWidget(self.limit_hi_on)
        ll3.addWidget(self.limit_hi)
        for sp_ in (self.limit_val, self.limit_hi, self.brush):
            sp_.setMinimumWidth(0)
        self.limit_val.setToolTip("Paint only where the image value is at least this (e.g. SUV 2.5)")
        self.limit_hi.setToolTip("... and at most this (e.g. to skip the bladder)")
        tf.addRow("Paint only", ll2)
        tf.addRow("", ll3)
        self.grow_mode = QtWidgets.QComboBox()
        self.grow_mode.addItems(["% of local max", "absolute value"])
        self.grow_val = _spin(0, 1e5, 41, 1, 1)
        self.grow_mode.currentIndexChanged.connect(lambda i: self.grow_val.setValue(41 if i == 0 else 2.5))
        gl = QtWidgets.QHBoxLayout()
        gl.addWidget(self.grow_mode)
        gl.addWidget(self.grow_val)
        self.grow_mode.setToolTip("Grow tool: click on a PET lesion and the whole lesion is outlined automatically.\n"
                                  "'% of local max': voxels above this percentage of the hottest voxel near the click "
                                  "(e.g. 41 %).\n'absolute value': voxels above a fixed SUV (e.g. 2.5).")
        glab = QtWidgets.QLabel("Grow")
        glab.setToolTip(self.grow_mode.toolTip())
        tf.addRow(glab, gl)
        il = QtWidgets.QHBoxLayout()
        self.island_n = QtWidgets.QSpinBox()
        self.island_n.setRange(1, 100000)
        self.island_n.setValue(5)
        ib = QtWidgets.QPushButton("Islands <")
        ib.clicked.connect(self.remove_islands)
        il.addWidget(ib)
        il.addWidget(self.island_n)
        il.addWidget(QtWidgets.QLabel("voxels"))
        tf.addRow(il)
        th = QtWidgets.QHBoxLayout()
        self.thr_mode = QtWidgets.QComboBox()
        self.thr_mode.addItem("Whole image ≥ value (add)", "add")
        self.thr_mode.addItem("Mask: keep ≥ value", "refine_abs")
        self.thr_mode.addItem("Each lesion: keep ≥ % of its max", "refine_rel")
        self.thr_mode.setToolTip("Threshold segmentation on the PET/SPECT image.\n"
                                 "Whole image: every voxel at or above the value is added (physiological uptake too).\n"
                                 "Mask: removes mask voxels below the value (clean a rough mask).\n"
                                 "Each lesion: removes voxels below e.g. 41 % of that lesion's own maximum.")
        self.thr_val = _spin(0, 1e5, 2.5, 0.5, 2)
        self.thr_val.setMinimumWidth(0)
        thb = QtWidgets.QPushButton("Apply")
        thb.setStyleSheet("padding: 3px 6px;")
        thb.clicked.connect(self.threshold_mask)
        self.thr_mode.currentIndexChanged.connect(self._thr_mode_changed)
        th.addWidget(self.thr_val)
        th.addWidget(thb)
        tf.addRow("Threshold", self.thr_mode)
        tf.addRow(th)
        sm = QtWidgets.QHBoxLayout()
        smb = QtWidgets.QPushButton("Smooth")
        smb.setToolTip("Rounds jagged edges and removes thin spurs of the active mask (Gaussian blur of the mask, "
                       "sigma in mm, then re-threshold). One undo step.")
        smb.clicked.connect(self.smooth_mask)
        self.smooth_mm = _spin(0.5, 20, 2.0, 0.5, 1, " mm")
        sm.addWidget(smb)
        sm.addWidget(self.smooth_mm)
        tf.addRow(sm)
        mg = QtWidgets.QHBoxLayout()
        mgb = QtWidgets.QPushButton("Expand ±")
        mgb.setToolTip("Moves the whole outline of the active mask outwards (positive) or inwards (negative) by a "
                       "distance in real millimetres, also for non-cubic voxels. One undo step.")
        mgb.clicked.connect(self.margin_mask)
        self.margin_mm = _spin(-30, 30, 1.0, 0.5, 1, " mm")
        mg.addWidget(mgb)
        mg.addWidget(self.margin_mm)
        tf.addRow(mg)
        ur = QtWidgets.QHBoxLayout()
        ub = QtWidgets.QPushButton("Undo")
        ub.clicked.connect(self.undo)
        rb = QtWidgets.QPushButton("Redo")
        rb.clicked.connect(self.redo)
        ur.addWidget(ub)
        ur.addWidget(rb)
        tf.addRow(ur)
        lay.addWidget(tbox)
        self.lesion_model = PandasModel()
        self.lesion_table = table_view(self.lesion_model)
        self.lesion_table.setMinimumHeight(180)
        self.lesion_table.doubleClicked.connect(self._jump_lesion)
        lesb = QtWidgets.QPushButton("List lesions of the active mask")
        lesb.clicked.connect(self.list_lesions)
        lay.addWidget(lesb)
        lay.addWidget(self.lesion_table, 1)
        return w

    def _tool_row(self, keys):
        """Row of tool buttons; the same tool may have a button on several tabs, all kept in step by set_tool."""
        row = QtWidgets.QHBoxLayout()
        labels = {k: (lab, sc) for k, lab, sc in TOOLS}
        for key in keys:
            lab, sc = labels[key]
            btn = QtWidgets.QToolButton()
            btn.setText(lab)
            btn.setToolTip(f"{lab} ({sc})")
            btn.setCheckable(True)
            btn.setChecked(key == "navigate")
            btn.clicked.connect(lambda _=False, k=key: self.set_tool(k))
            row.addWidget(btn)
            self.tool_buttons.setdefault(key, []).append(btn)
        row.addStretch(1)
        return row

    def _roi_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        help_ = QtWidgets.QLabel("Draw: choose a shape, press the ROI tool (key R) and draw on any slice. "
                                 "Open: with the Navigate tool click inside an ROI (or double-click it in the list) and "
                                 "its details window opens. ROIs are separate from the segmentation masks and are saved "
                                 "as their own files.")
        help_.setWordWrap(True)
        help_.setStyleSheet("color: gray; font-size: 9pt;")
        lay.addWidget(help_)
        lay.addLayout(self._tool_row(("navigate", "roi", "profile")))
        dbox = QtWidgets.QGroupBox("Draw")
        df = QtWidgets.QFormLayout(dbox)
        self.roi_shape = QtWidgets.QComboBox()
        for label, key in (("Freehand (drag around the region)", "freehand"), ("Polygon (click corners, double-click closes)", "polygon"),
                           ("Ellipse (drag from corner to corner)", "ellipse"), ("Circle 2D (click the centre)", "circle"),
                           ("Sphere 3D (click the centre)", "sphere")):
            self.roi_shape.addItem(label, key)
        self.roi_shape.currentIndexChanged.connect(lambda _i: self.set_tool("roi"))
        df.addRow("Shape", self.roi_shape)
        self.roi_radius = _spin(1, 100, 15, 1, 1, " mm")
        self.roi_radius.setToolTip("Radius of circle and sphere ROIs. 15 mm = 3 cm diameter (PERCIST liver reference).")
        self.roi_radius.valueChanged.connect(lambda _v: self.viewer.refresh())
        df.addRow("Radius", self.roi_radius)
        self.roi_mode = QtWidgets.QComboBox()
        self.roi_mode.addItem("New ROI", "new")
        self.roi_mode.addItem("Add to the selected ROI", "add")
        self.roi_mode.addItem("Remove from the selected ROI", "remove")
        self.roi_mode.setToolTip("Correct an ROI that went wrong: select it in the list, choose Add or Remove and draw "
                                 "the part to add or cut away (any shape). 'Undo ROI edit' goes back step by step.")
        df.addRow("Drawing", self.roi_mode)
        ub = QtWidgets.QPushButton("Undo ROI edit")
        ub.setToolTip("Undoes the last add/remove on the selected ROI (up to 20 steps)")
        ub.clicked.connect(self.undo_roi_edit)
        df.addRow(ub)
        lay.addWidget(dbox)
        lbox = QtWidgets.QGroupBox("ROIs of this case")
        ll = QtWidgets.QVBoxLayout(lbox)
        self.roi_list = QtWidgets.QListWidget()
        self.roi_list.setMaximumHeight(150)
        self.roi_list.itemChanged.connect(self._roi_item_changed)
        self.roi_list.currentItemChanged.connect(self._roi_selected)
        self.roi_list.itemDoubleClicked.connect(lambda it: self.open_roi_details(it.data(QtCore.Qt.UserRole)))
        ll.addWidget(self.roi_list)
        g = QtWidgets.QGridLayout()
        for n_, (label, fn, tip) in enumerate([
                ("Details", lambda: self.open_roi_details(), "Window with all numbers of the selected ROI"),
                ("Rename", self.rename_roi, "New name for the selected ROI"),
                ("Delete", self.delete_roi, "Removes the selected ROI from the view (a saved file stays on disk)"),
                ("Save…", self.save_roi, "Saves the selected ROI as a label file plus a .roi.json"),
                ("Save all…", self.save_all_rois, "Saves every ROI into one folder"),
                ("Load…", self.load_roi_dialog, "Loads ROI files saved before"),
                ("Mask → ROI", self.mask_to_roi, "Copies the active segmentation mask into a new ROI"),
                ("Table", self.show_measure_table, "Table of all ROIs, with counts-to-activity calibration")]):
            b = QtWidgets.QPushButton(label)
            b.setToolTip(tip)
            b.setStyleSheet("padding: 3px 6px;")
            b.setMinimumWidth(0)
            b.clicked.connect(fn)
            g.addWidget(b, n_ // 4, n_ % 4)
        ll.addLayout(g)
        lay.addWidget(lbox)
        pbox = QtWidgets.QGroupBox("PERCIST background threshold")
        pl = QtWidgets.QGridLayout(pbox)
        for n_, (label, fn, tip) in enumerate([
                ("Selected ROI = reference", self.roi_set_reference, "Use the selected ROI (3 cm sphere in normal liver) as the background"),
                ("→ Threshold", self.roi_to_threshold, "1.5 x reference mean + 2 SD into the Threshold value (Segment tab)")]):
            b = QtWidgets.QPushButton(label)
            b.setToolTip(tip)
            b.setMinimumWidth(0)
            b.clicked.connect(fn)
            pl.addWidget(b, 0, n_)
        lay.addWidget(pbox)
        self.roi_text = QtWidgets.QLabel("Profile tool (key O): click two points for a line profile with FWHM.")
        self.roi_text.setWordWrap(True)
        self.roi_text.setMinimumWidth(0)
        self.roi_text.setStyleSheet("color: gray; font-size: 9pt;")
        lay.addWidget(self.roi_text)
        lay.addStretch(1)
        return w

    def _radiomics_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        f = QtWidgets.QFormLayout()
        self.r_disc = QtWidgets.QComboBox()
        self.r_disc.addItems(["fixed bin width", "fixed bin count"])
        self.r_bin = _spin(0.001, 1000, 0.5, 0.25, 3)
        self.r_disc.setToolTip("Grey-level discretisation for the TEXTURE features (GLCM, GLRLM, GLSZM, GLDM, NGTDM); "
                               "it is not used for segmentation. Fixed bin width keeps the SUV scale comparable between "
                               "patients (IBSI; PET), fixed bin count is for images without an absolute scale (MR).")
        self.r_bin.setToolTip("Bin width in the unit of the image: SUV 0.5 for PET (the value fixed in this thesis), "
                              "HU 25 for CT; SPECT in counts has no fixed scale, use a bin count (e.g. 32) or a width that "
                              "gives about 30-130 bins in the lesions. With 'fixed bin count' this is the number of bins.")
        self.r_res = _spin(0, 20, 4.0, 0.5, 2, " mm")
        self.r_res.setToolTip("Isotropic resampling; 0 = original voxels")
        self.r_reseg_on = QtWidgets.QCheckBox("re-segment: keep values ≥")
        self.r_reseg_on.setChecked(True)
        self.r_reseg = _spin(-2000, 1e5, 0.0, 0.5, 2)
        rr = QtWidgets.QHBoxLayout()
        rr.addWidget(self.r_reseg_on)
        rr.addWidget(self.r_reseg)
        self.r_method = QtWidgets.QComboBox()
        self.r_method.addItems(["PyRadiomics (default)", "IBSI (grid centres aligned, mask interpolated)"])
        self.r_method.setToolTip(
            "How the image and mask are resampled before the features. PyRadiomics: as in earlier PySegment "
            "versions and the PROSPeR pipeline. IBSI: the grid and the mask interpolation of the IBSI reference "
            "manual (validated against the IBSI chest CT reference values; see validation/ibsi1); with IBSI the LoG "
            "filter is also computed as IBSI 2 defines it (exact kernel, no sigma^2 scaling; validation/ibsi2). The two give "
            "slightly different values; use one method for a whole study.")
        self.r_interp = QtWidgets.QComboBox()
        self.r_interp.addItems(["tricubic spline", "trilinear", "nearest neighbour"])
        self.r_round = QtWidgets.QCheckBox("round to whole numbers after interpolation (CT in HU)")
        self.r_reseg_hi_on = QtWidgets.QCheckBox("and values ≤")
        self.r_reseg_hi = _spin(-2000, 1e6, 400.0, 0.5, 2)
        rh = QtWidgets.QHBoxLayout()
        rh.addWidget(self.r_reseg_hi_on)
        rh.addWidget(self.r_reseg_hi)
        self.r_outlier = QtWidgets.QCheckBox("outlier filter: keep mean ± 3 SD (IBSI method only)")
        self.r_outlier.setToolTip("IBSI outlier re-segmentation, applied after the value range.")

        def _method_changed(i):
            self.r_outlier.setEnabled(i == 1)
            self.r_round.setEnabled(i == 1)
            if i == 0:
                self.r_outlier.setChecked(False)
                self.r_round.setChecked(False)
        self.r_method.currentIndexChanged.connect(_method_changed)
        _method_changed(0)
        self.r_min_tex = QtWidgets.QSpinBox()
        self.r_min_tex.setRange(1, 100000)
        self.r_min_tex.setValue(64)
        self.r_per_lesion = QtWidgets.QCheckBox("per lesion as well as the whole mask")
        self.r_per_lesion.setChecked(True)
        self.r_classes = {c: QtWidgets.QCheckBox(c) for c in ["shape", "firstorder", *TEXTURE]}
        cl = QtWidgets.QGridLayout()
        for i, (c, cb) in enumerate(self.r_classes.items()):
            cb.setChecked(True)
            cl.addWidget(cb, i // 4, i % 4)
        self.r_log = QtWidgets.QCheckBox("LoG (σ 2, 4, 6 mm)")
        self.r_wav = QtWidgets.QCheckBox("Wavelet")
        il = QtWidgets.QHBoxLayout()
        il.addWidget(QtWidgets.QLabel("Original +"))
        il.addWidget(self.r_log)
        il.addWidget(self.r_wav)
        f.addRow("Discretisation", self.r_disc)
        f.addRow("Bin", self.r_bin)
        f.addRow("Resampling", self.r_res)
        f.addRow("Method", self.r_method)
        f.addRow("Interpolation", self.r_interp)
        f.addRow(self.r_round)
        f.addRow(rr)
        f.addRow(rh)
        f.addRow(self.r_outlier)
        f.addRow("Texture if ≥ voxels", self.r_min_tex)
        f.addRow(self.r_per_lesion)
        f.addRow(cl)
        f.addRow("Images", il)
        lay.addLayout(f)
        sc = QtWidgets.QHBoxLayout()
        self.r_scope = QtWidgets.QComboBox()
        self.r_scope.addItems(["Current case, active mask", "All cases"])
        self.r_which = QtWidgets.QComboBox()
        self.r_which.setEditable(True)
        self.r_which.setToolTip("All cases: a reader (that reader's saved masks) or a mask name")
        sc.addWidget(self.r_scope)
        sc.addWidget(self.r_which)
        lay.addLayout(sc)
        row = QtWidgets.QHBoxLayout()
        run = QtWidgets.QPushButton("Extract features")
        run.clicked.connect(self.run_radiomics)
        exp = QtWidgets.QPushButton("Export CSV…")
        exp.clicked.connect(lambda: self.export_table(self.feat_model.df, "features", with_settings=True))
        row.addWidget(run)
        row.addWidget(exp)
        lay.addLayout(row)
        self.feat_model = PandasModel()
        ft = table_view(self.feat_model)
        ft.setMinimumHeight(200)
        lay.addWidget(ft, 1)
        return w

    def _agreement_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        cb = QtWidgets.QGroupBox("Current case")
        cf = QtWidgets.QFormLayout(cb)
        self.ag_a = QtWidgets.QComboBox()
        self.ag_b = QtWidgets.QComboBox()
        cf.addRow("Mask A", self.ag_a)
        cf.addRow("Mask B", self.ag_b)
        go = QtWidgets.QPushButton("Compare")
        go.clicked.connect(self.compare_layers)
        cf.addRow(go)
        self.ag_label = QtWidgets.QLabel("")
        self.ag_label.setWordWrap(True)
        cf.addRow(self.ag_label)
        lay.addWidget(cb)
        pb = QtWidgets.QGroupBox("All cases")
        pf = QtWidgets.QFormLayout(pb)
        self.pa = QtWidgets.QComboBox()
        self.pb = QtWidgets.QComboBox()
        for c in (self.pa, self.pb):
            c.setEditable(True)
        pf.addRow("Reader / mask A", self.pa)
        pf.addRow("Reader / mask B", self.pb)
        self.icc_thr = _spin(0, 1, 0.75, 0.05, 2)
        pf.addRow("ICC threshold", self.icc_thr)
        r = QtWidgets.QHBoxLayout()
        b1 = QtWidgets.QPushButton("Dice / HD95")
        b1.clicked.connect(self.project_agreement)
        b2 = QtWidgets.QPushButton("ICC of features")
        b2.setToolTip("Uses the settings of the Radiomics tab")
        b2.clicked.connect(self.project_icc)
        r.addWidget(b1)
        r.addWidget(b2)
        pf.addRow(r)
        self.icc_label = QtWidgets.QLabel("")
        self.icc_label.setWordWrap(True)
        pf.addRow(self.icc_label)
        lay.addWidget(pb)
        self.ag_model = PandasModel()
        at = table_view(self.ag_model)
        at.setMinimumHeight(200)
        lay.addWidget(at, 1)
        exp = QtWidgets.QPushButton("Export table…")
        exp.clicked.connect(lambda: self.export_table(self.ag_model.df, "agreement"))
        lay.addWidget(exp)
        return w

    def _display_tab(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        self.pet_max = _spin(0.1, 1000, 10, 1, 1)
        self.pet_max.valueChanged.connect(lambda v: self._set_win("PET", [0.0, v]))
        self.ct_lo = _spin(-2000, 4000, -160, 10, 0)
        self.ct_hi = _spin(-2000, 4000, 240, 10, 0)
        for s in (self.ct_lo, self.ct_hi):
            s.valueChanged.connect(lambda _: self._set_win("CT", [self.ct_lo.value(), self.ct_hi.value()]))
        self.opacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(60)
        self.opacity.valueChanged.connect(self._set_opacity)
        self.pet_cmap = QtWidgets.QComboBox()
        self.pet_cmap.addItems(["inferno", "magma", "plasma", "viridis", "gray"])
        self.pet_cmap.currentTextChanged.connect(lambda _: self.viewer.refresh())
        self.cursor_cb = QtWidgets.QCheckBox("Show cursor lines")
        self.cursor_cb.setChecked(True)
        self.cursor_cb.toggled.connect(self._toggle_cursor)
        self.layout_box = QtWidgets.QComboBox()
        self.layout_box.addItems(Viewer.LAYOUTS)
        self.layout_box.currentTextChanged.connect(self.viewer.set_layout)
        f.addRow("PET window 0 –", self.pet_max)
        f.addRow("CT window from", self.ct_lo)
        f.addRow("CT window to", self.ct_hi)
        f.addRow("Overlay opacity", self.opacity)
        self.base_box, self.over_box = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        self.base_box.setToolTip("Image shown as the grey background")
        self.over_box.setToolTip("Image shown in colour over the background (choose the registered image here)")
        self.base_box.activated.connect(self._display_picked)
        self.over_box.activated.connect(self._display_picked)
        f.addRow("Background image", self.base_box)
        f.addRow("Overlay image", self.over_box)
        self.mask_fill_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.mask_fill_slider.setRange(0, 100)
        self.mask_fill_slider.setValue(100)
        self.mask_fill_slider.setToolTip("Fill of the mask colour; 0 = outline only")
        self.mask_fill_slider.valueChanged.connect(lambda v: (setattr(self, "mask_fill", v), self.viewer.refresh()))
        f.addRow("Mask fill", self.mask_fill_slider)
        f.addRow("PET colours", self.pet_cmap)
        f.addRow("Layout", self.layout_box)
        f.addRow(self.cursor_cb)
        f.addRow(QtWidgets.QLabel("Mouse: wheel = slice, Ctrl+wheel = zoom, right drag = zoom,\n"
                                  "middle drag = pan, Shift+left drag = window."))
        return w

    # ================================================================== display callbacks (used by views)
    def display_names(self):
        s = self.session
        pet, ct = s.pet_name, s.ct_name
        base, over = (ct, pet) if (pet and ct) else ((pet or ct or s.ref.name), None)
        if s.base_pick in s.volumes:
            base = s.base_pick
        if s.over_pick is not None:
            over = s.over_pick if s.over_pick in s.volumes else None
        elif over == base:
            over = None
        if over == base:
            over = None
        return base, over

    def mask_fill_scale(self):
        return self.mask_fill / 100.0

    def _fill_display_boxes(self):
        s = self.session
        if not s:
            return
        base, over = self.display_names()
        for box, cur, extra in ((self.base_box, base, []), (self.over_box, over, ["(none)"])):
            box.blockSignals(True)
            box.clear()
            box.addItems(extra + list(s.volumes))
            box.setCurrentText(cur if cur else "(none)")
            box.blockSignals(False)

    def _display_picked(self, *_):
        s = self.session
        if not s:
            return
        s.base_pick = self.base_box.currentText()
        o = self.over_box.currentText()
        s.over_pick = "" if o == "(none)" else o
        self.viewer.refresh()

    def show_pair(self, base: str, over: str | None):
        """Show ``over`` in colour on ``base`` (used after a registration so the moved image is visible)."""
        self.session.base_pick, self.session.over_pick = base, (over or "")
        self._fill_display_boxes()
        self.viewer.refresh()

    def levels(self, name):
        v = self.session.volumes[name]
        win = self.win.get(v.kind)
        if win is None:
            a = self.session.can[name]
            lo, hi = np.percentile(a[::4, ::4, ::4], [1, 99.5])
            win = [float(lo), float(hi)]
            self.win[v.kind] = win
        return tuple(win)

    def lut(self, name, base: bool):
        kind = self.session.volumes[name].kind
        if kind in ("PET", "SPECT"):
            if base:                                           # PET alone: inverted grey, like most readers
                return _lut("gray", invert=True)
            return _lut(self.pet_cmap.currentText())
        return _lut("gray")

    def _brush_step(self, d: int):
        v = self.brush.value()
        step = 0.5 if v < 5 else 1.0 if v < 20 else 2.0
        self.brush.setValue(max(0.5, v + d * step))

    def _brush_changed(self, v):
        self.brush_slider.blockSignals(True)
        self.brush_slider.setValue(int(round(v * 2)))
        self.brush_slider.blockSignals(False)
        if self.session:
            sp = self.session.orient.spacing_xyz
            vox = min(sp)
            self.brush_note.setText(
                f"Voxel {sp[0]:.1f} x {sp[1]:.1f} x {sp[2]:.1f} mm. "
                + ("The radius is below half a voxel: a stroke paints one voxel at a time." if v < 0.5 * vox else
                   f"About {2 * v / vox:.1f} voxels across."))
        self.viewer.refresh()

    def _set_win(self, kind, w):
        self.win[kind] = w
        self.viewer.refresh()

    def _set_opacity(self, v):
        self.overlay_opacity = v / 100.0
        self.viewer.refresh()

    def _toggle_cursor(self, on):
        self.show_cursor = on
        self.viewer.refresh()

    def on_window_drag(self, dx, dy):
        if not self.session:
            return
        name = self.display_names()[1] or self.display_names()[0]
        kind = self.session.volumes[name].kind
        lo, hi = self.levels(name)
        width, centre = hi - lo, (hi + lo) / 2
        width = max(1e-3, width * (1 + dx / 200))
        centre += dy * width / 200
        self.win[kind] = [centre - width / 2, centre + width / 2]
        if kind == "PET":
            self.pet_max.blockSignals(True)
            self.pet_max.setValue(self.win[kind][1])
            self.pet_max.blockSignals(False)
        self.viewer.refresh()

    def on_slice_changed(self, view, index):
        c = list(self.cursor)
        c[VIEW_AXIS[view]] = index
        self.cursor = tuple(c)
        self.viewer.views[view].refresh()
        for v in self.viewer.all():
            v.set_cursor(self.cursor)

    def set_cursor(self, idx, jump=False):
        self.cursor = tuple(int(i) for i in idx)
        for name, v in self.viewer.views.items():
            v.index = self.cursor[VIEW_AXIS[name]]
        self.viewer.refresh()
        self.on_hover(self.cursor)

    def on_hover(self, idx):
        if not self.session:
            return
        vals = self.session.value_at(*idx)
        units = {n: v.unit for n, v in self.session.volumes.items()}
        txt = " · ".join(f"{n} {vals[n]:.2f} {units[n]}".strip() for n in vals)
        orig = self.session.orient.to_original(idx, self.session.ref.array.shape)
        self.statusBar().showMessage(f"{txt}   voxel (x {orig[2]}, y {orig[1]}, z {orig[0]})")

    def on_click(self, view, r, c, double=False):
        if not self.session:
            return
        idx_view = self.viewer.views[view].index
        from rsw.orient import plane_to_index
        idx = plane_to_index(view, idx_view, r, c)
        if not all(0 <= i < n for i, n in zip(idx, self.session.shape)):
            return
        if self.tool == "poly":
            self.poly_click(view, idx_view, r, c, double)
        elif self.tool in ("paint", "erase") and self.viewer.views[view].paintable:
            self.on_paint(view, r, c, True, True)          # a single click paints (or erases) one brush spot
        elif self.tool == "grow":
            self.grow_at(view, idx_view, r, c)
        elif self.tool == "roi":
            self.roi_click(view, idx_view, r, c, idx, double)
        elif self.tool == "profile":
            self.profile_click(idx)
        elif self.tool == "label":
            self.add_label(view, idx_view, r, c)
        elif self.tool == "navigate":
            self.set_cursor(idx)
            hit = self.session.roi_hit(idx)
            if hit and not double:
                self.open_roi_details(hit)


    # ------------------------------------------------------------------ measurements (ROI, profile, filter)
    def _mm(self, idx):
        sp = self.session.orient.spacing_zyx
        return [float(i) * s for i, s in zip(idx, sp)]

    # ---- ROI drawing (tool "roi") and the ROI list
    def ring_radius(self):
        """Radius (mm) of the circle shown under the mouse, or None for no circle."""
        if self.tool in ("paint", "erase"):
            return float(self.brush.value())
        if self.tool == "roi" and self.roi_shape.currentData() in ("circle", "sphere"):
            return float(self.roi_radius.value())
        return None

    def roi_click(self, view, index, r, c, idx, double=False):
        s, shape = self.session, self.roi_shape.currentData()
        if shape == "polygon":
            self.poly_click(view, index, r, c, double, target="roi")
        elif shape == "circle":
            from rsw.gui.viewer import pixel_size
            from rsw.roi import circle_region, plane_shape
            reg = circle_region(plane_shape(s.shape, view), r, c, self.roi_radius.value(), pixel_size(view, s.orient.spacing_zyx))
            self._roi_apply(reg, "circle", view, index)
        elif shape == "sphere":
            reg, _ = s.roi_at(idx, self.roi_radius.value())
            self._roi_apply(reg, "sphere")
        else:
            self.status("Freehand and ellipse ROIs are drawn by dragging with the left mouse button.")

    def on_roi_drag(self, view, r, c, start, finish):
        s = self.session
        if not s:
            return
        shape = self.roi_shape.currentData()
        index = self.viewer.views[view].index
        if start or self.poly is None or self.poly.get("view") != view:
            self.poly = {"view": view, "index": index, "pts": [(r, c)], "target": "roi", "start": (r, c)}
        elif shape == "freehand":
            if self.poly["pts"][-1] != (r, c):
                self.poly["pts"].append((r, c))
        elif shape == "ellipse":
            r0, c0 = self.poly["start"]
            t = np.linspace(0, 2 * np.pi, 48)
            rc, cc, ar, ac = (r0 + r) / 2, (c0 + c) / 2, abs(r - r0) / 2, abs(c - c0) / 2
            self.poly["pts"] = [(rc + ar * np.sin(a), cc + ac * np.cos(a)) for a in t]
            self.poly["end"] = (r, c)
        self.viewer.views[view]._draw_poly()
        if not finish:
            return
        poly, self.poly = self.poly, None
        from rsw.roi import ellipse_region, plane_shape
        shape2d = plane_shape(s.shape, view)
        if shape == "ellipse":
            r0, c0 = poly["start"]
            r1, c1 = poly.get("end", (r, c))
            reg = ellipse_region(shape2d, r0, c0, r1, c1)
        else:
            if len(poly["pts"]) < 3:
                self.status("Drag around the region to draw a freehand ROI.")
                self.viewer.refresh()
                return
            reg = polygon_mask(shape2d, [(a + 0.5, b + 0.5) for a, b in poly["pts"]])
        self._roi_apply(reg, shape, view, index)

    def _roi_apply(self, region, shape, view="", index=None):
        """New ROI, or add to / remove from the selected ROI, depending on the 'Drawing' choice."""
        s = self.session
        mode = self.roi_mode.currentData()
        if view:
            region = self._plane3d(view, index, region)
        if mode != "new":
            target = s.roi_selected if s.roi_selected in s.rois else None
            if target is None:
                self.status("Select the ROI to edit in the list first.")
                self.viewer.refresh()
                return
            try:
                n = s.edit_roi(target, region, mode)
            except ValueError as e:
                self.status(str(e))
                self.viewer.refresh()
                return
            self.status(f"{target}: {'added' if mode == 'add' else 'removed'}; now {n} voxels (Undo ROI edit goes back)")
            self._roi_created(target, edited=True)
            return
        self._roi_created(s.add_roi(region, shape, view, index))

    def _plane3d(self, view, index, region2d):
        from rsw.roi import plane_box
        reg3 = np.zeros(self.session.shape, bool)
        reg3[plane_box(view, index)] = region2d
        return reg3

    def undo_roi_edit(self):
        s = self.session
        name = s.roi_selected if s and s.roi_selected in s.rois else None
        if name and s.undo_roi(name):
            self._roi_created(name, edited=True)
        else:
            self.status("Nothing to undo for the selected ROI.")

    def _roi_created(self, name, edited=False):
        self._refresh_rois()
        self.viewer.refresh()
        if edited and getattr(self, "roi_dlg", None) is not None and self.roi_dlg.isVisible() and self.roi_dlg.name == name:
            self.roi_dlg.show_roi(name)
        info, rows = self.session.roi_details(name)
        m = next((r for r in rows if r["image"] == self.session.measure_name), rows[0])
        self.status(f"{name} ({info['shape']}): {info['volume_ml']:.2f} mL, mean {m['mean']:.4g} {m['unit']} on {m['image']}; "
                    "click inside it with the Navigate tool for all details")
        if getattr(self, "measure_dlg", None) is not None and self.measure_dlg.isVisible():
            self.measure_dlg.refresh()

    def _refresh_rois(self):
        s = self.session
        self.roi_list.blockSignals(True)
        self.roi_list.clear()
        for name, r in (s.rois.items() if s else []):
            it = QtWidgets.QListWidgetItem(("● " if not r.saved else "") + f"{name}  [{r.shape}{', ' + r.view if r.view else ''}]")
            it.setData(QtCore.Qt.UserRole, name)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if r.visible else QtCore.Qt.Unchecked)
            it.setForeground(QtGui.QColor(*r.color))
            self.roi_list.addItem(it)
            if name == s.roi_selected:
                self.roi_list.setCurrentItem(it)
        self.roi_list.blockSignals(False)

    def _roi_item_changed(self, it):
        s = self.session
        name = it.data(QtCore.Qt.UserRole)
        if s and name in s.rois:
            s.rois[name].visible = it.checkState() == QtCore.Qt.Checked
            self.viewer.refresh()

    def _roi_selected(self, it, _prev=None):
        if it is not None and self.session:
            self.session.roi_selected = it.data(QtCore.Qt.UserRole)
            self.viewer.refresh()

    def _selected_roi(self):
        s = self.session
        if not s or not s.rois:
            self.status("No ROI yet: draw one with the ROI tool (ROI tab).")
            return None
        return s.roi_selected if s.roi_selected in s.rois else next(reversed(s.rois))

    def open_roi_details(self, name=None):
        from rsw.gui.roi_dialog import RoiDetailsDialog
        name = name or self._selected_roi()
        if not name:
            return None
        self.session.roi_selected = name
        if getattr(self, "roi_dlg", None) is None:
            self.roi_dlg = RoiDetailsDialog(self)
        self.roi_dlg.show_roi(name)
        self.roi_dlg.show()
        self.roi_dlg.raise_()
        self._refresh_rois()
        return self.roi_dlg

    def rename_roi(self):
        name = self._selected_roi()
        if not name:
            return
        new, ok = QtWidgets.QInputDialog.getText(self, APP_NAME, "New name of the ROI:", text=name)
        if ok and new.strip():
            try:
                self.session.rename_roi(name, new.strip())
            except ValueError as e:
                self.status(str(e))
            self._refresh_rois()

    def delete_roi(self):
        name = self._selected_roi()
        if not name:
            return
        r = self.session.rois[name]
        if QtWidgets.QMessageBox.question(self, APP_NAME, f"Remove the ROI '{name}' from this view?"
                                          + ("" if r.saved else "\n\nIt is NOT saved.")) != QtWidgets.QMessageBox.Yes:
            return
        self.session.remove_roi(name)
        self._refresh_rois()
        self.viewer.refresh()

    def _roi_folder(self):
        s = self.session
        if s.project is not None and s.case is not None:
            try:
                return str(s.project.path(s.case.images[s.ref.name]).parent)
            except Exception:  # noqa: BLE001
                pass
        return str(Path(s.ref.path).parent) if s.ref.path else ""

    def save_roi(self, path=None):
        name = self._selected_roi()
        if not name:
            return None
        if not path:
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save ROI", str(Path(self._roi_folder()) / f"roi_{safe}.nii.gz"),
                                                            "NIfTI (*.nii.gz *.nii)")
        if not path:
            return None
        out = self.session.save_roi(name, path)
        self._refresh_rois()
        self.status(f"ROI '{name}' saved: {out}")
        return out

    def save_all_rois(self, folder=None):
        s = self.session
        if not s or not s.rois:
            return []
        folder = folder or QtWidgets.QFileDialog.getExistingDirectory(self, "Folder for the ROI files", self._roi_folder())
        if not folder:
            return []
        out = []
        for name in list(s.rois):
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
            out.append(s.save_roi(name, Path(folder) / f"roi_{safe}.nii.gz"))
        self._refresh_rois()
        self.status(f"{len(out)} ROI file(s) saved in {folder}")
        return out

    def load_roi_dialog(self, paths=None):
        if not self.session:
            return
        if not paths:
            paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Load ROI files", self._roi_folder(), "NIfTI (*.nii.gz *.nii)")
        for p in paths or []:
            try:
                self.session.load_roi(p)
            except Exception as e:  # noqa: BLE001
                self.status(f"{Path(p).name}: {e}")
        self._refresh_rois()
        self.viewer.refresh()

    def mask_to_roi(self):
        s = self.session
        if not (s and s.layer()):
            self.status("Select a mask layer first (Segment tab).")
            return
        try:
            self._roi_created(s.roi_from_layer())
        except ValueError as e:
            self.status(str(e))

    def open_dosimetry(self):
        from rsw.gui.dosimetry_dialog import DosimetryDialog
        if getattr(self, "dosimetry_dlg", None) is None:
            self.dosimetry_dlg = DosimetryDialog(self)
        self.dosimetry_dlg.show()
        self.dosimetry_dlg.raise_()
        return self.dosimetry_dlg

    def show_measure_table(self):
        from rsw.gui.measure_dialog import MeasureDialog
        if getattr(self, "measure_dlg", None) is None:
            self.measure_dlg = MeasureDialog(self)
        self.measure_dlg.refresh()
        self.measure_dlg.show()
        self.measure_dlg.raise_()

    def profile_click(self, idx):
        from rsw import measure
        s = self.session
        if self.profile_pts is None or len(self.profile_pts) >= 2:
            self.profile_pts = [tuple(idx)]
            self.roi_text.setText("Profile: first point set; click the second point.")
            return
        self.profile_pts.append(tuple(idx))
        name = s.measure_name
        d, v = measure.line_profile(s.can[name], self.profile_pts[0], self.profile_pts[1], s.orient.spacing_zyx)
        w = measure.fwhm(d, v)
        self.last_profile = (d, v)
        self.last_roi = {"kind": "profile", "image": name, "length_mm": float(d[-1]), "max": float(v.max()),
                         "mean": float(v.mean()), "fwhm_mm": w,
                         **dict(zip(("z_mm", "y_mm", "x_mm"), self._mm(self.profile_pts[0])))}
        self.roi_text.setText(f"Profile {d[-1]:.1f} mm on {name}: max {v.max():.2f}, FWHM "
                              + ("not defined (no peak)" if w != w else f"{w:.1f} mm"))
        self.show_profile(d, v, name)

    def show_profile(self, d, v, name):
        import pyqtgraph as pg
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Line profile – {name}")
        lay = QtWidgets.QVBoxLayout(dlg)
        pw = pg.PlotWidget()
        pw.plot(d, v, pen=pg.mkPen("y", width=2))
        pw.setLabel("bottom", "distance (mm)")
        pw.setLabel("left", self.session.volumes[name].unit or "value")
        lay.addWidget(pw)
        b = QtWidgets.QPushButton("Save profile as CSV")
        b.clicked.connect(lambda: self._save_profile(d, v))
        lay.addWidget(b)
        dlg.resize(520, 340)
        dlg.show()
        self._profile_dlg = dlg

    def _save_profile(self, d, v):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save profile", "profile.csv", "CSV (*.csv)")
        if path:
            pd.DataFrame({"distance_mm": d, "value": v}).to_csv(path, index=False)
            self.status(f"Profile saved: {path}")

    def roi_set_reference(self):
        name = self._selected_roi()
        if not name:
            return
        info, rows = self.session.roi_details(name)
        m = next((r for r in rows if r["image"] == self.session.measure_name), None)
        if m is None:
            return
        self.ref_roi = (m["mean"], m["sd"])
        self.status(f"Reference = {name} on {m['image']}: mean {m['mean']:.3f}, SD {m['sd']:.3f}")

    def roi_to_threshold(self):
        from rsw import measure
        if not self.ref_roi:
            self.status("Set a reference ROI first (ROI tab).")
            return
        thr = measure.percist_threshold(*self.ref_roi)
        self.thr_mode.setCurrentIndex(0)
        self.thr_val.setValue(thr)
        self.status(f"PERCIST threshold = 1.5 x {self.ref_roi[0]:.2f} + 2 x {self.ref_roi[1]:.2f} = {thr:.2f}; "
                    "press Apply in Threshold (Segment tab) to segment.")

    def add_filtered_image(self, sigma=None):
        s = self.session
        if not s:
            return None
        if sigma is None:
            sigma, ok = QtWidgets.QInputDialog.getDouble(self, APP_NAME, "Gaussian filter: sigma in mm (FWHM = 2.355 x sigma)",
                                                         2.0, 0.1, 50.0, 1)
            if not ok:
                return None
        name = s.add_filtered(s.measure_name, float(sigma))
        self._fill_display_boxes()
        self.viewer.refresh()
        self.status(f"New image '{name}' added (Gaussian, sigma {sigma:g} mm); choose it in the Display boxes.")
        return name

    # ------------------------------------------------------------------ DICOM tools
    def dicom_header_dialog(self):
        from rsw import dicomtools
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "DICOM series folder")
        if not folder:
            return
        try:
            rows = dicomtools.read_header(folder)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, APP_NAME, str(e))
            return
        ok, msg = dicomtools.suv_readiness(folder)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("DICOM header")
        lay = QtWidgets.QVBoxLayout(dlg)
        lab = QtWidgets.QLabel(msg)
        lab.setWordWrap(True)
        lay.addWidget(lab)
        lay.addWidget(table_view(PandasModel(pd.DataFrame(rows, columns=["tag", "value"]))))
        dlg.resize(620, 520)
        dlg.exec()

    def dicom_anonymise_dialog(self):
        from rsw import dicomtools
        src = QtWidgets.QFileDialog.getExistingDirectory(self, "DICOM folder to anonymise")
        if not src:
            return
        dst = QtWidgets.QFileDialog.getExistingDirectory(self, "Empty output folder for the anonymised copy")
        if not dst:
            return
        code, ok = QtWidgets.QInputDialog.getText(self, APP_NAME, "Study code that replaces name and ID:", text="PT001")
        if not ok or not code:
            return
        days, ok = QtWidgets.QInputDialog.getInt(self, APP_NAME, "Shift all dates by (days; 0 = keep dates):", 0, -36500, 36500)
        if not ok:
            return
        res = dicomtools.anonymize_folder(src, dst, code, shift_days=days or None)
        QtWidgets.QMessageBox.information(
            self, APP_NAME, f"{res['written']} file(s) written, {res['skipped_non_dicom']} skipped.\n\n"
            "Weight, dose and times are kept so SUV still works. Text burned into the pixels is NOT removed: "
            "check the images before sharing.")

    # ------------------------------------------------------------------ polygon tool and mask operations
    def poly_click(self, view, index, r, c, double=False, target="mask"):
        s = self.session
        if target == "mask" and not s.layer():
            self.status("Select or create a mask layer first (Segment tab).")
            return
        if self.poly is None or self.poly["view"] != view or self.poly["index"] != index or self.poly.get("target", "mask") != target:
            self.poly = {"view": view, "index": index, "pts": [], "target": target}
        if not (double and self.poly["pts"] and self.poly["pts"][-1] == (r, c)):
            self.poly["pts"].append((r, c))
        if double:
            self.poly_finish()
        else:
            self.status(f"Polygon: {len(self.poly['pts'])} corners; double-click or Enter closes, Esc cancels")
            self.viewer.refresh()

    def poly_cancel(self):
        self.poly = None
        self.viewer.refresh()

    def poly_finish(self):
        s, poly = self.session, self.poly
        if not s or not poly:
            return
        self.poly = None
        if poly.get("target") == "roi":
            if len(poly["pts"]) < 3:
                self.status("A polygon needs at least 3 corners.")
                self.viewer.refresh()
                return
            from rsw.roi import plane_shape
            reg = polygon_mask(plane_shape(s.shape, poly["view"]), [(r + 0.5, c + 0.5) for r, c in poly["pts"]])
            self._roi_apply(reg, "polygon", poly["view"], poly["index"])
            return
        layer = s.layer()
        if layer is None or layer.locked:
            self.status("No editable active layer.")
            self.viewer.refresh()
            return
        if len(poly["pts"]) < 3:
            self.status("A polygon needs at least 3 corners.")
            self.viewer.refresh()
            return
        view, idx = poly["view"], poly["index"]
        slc = s.can[s.ref.name]
        shape2d = {"axial": slc.shape[1:], "coronal": (slc.shape[0], slc.shape[2]), "sagittal": slc.shape[:2]}[view]
        region = polygon_mask(shape2d, [(r + 0.5, c + 0.5) for r, c in poly["pts"]])
        box = {"axial": (idx, slice(None), slice(None)), "coronal": (slice(None), idx, slice(None)),
               "sagittal": (slice(None), slice(None), idx)}[view]
        n = layer.apply(region, box, 0 if self.poly_erase.isChecked() else 1)
        self.status(f"Polygon {'removed' if self.poly_erase.isChecked() else 'added'}: {n} voxels")
        self._refresh_layers()
        self.viewer.refresh()

    def _mask_menu_actions(self, menu):
        ops = [("Fill holes (each slice)", lambda: self._mask_op(lambda l: seg.fill_holes(l, True), "Fill holes")),
               ("Fill 3D cavities", lambda: self._mask_op(lambda l: seg.fill_holes(l, False), "Fill cavities")),
               ("Keep largest part", lambda: self._mask_op(seg.keep_largest, "Keep largest part")),
               ("Fill between slices (interpolate)",
                lambda: self._mask_op(lambda l: seg.interpolate_slices(l, self.session.orient.spacing_xyz),
                                      "Fill between slices")),
               ("Clear active mask", lambda: self._mask_op(seg.clear_mask, "Clear")),
               ("Duplicate layer", self.duplicate_layer),
               ("Union with layer…", lambda: self._mask_combine("union")),
               ("Intersect with layer…", lambda: self._mask_combine("intersect")),
               ("Subtract layer…", lambda: self._mask_combine("subtract"))]
        for text, fn in ops:
            a = menu.addAction(text)
            a.triggered.connect(lambda _=False, f=fn: f())

    def _mask_op(self, fn, text):
        self._layer_edit(lambda: self.session.edit_active(fn), text)

    def _mask_combine(self, op):
        s = self.session
        if not (s and s.layer()):
            self.status("Select a mask layer first.")
            return
        others = [n for n in s.layers if n != s.active]
        if not others:
            self.status("Create or load a second layer first.")
            return
        name, ok = QtWidgets.QInputDialog.getItem(self, APP_NAME, f"{op.capitalize()} the active layer with:", others, 0, False)
        if ok:
            self._layer_edit(lambda: s.combine_with(name, op), f"{op.capitalize()} with {name}")

    def duplicate_layer(self):
        s = self.session
        if not (s and s.layer()):
            self.status("Select a mask layer first.")
            return
        new = s.duplicate_layer()
        self._refresh_layers()
        self.viewer.refresh()
        self.status(f"Duplicated as '{new}' (unsaved)")

    def on_paint(self, view, r, c, start, finish):
        s = self.session
        if not s or not s.layer():
            if start:
                self.status("Select or create a mask layer first (Segment tab).")
            return
        idx_view = self.viewer.views[view].index
        layer = s.layer()
        if layer.locked:
            return
        if start:
            ax = VIEW_AXIS[view]
            sp = s.orient.spacing_zyx[ax]
            half = int(np.ceil(self.brush.value() / sp)) if self.sphere.isChecked() else 0
            box = [slice(None)] * 3
            box[ax] = slice(max(0, idx_view - half), idx_view + half + 1)
            layer.begin_group(tuple(box))
        low = self.limit_val.value() if self.limit_on.isChecked() else None
        high = self.limit_hi.value() if self.limit_hi_on.isChecked() else None
        try:
            s.paint(view, idx_view, r, c, self.brush.value(), 0 if self.tool == "erase" else 1,
                    self.sphere.isChecked(), low, high)
        except Exception as e:  # noqa: BLE001
            self.status(str(e))
        if finish:
            layer.end_group()
            self._refresh_layers()
        self.viewer.refresh()

    # ================================================================== project / case
    def status(self, text):
        self.statusBar().showMessage(text, 8000)
        self.logbox.appendPlainText(text)

    def open_folder_dialog(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Open project folder", self.settings.value("last_dir", ""))
        if d:
            self.open_folder(d)

    def open_folder(self, folder):
        try:
            self.project = open_project(folder)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, APP_NAME, f"Could not open the folder: {e}")
            return
        self.settings.setValue("last_dir", str(folder))
        self.readers_edit.setText(", ".join(self.project.readers))
        self.structure_edit.setText(self.project.structure)
        self._fill_project()
        self.status(f"Project {folder}: {len(self.project.cases)} cases ({self.project.layout} layout)")

    def rescan(self):
        if self.project:
            self.project = self.project.rescan()
            self._fill_project()

    def save_project(self):
        if not self.project:
            return
        self.project.readers = [r.strip() for r in self.readers_edit.text().split(",") if r.strip()]
        self.project.structure = self.structure_edit.text().strip() or "lesions"
        f = self.project.save()
        self._fill_project()
        self.status(f"Project settings saved in {f}")

    def _fill_project(self):
        p = self.project
        self.case_tree.clear()
        for c in p.cases:
            readers = [r for r in p.readers if p.path(p.reader_mask_rel(c, r)).exists()]
            it = QtWidgets.QTreeWidgetItem([c.id, ", ".join(c.images), f"{len(c.masks)}"
                                            + (f" ({', '.join(readers)})" if readers else "")])
            self.case_tree.addTopLevelItem(it)
        for i in range(3):
            self.case_tree.resizeColumnToContents(i)
        self.proj_label.setText(f"{p.root}\n{len(p.cases)} cases · layout: {p.layout} · readers: {', '.join(p.readers)}")
        self.reader_box.clear()
        self.reader_box.addItems(p.readers)
        names = sorted({n for c in p.cases for n in c.masks})
        for combo in (self.pa, self.pb, self.r_which):
            combo.clear()
            combo.addItems([*p.readers, *names])
        if len(p.readers) > 1:
            self.pb.setCurrentIndex(1)

    def _confirm_unsaved(self) -> bool:
        if not self.session or not self.session.unsaved():
            return True
        r = QtWidgets.QMessageBox.question(
            self, APP_NAME, "Unsaved changes in: " + ", ".join(self.session.unsaved()) + ".\nSave them now?",
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel)
        if r == QtWidgets.QMessageBox.Cancel:
            return False
        if r == QtWidgets.QMessageBox.Save:
            for n in self.session.unsaved():
                if not self._save_layer(n):
                    return False
        return True

    def open_case(self, cid, ask=True):
        if not self.project or (ask and not self._confirm_unsaved()):
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            reader = self.reader_box.currentText()
            self.session = Session.open_case(self.project, self.project.case(cid), reader, self.blind.isChecked())
        except Exception as e:  # noqa: BLE001
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(self, APP_NAME, f"Could not open {cid}: {e}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self._after_open(cid)

    def open_images_dialog(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Open images (PET, CT, MRI)",
                                                          self.settings.value("last_dir", ""),
                                                          "Images (*.nii *.nii.gz *.nrrd *.mha *.mhd)")
        if files and self._confirm_unsaved():
            self.session = Session.open_files(files)
            self._after_open(Path(files[0]).name)

    def open_dicom_dialog(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open DICOM folder (PET/CT, SPECT, MRI)",
                                                            self.settings.value("last_dir", ""))
        if folder:
            self.open_dicom(folder)

    def open_dicom(self, folder, choose=None):
        """All image series under a folder (also several series in one folder, or one sub-folder each); with more
        than one series a list asks which to open (PET/SPECT and CT ticked). `choose` replaces the list (tests)."""
        from rsw.suv import list_series
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            series = list_series(Path(folder))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if not series:
            QtWidgets.QMessageBox.warning(self, "DICOM", f"No DICOM image series found in\n{folder}")
            return
        if choose is not None:
            picked = choose(series)
        elif len(series) == 1:
            picked = series
        else:
            picked = self._choose_series(series)
        if not picked or not self._confirm_unsaved():
            return
        vols, names = [], set()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for e in picked:
                if e.get("planar"):
                    vols += io.load_planar(e["files"])
                    continue
                name = {"PT": "PET", "NM": "SPECT"}.get(e["modality"], e["modality"])
                while name in names:
                    name += "_2"
                names.add(name)
                vols.append(io.load_volume(Path(folder), name=name, files=e["files"]))
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "DICOM", f"Could not read the series:\n{exc}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.settings.setValue("last_dir", str(folder))
        self.session = Session(None, None, vols)
        self._after_open(Path(folder).name)
        if all(v.info.get("planar") for v in vols):                # planar ANT/POST: one large coronal view
            self.viewer.set_layout("Coronal")

    def _choose_series(self, series):
        d = QtWidgets.QDialog(self)
        d.setWindowTitle("Choose the series to open")
        lay = QtWidgets.QVBoxLayout(d)
        lst = QtWidgets.QListWidget()
        first = {}
        for e in series:
            it = QtWidgets.QListWidgetItem(f"{e['modality']}  {e['description'] or '(no description)'}  – {e['n']} files")
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            tick = e["modality"] in ("PT", "NM", "CT") and e["modality"] not in first and e["n"] > 1
            first.setdefault(e["modality"], e["uid"]) if tick else None
            it.setCheckState(QtCore.Qt.Checked if tick else QtCore.Qt.Unchecked)
            lst.addItem(it)
        lay.addWidget(lst)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(d.accept)
        bb.rejected.connect(d.reject)
        lay.addWidget(bb)
        d.resize(520, 320)
        if d.exec() != QtWidgets.QDialog.Accepted:
            return []
        return [e for i, e in enumerate(series) if lst.item(i).checkState() == QtCore.Qt.Checked]

    def _after_open(self, title):
        s = self.session
        self.cursor = tuple(n // 2 for n in s.shape)
        self.win = {"PET": [0.0, self.pet_max.value()], "CT": [self.ct_lo.value(), self.ct_hi.value()],
                    "MR": None, "SPECT": None, "other": None}
        self.viewer.set_session(s)
        self._brush_changed(self.brush.value())
        self.set_cursor(self.cursor)
        self._refresh_layers()
        self._refresh_rois()
        self.label_selected = None
        self._fill_label_list()
        self._fill_display_boxes()
        self.setWindowTitle(f"{APP_NAME} – {title}")
        self.status(f"Opened {title}: " + ", ".join(f"{n} ({v.kind})" for n, v in s.volumes.items())
                    + (f"; masks: {', '.join(s.layers)}" if s.layers else ""))

    # ================================================================== layers and tools
    def _refresh_layers(self):
        s = self.session
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        for name, l in (s.layers.items() if s else []):
            it = QtWidgets.QListWidgetItem(("● " if not l.saved else "") + name)
            it.setData(QtCore.Qt.UserRole, name)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if l.visible else QtCore.Qt.Unchecked)
            it.setForeground(QtGui.QColor(*l.color))
            self.layer_list.addItem(it)
            if name == s.active:
                self.layer_list.setCurrentItem(it)
        self.layer_list.blockSignals(False)
        for combo in (self.ag_a, self.ag_b):
            cur = combo.currentText()
            combo.clear()
            combo.addItems(list(s.layers) if s else [])
            if cur:
                combo.setCurrentText(cur)
        if s and len(s.layers) > 1 and self.ag_a.currentText() == self.ag_b.currentText():
            self.ag_b.setCurrentIndex(1)

    def _layer_visibility(self, item):
        name = item.data(QtCore.Qt.UserRole)
        if self.session and name in self.session.layers:
            self.session.layers[name].visible = item.checkState() == QtCore.Qt.Checked
            self.viewer.refresh()

    def _layer_selected(self, _text):
        it = self.layer_list.currentItem()
        if it and self.session:
            self.session.active = it.data(QtCore.Qt.UserRole)
            self.viewer.refresh()

    # ------------------------------------------------------------------ labels on the image
    show_names = False
    label_selected = None

    def _labels_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        help_ = QtWidgets.QLabel("Label tool (key T): click on any view to add a text with an arrow pointing at the "
                                 "point. Drag the text or the arrow tip to move them; click a label to select it, "
                                 "double-click to edit it. Labels are exported by File > Save view as image.")
        help_.setWordWrap(True)
        help_.setStyleSheet("color: gray; font-size: 9pt;")
        lay.addWidget(help_)
        lay.addLayout(self._tool_row(("navigate", "label")))
        self.label_list = QtWidgets.QListWidget()
        self.label_list.currentRowChanged.connect(lambda i: self.select_label(i if i >= 0 else None, jump=True))
        self.label_list.itemDoubleClicked.connect(lambda _it: self.edit_label())
        lay.addWidget(self.label_list)
        row = QtWidgets.QHBoxLayout()
        for text, fn in (("Edit text…", self.edit_label), ("Delete", self.delete_label), ("Delete all", self.remove_all_labels)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        box = QtWidgets.QGroupBox("Style (selected label, and new labels)")
        f = QtWidgets.QFormLayout(box)
        self.label_arrow = QtWidgets.QCheckBox("Arrow to the point")
        self.label_arrow.setChecked(True)
        self.label_size = QtWidgets.QSpinBox()
        self.label_size.setRange(7, 40)
        self.label_size.setValue(12)
        self.label_color = QtWidgets.QPushButton()
        self._label_rgb = (255, 220, 0)
        self._paint_label_color()
        self.label_color.clicked.connect(self._choose_label_color)
        self.label_arrow.toggled.connect(lambda _v: self._style_selected())
        self.label_size.valueChanged.connect(lambda _v: self._style_selected())
        f.addRow(self.label_arrow)
        f.addRow("Text size", self.label_size)
        f.addRow("Colour", self.label_color)
        lay.addWidget(box)
        self.names_check = QtWidgets.QCheckBox("Show mask and ROI names on the image")
        self.names_check.toggled.connect(self.toggle_names)
        lay.addWidget(self.names_check)
        lay.addStretch(1)
        return w

    def _paint_label_color(self):
        self.label_color.setStyleSheet("background-color: rgb(%d,%d,%d)" % self._label_rgb)

    def _choose_label_color(self):
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(*self._label_rgb), self, "Label colour")
        if c.isValid():
            self._label_rgb = (c.red(), c.green(), c.blue())
            self._paint_label_color()
            self._style_selected()

    def _style_selected(self):
        a = self._selected_label()
        if a is not None:
            a.update(arrow=self.label_arrow.isChecked(), size=self.label_size.value(), color=self._label_rgb)
            self.viewer.refresh()

    def _selected_label(self):
        s = self.session
        i = self.label_selected
        return s.annotations[i] if s and i is not None and 0 <= i < len(s.annotations) else None

    def _fill_label_list(self):
        if not hasattr(self, "label_list"):
            return
        self.label_list.blockSignals(True)
        self.label_list.clear()
        for a in (self.session.annotations if self.session else []):
            self.label_list.addItem(f"{a['text']}   [{a['view']} {a['index'] + 1}]")
        if self.label_selected is not None:
            self.label_list.setCurrentRow(self.label_selected)
        self.label_list.blockSignals(False)

    def add_label(self, view, idx_view, r, c):
        """Label tool: click a point, type the text; the arrow points at the clicked point."""
        text, ok = QtWidgets.QInputDialog.getText(self, "Label", "Text of the label (the arrow points at the clicked point):")
        if not (ok and text.strip()):
            return
        sv = self.viewer.views[view]
        rh, cw = sv.pixel
        d = 12.0 * max(1.0, max(cw, rh) / 2)                       # text 12 mm from the point (more on coarse grids),
        from rsw.orient import view_slice                          # towards the middle so it stays inside the image
        nr, nc = view_slice(self.session.can[next(iter(self.session.can))], view, idx_view).shape
        dx, dy = (d if c < nc / 2 else -d), (d if r < nr / 2 else -d)
        self.session.annotations.append({"view": view, "index": int(idx_view), "x": (c + 0.5) * cw, "y": (r + 0.5) * rh,
                                         "dx": dx, "dy": dy, "text": text.strip(), "arrow": self.label_arrow.isChecked(),
                                         "size": self.label_size.value(), "color": self._label_rgb})
        self.label_selected = len(self.session.annotations) - 1
        self._fill_label_list()
        self.viewer.refresh()
        self.status("Label added: drag its text or arrow tip to move it; File > Save view as image exports it")

    def select_label(self, i, jump=False, edit=False):
        self.label_selected = i
        a = self._selected_label()
        if a is not None:
            for wdg, val in ((self.label_arrow, a["arrow"]), (self.label_size, a.get("size", 12))):
                wdg.blockSignals(True)
                (wdg.setChecked if wdg is self.label_arrow else wdg.setValue)(val)
                wdg.blockSignals(False)
            self._label_rgb = tuple(a["color"])
            self._paint_label_color()
            if jump and self.viewer.views[a["view"]].index != a["index"]:
                self.viewer.views[a["view"]].index = a["index"]
        self._fill_label_list()
        self.viewer.refresh()
        if edit:
            self.edit_label()

    def move_label(self, sliceview, i, part, x, y):
        a = self.session.annotations[i]
        if part == "text":
            a["dx"], a["dy"] = x - a["x"], y - a["y"]
        else:                                                      # tip: the text stays where it is
            tx, ty = a["x"] + a["dx"], a["y"] + a["dy"]
            a["x"], a["y"], a["dx"], a["dy"] = x, y, tx - x, ty - y
        self.label_selected = i
        sliceview.draw_labels()

    def edit_label(self):
        a = self._selected_label()
        if a is None:
            return
        text, ok = QtWidgets.QInputDialog.getText(self, "Label", "Text of the label:", text=a["text"])
        if ok and text.strip():
            a["text"] = text.strip()
            self._fill_label_list()
            self.viewer.refresh()

    def delete_label(self):
        if self._selected_label() is not None:
            self.session.annotations.pop(self.label_selected)
            self.label_selected = None
            self._fill_label_list()
            self.viewer.refresh()

    def remove_all_labels(self):
        if self.session and self.session.annotations:
            self.session.annotations.clear()
            self.label_selected = None
            self._fill_label_list()
            self.viewer.refresh()

    def toggle_names(self, on=None):
        self.show_names = bool(self.names_check.isChecked() if on is None else on)
        self.viewer.refresh()

    def set_tool(self, key):
        self.tool = key
        for k, buttons in self.tool_buttons.items():
            for b in buttons:
                b.setChecked(k == key)
        if key not in ("poly", "roi"):
            self.poly = None
            self.viewer.refresh()
        if self.ring_radius() is None:
            for v in self.viewer.all():
                v.ring.setVisible(False)
        self.status(f"Tool: {key}" + (" – click on a PET lesion to outline it automatically" if key == "grow" else ""))

    def remove_active_layer(self):
        s = self.session
        if not (s and s.active):
            self.status("No mask layer selected.")
            return
        name = s.active
        layer = s.layers[name]
        text = f"Remove the layer '{name}' from this view?"
        text += ("\n\nIt has UNSAVED changes that will be lost." if not layer.saved else
                 "\n\nThe file saved on disk is not deleted; reopen the case to load it again.")
        if QtWidgets.QMessageBox.question(self, APP_NAME, text) != QtWidgets.QMessageBox.Yes:
            return
        s.remove_layer(name)
        self._refresh_layers()
        self.viewer.refresh()
        self.status(f"Layer '{name}' removed from the view.")

    def new_layer(self):
        if not self.session:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, APP_NAME, "Name of the new mask layer:", text="mask")
        if ok and name:
            self.session.add_empty_layer(name)
            self.session.active = name
            self._refresh_layers()

    def load_mask_dialog(self):
        if not self.session:
            return
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load mask", self.settings.value("last_dir", ""),
                                                     "Masks (*.nii *.nii.gz *.nrrd *.mha)")
        if f:
            name = Path(f).name.split(".")[0]
            self.session.add_layer_from_file(name, f)
            self.session.active = name
            self._refresh_layers()
            self.viewer.refresh()

    def start_reader(self):
        s = self.session
        if not s or not self.project:
            QtWidgets.QMessageBox.information(self, APP_NAME, "Open a case of a project first.")
            return
        s.blind = self.blind.isChecked()
        reader = self.reader_box.currentText()
        name, source = s.start_reader(reader)
        msg = {"continue": "continuing this reader's saved mask", "auto": "starting from the automatic mask",
               "empty": "no automatic mask found: starting from an empty mask"}[source]
        self._refresh_layers()
        self.viewer.refresh()
        self.set_tool("paint")
        self.status(f"{reader}: {msg} (layer {name})" + (" · other readers hidden" if s.blind else ""))

    def grow_at(self, view, index, r, c):
        s = self.session
        if not s.layer():
            self.status("Select or create a mask layer first.")
            return
        mode = "percent" if self.grow_mode.currentIndex() == 0 else "absolute"
        try:
            n, thr = s.grow(view, index, r, c, mode, self.grow_val.value())
            self.status(f"Grow: {n} voxels added (threshold {thr:.2f})")
        except RegionTooLarge as e:
            self.status(f"Grow refused: {e}")
        self._refresh_layers()
        self.viewer.refresh()

    def remove_islands(self):
        from rsw.seg import remove_islands
        if self.session and self.session.layer():
            n = remove_islands(self.session.layer(), self.island_n.value())
            self.status(f"{n} islands removed")
            self._refresh_layers()
            self.viewer.refresh()

    def _layer_edit(self, fn, text):
        if not (self.session and self.session.layer()):
            self.status("Select or create a mask layer first (Segment tab).")
            return
        if self.session.layer().locked:
            self.status("The active layer is locked.")
            return
        n = fn()
        self.status(f"{text}: {n} voxels changed (Undo restores)")
        self._refresh_layers()
        self.viewer.refresh()

    def _thr_mode_changed(self):
        rel = self.thr_mode.currentData() == "refine_rel"
        self.thr_val.setSuffix(" %" if rel else "")
        self.thr_val.setValue(41.0 if rel else 2.5)

    def threshold_mask(self):
        mode, val = self.thr_mode.currentData(), self.thr_val.value()
        if mode == "refine_rel":
            self._layer_edit(lambda: self.session.threshold(mode, percent=val), f"Threshold {val:.0f} % of lesion max")
        else:
            self._layer_edit(lambda: self.session.threshold(mode, low=val), f"Threshold ≥ {val:g}")

    def smooth_mask(self):
        self._layer_edit(lambda: self.session.smooth(self.smooth_mm.value()), "Smooth outline")

    def margin_mask(self):
        self._layer_edit(lambda: self.session.margin(self.margin_mm.value()), f"Margin {self.margin_mm.value():+.1f} mm")

    def undo(self):
        if self.session and self.session.layer() and self.session.layer().undo():
            self._refresh_layers()
            self.viewer.refresh()

    def redo(self):
        if self.session and self.session.layer() and self.session.layer().redo():
            self._refresh_layers()
            self.viewer.refresh()

    def list_lesions(self):
        if self.session and self.session.layer():
            les = self.session.lesions()
            df = pd.DataFrame(les)
            if not df.empty:
                df = df.sort_values("volume_ml", ascending=False)
            self.lesion_model.set(df)
            self.status(f"{len(les)} lesions in {self.session.active}")

    def _jump_lesion(self, index):
        row = self.lesion_model.df.iloc[index.row()]
        self.set_cursor(row["center_zyx"], jump=True)

    # ================================================================== registration
    def open_registration(self):
        if not self.session or len(self.session.native) < 2:
            self.status("Open a case with at least two images to register.")
            return None
        from rsw.gui.registration_dialog import RegistrationDialog
        old = getattr(self, "reg_dialog", None)
        if old is not None and old.isVisible():
            old.reject()                                   # discard its pending preview before a new dialog starts
        self.reg_dialog = RegistrationDialog(self)
        guard_wheel(self.reg_dialog)
        self.reg_dialog.show()
        return self.reg_dialog

    def change_reference(self, name: str) -> bool:
        """Show everything on the grid of image ``name`` (the fixed image of a registration). Asks first when mask
        layers exist, because they are resampled and their undo history is cleared."""
        s = self.session
        if name == s.ref.name:
            return True
        if s.layers and QtWidgets.QMessageBox.question(
                self, APP_NAME, f"The viewer will switch to the grid of '{name}'. Mask layers are resampled to it "
                "(nearest neighbour), their undo history is cleared and they count as unsaved. Continue?") \
                != QtWidgets.QMessageBox.Yes:
            return False
        s.set_reference(name)
        self.cursor = tuple(n // 2 for n in s.shape)
        self.viewer.set_session(s)
        self.set_cursor(self.cursor)
        self._refresh_layers()
        self._refresh_rois()
        self._fill_display_boxes()
        self.status(f"Reference image is now {name}")
        return True

    def load_transform_dialog(self):
        if not self.session or len(self.session.native) < 2:
            self.status("Open a case with at least two images first.")
            return
        from rsw import registration as rg
        names = [n for n in self.session.native if n != self.session.ref.name]
        name, ok = QtWidgets.QInputDialog.getItem(self, APP_NAME, "Apply the transform to image:", names, 0, False)
        if not ok:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Transform file", "", "ITK transform (*.tfm *.h5 *.txt)")
        if path:
            self.session.set_transform(name, rg.load_transform(path))
            self.viewer.refresh()
            self.status(f"Applied {path} to {name} (transform must map {self.session.ref.name} space to {name} space).")

    def clear_registration(self):
        if not self.session or not self.session.transforms:
            self.status("No image is registered.")
            return
        name, ok = QtWidgets.QInputDialog.getItem(self, APP_NAME, "Remove registration of:", list(self.session.transforms),
                                                  0, False)
        if ok:
            self.session.clear_transform(name)
            self.viewer.refresh()
            self.status(f"Registration of {name} removed.")

    # ================================================================== saving
    def _save_layer(self, name, path=None) -> bool:
        layer = self.session.layers[name]
        if not (path or layer.path):
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save mask", f"{name}.nii.gz", "NIfTI (*.nii.gz)")
            if not path:
                return False
        res = self.session.save_layer(name, path)
        self.status(f"Saved {res['path']} · {res['voxels']} voxels · {res['volume_ml']:.1f} mL"
                    + (f" · previous version in backup/" if res["backup"] else ""))
        self._refresh_layers()
        if self.project:
            self.project = self.project.rescan()
            self._fill_project()
        return True

    def save_active(self):
        if self.session and self.session.active:
            self._save_layer(self.session.active)

    def save_active_as(self):
        if self.session and self.session.active:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save mask as", f"{self.session.active}.nii.gz",
                                                            "NIfTI (*.nii.gz)")
            if path:
                self._save_layer(self.session.active, path)

    def export_images(self):
        """DICOM (or any loaded image) -> NIfTI: PET in SUV, CT in HU; own grid or the PET grid."""
        if not self.session:
            self.status("Open a case first.")
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder for the NIfTI images")
        if not folder:
            return
        same = len(self.session.native) > 1 and QtWidgets.QMessageBox.question(
            self, APP_NAME, "Save all images on the grid of the reference image (PET), as shown in the viewer?\n\n"
            "Yes: same size and voxels as PET (fused).\nNo: every image keeps its own resolution.") == QtWidgets.QMessageBox.Yes
        files = self.session.export_images(folder, on_reference_grid=same)
        self.status(f"Exported {len(files)} image(s) to {folder}: " + ", ".join(p.name for p in files))

    def export_table(self, df, what, with_settings=False):
        if df is None or df.empty:
            self.status("Nothing to export.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, f"Export {what}", f"{what}.csv", "CSV (*.csv)")
        if path:
            if with_settings:
                save_table(df, self.radiomics_settings(), path)
            else:
                df.to_csv(path, index=False)
            self.status(f"Exported {path}")

    @staticmethod
    def render_png(widget, path, scale: float = 3.0, dpi: int = 300) -> tuple:
        """Draw a widget into a PNG at `scale` times its screen size (text, lines and labels are drawn again at the
        higher resolution, not enlarged pixels), marked with `dpi` for print."""
        size = widget.size() * scale
        img = QtGui.QImage(size, QtGui.QImage.Format_ARGB32)
        img.fill(QtGui.QColor("white"))
        img.setDotsPerMeterX(round(dpi / 0.0254))
        img.setDotsPerMeterY(round(dpi / 0.0254))
        p = QtGui.QPainter(img)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing | QtGui.QPainter.SmoothPixmapTransform)
        p.scale(scale, scale)
        widget.render(p, QtCore.QPoint(0, 0))
        p.end()
        img.save(str(path))
        return img.width(), img.height()

    def save_screenshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save view", "view.png", "PNG (*.png)")
        if path:
            w, h = self.render_png(self.viewer, path)
            self.status(f"Saved {path} ({w} × {h} pixels, 300 dpi)")

    def save_window_image(self, window=None):
        """A program window (the main window, or with F12 any open dialog) at three times the screen resolution."""
        window = window or self
        path, _ = QtWidgets.QFileDialog.getSaveFileName(window, "Save window as image", "window.png", "PNG (*.png)")
        if path:
            w, h = self.render_png(window, path)
            self.status(f"Saved {path} ({w} × {h} pixels, 300 dpi)")

    def eventFilter(self, obj, ev):
        if (ev.type() == QtCore.QEvent.KeyPress and ev.key() == QtCore.Qt.Key_F12
                and isinstance(obj, QtWidgets.QWidget) and not getattr(self, "_saving_window", False)):
            self._saving_window = True
            try:
                self.save_window_image(obj.window())
            finally:
                self._saving_window = False
            return True
        return super().eventFilter(obj, ev)

    # ================================================================== radiomics / agreement
    def radiomics_settings(self) -> Settings:
        return Settings(
            discretisation="width" if self.r_disc.currentIndex() == 0 else "count",
            bin_value=self.r_bin.value(), resample_mm=self.r_res.value() or None,
            resegment_low=self.r_reseg.value() if self.r_reseg_on.isChecked() else None,
            resegment_high=self.r_reseg_hi.value() if self.r_reseg_hi_on.isChecked() else None,
            resampling="ibsi" if self.r_method.currentIndex() == 1 else "pyradiomics",
            interpolator=["sitkBSpline", "sitkLinear", "sitkNearestNeighbor"][self.r_interp.currentIndex()],
            round_intensities=self.r_round.isChecked(),
            outlier_sigma=3.0 if self.r_outlier.isChecked() else None,
            classes=[c for c, cb in self.r_classes.items() if cb.isChecked()],
            image_types=["Original"] + (["LoG"] if self.r_log.isChecked() else []) + (["Wavelet"] if self.r_wav.isChecked() else []),
            min_texture_voxels=self.r_min_tex.value(), per_lesion=self.r_per_lesion.isChecked())

    def _run(self, fn, done, label):
        job = Job(fn, self)
        self.jobs.append(job)
        self.progress.setValue(0)
        self.progress.show()
        job.sig.progress.connect(lambda i, n, t: (self.progress.setMaximum(max(n, 1)), self.progress.setValue(i),
                                                  self.statusBar().showMessage(f"{label}: {t}")))
        job.sig.done.connect(lambda res: (self.progress.hide(), done(res)))
        job.sig.failed.connect(lambda tb: (self.progress.hide(), self.logbox.appendPlainText(tb),
                                           QtWidgets.QMessageBox.warning(self, APP_NAME, f"{label} failed:\n"
                                                                         + tb.strip().splitlines()[-1])))
        job.finished.connect(lambda: self.jobs.remove(job) if job in self.jobs else None)
        job.start()
        return job

    def run_radiomics(self):
        st = self.radiomics_settings()
        if self.r_scope.currentIndex() == 0:
            s = self.session
            if not s or not s.layer():
                self.status("Open a case and select a mask first.")
                return
            img = s.volumes[s.pet_name or s.ref.name]
            name = s.active
            base = s._base[name].copy()
            if st.discretisation == "width" and any(c in st.classes for c in TEXTURE):
                v = img.array[base > 0]
                if st.resegment_low is not None:
                    v = v[v >= st.resegment_low]
                nb = (float(v.max()) - float(v.min())) / st.bin_value if v.size else 0
                if v.size and not 8 <= nb <= 256 and QtWidgets.QMessageBox.question(
                        self, APP_NAME, f"With bin width {st.bin_value:g} {img.unit} the mask of {img.name} spans about "
                        f"{nb:.0f} grey levels (IBSI practice: roughly 30-130). Texture features would be unstable.\n\n"
                        "Continue anyway?") != QtWidgets.QMessageBox.Yes:
                    return None
            cid = s.case.id if s.case else img.name
            return self._run(lambda prog: extract(img.array, s.geo, base, cid, name, st),
                             self._show_features, "Radiomics")
        if not self.project:
            self.status("Open a project first.")
            return
        which = self.r_which.currentText()
        return self._run(lambda prog: analysis.radiomics_project(self.project, which, st, prog),
                         self._show_features, "Radiomics")

    def _show_features(self, df):
        self.feat_model.set(df)
        n = len([c for c in df.columns if c.startswith("original_") or c.startswith(("log", "wavelet"))])
        self.status(f"Radiomics: {len(df)} rows × {n} features")

    def compare_layers(self):
        s = self.session
        a, b = self.ag_a.currentText(), self.ag_b.currentText()
        if not s or a not in s.layers or b not in s.layers:
            return
        r = metrics.agreement(s.layers[a].array, s.layers[b].array, s.orient.spacing_xyz)
        self.ag_label.setText(f"Dice {r['dice']:.3f} · Jaccard {r['jaccard']:.3f} · HD95 {r['hd95_mm']:.1f} mm\n"
                              f"Volume A {r['volume_a_ml']:.1f} mL · B {r['volume_b_ml']:.1f} mL "
                              f"({r['volume_diff_pct']:+.1f}%)")

    def project_agreement(self):
        if not self.project:
            return
        a, b = self.pa.currentText(), self.pb.currentText()
        self._run(lambda prog: analysis.agreement_project(self.project, a, b, prog),
                  lambda df: (self.ag_model.set(df), self.icc_label.setText(
                      f"{len(df)} cases · median Dice {df['dice'].median():.3f}" if not df.empty else "No pairs found")),
                  "Agreement")

    def project_icc(self):
        if not self.project:
            return
        a, b, thr, st = self.pa.currentText(), self.pb.currentText(), self.icc_thr.value(), self.radiomics_settings()
        self._run(lambda prog: analysis.icc_project(self.project, a, b, st, thr, prog),
                  lambda res: (self.ag_model.set(res[0]), self.icc_label.setText(analysis.summary_icc(res[0]))), "ICC")

    # ================================================================== close
    def closeEvent(self, ev):
        if self._confirm_unsaved():
            ev.accept()
        else:
            ev.ignore()
