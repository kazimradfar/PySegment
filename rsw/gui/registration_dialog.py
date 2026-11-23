"""Registration dialog: estimate a transform between two images of the open case, check it (checkerboard, numbers),
accept or reject it, save it. Only PySide6 (LGPL) and SimpleITK; saving a transform works like every other
save. The estimate never changes the native images; accepting only changes what the viewer shows and exports."""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from rsw import APP_NAME
from rsw import registration as rg
from rsw.io import is_counts
from rsw.gui.widgets import finish_scroll, scroll_content

METRICS = [("auto (by image types)", "auto"), ("Mattes mutual information", "mattes"),
           ("Mean squares (same activity scale)", "mean_squares"), ("Correlation (same CT/MR type)", "correlation")]
MODES = [("Rigid (6 parameters)", "rigid"), ("Affine (12 parameters)", "affine"), ("Deformable (B-spline)", "bspline")]


def checkerboard(fixed: np.ndarray, moving: np.ndarray, tiles: int = 8) -> np.ndarray:
    """Chessboard of two 2D slices, each scaled to 0..1 by its own 1-99.5 percentile, as uint8."""
    def norm(a):
        lo, hi = np.percentile(a, [1, 99.5])
        return np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
    f, m = norm(fixed), norm(moving)
    yy, xx = np.indices(f.shape)
    ty, tx = max(f.shape[0] // tiles, 1), max(f.shape[1] // tiles, 1)
    mask = ((yy // ty) + (xx // tx)) % 2 == 0
    return (np.where(mask, f, m) * 255).astype(np.uint8)


def fusion(fixed: np.ndarray, moving: np.ndarray, alpha: float = 0.75) -> np.ndarray:
    """Fixed slice in grey with the moving slice in orange on top (RGB uint8): easier to judge than a chessboard
    when the two images are of different kinds (CT and SPECT look nothing alike)."""
    def norm(a):
        lo, hi = np.percentile(a, [1, 99.5])
        return np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
    f, m = norm(fixed), norm(moving)
    w = alpha * m[..., None]
    rgb = f[..., None] * (1 - w) + w * np.array([1.0, 0.55, 0.0])
    return np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))


class RegistrationDialog(QtWidgets.QDialog):
    def __init__(self, main):
        super().__init__(main)
        self.main, self.s = main, main.session
        self.result: rg.RegResult | None = None
        self.previous = None            # transform of the run's image before the dialog changed it
        self.run_name = ""              # image the current result belongs to (fixed at run time)
        self._accepted = False
        self.pairs: list[dict] = []     # landmark pairs: fixed point (frame mm), moving point (native mm), validation flag
        self.pending_fixed = None
        self.pairs_ref = None           # reference (frame) the pair coordinates belong to
        self.snapshot = None            # (reference name, transforms) before this dialog changed the reference
        self.also_set: list[str] = []   # images that got the same transform (hybrid estimate), for Reject
        self.setWindowTitle(f"{APP_NAME} – image registration")
        names = list(self.s.native)
        content = scroll_content(self)
        lay = QtWidgets.QVBoxLayout(content)
        info = QtWidgets.QLabel("Fixed image = the one that stays (usually the CT). Moving image = the one that is moved onto "
                                "it (usually PET or SPECT). The viewer shows everything on the grid of the fixed image. "
                                "Images of one hybrid scanner (PET/CT, SPECT/CT) already share a frame; register them only "
                                "if the patient moved. Hybrid option: estimate on the two CTs and apply the result to the "
                                "PET/SPECT of the moving study.")
        info.setWordWrap(True)
        lay.addWidget(info)
        form = QtWidgets.QFormLayout()
        self.fixed = QtWidgets.QComboBox()
        self.fixed.addItems(names)
        default_fixed = next((n for n in names if self.s.native[n].kind == "CT"), self.s.ref.name)
        self.fixed.setCurrentText(default_fixed)
        self.moving = QtWidgets.QComboBox()
        self.fixed.currentTextChanged.connect(self._fill_moving)
        self.est_fixed = QtWidgets.QComboBox()
        self.est_fixed.addItems(["(same as fixed image)"] + names)
        self.moving_est = QtWidgets.QComboBox()
        self.moving_est.addItems(["(same as moving image)"] + names)
        self._fill_moving()
        self.mode = QtWidgets.QComboBox()
        for label, v in MODES:
            self.mode.addItem(label, v)
        self.metric_box = QtWidgets.QComboBox()
        for label, v in METRICS:
            self.metric_box.addItem(label, v)
        from rsw import registration_elastix as rx
        self.engine = QtWidgets.QComboBox()
        self.engine.addItem("SimpleITK (built in)", "simpleitk")
        self.engine.addItem("elastix" if rx.available() else "elastix (not installed: pip install itk-elastix)", "elastix")
        if not rx.available():
            self.engine.model().item(1).setEnabled(False)
        self.engine.setToolTip("elastix: stochastic optimisation, fast and accurate for deformable (B-spline) registration "
                               "of whole CTs; no landmark start")
        self.profile = QtWidgets.QComboBox()
        self.profile.addItem("General (by image types)", "general")
        self.profile.addItem("Lung motion: two CTs (4D CT phases, inspiration / expiration)", "lung")
        self.profile.setToolTip("Lung: 2 mm working grid, CT window -1000..200 HU, metric inside a lung mask, B-spline "
                                "every 10 mm (elastix only; validated on POPI 4D CT with expert landmarks)")
        self.engine.currentIndexChanged.connect(lambda *_: self.profile.setEnabled(self.engine.currentData() == "elastix"))
        self.profile.setEnabled(False)
        self.init = QtWidgets.QComboBox()
        self.init.addItem("Automatic (scanner coordinates if the images overlap, else image centres)", "auto")
        self.init.addItem("Image centres (geometry)", "geometry")
        self.init.addItem("Identity (images already roughly aligned)", "identity")
        self.init.addItem("Landmark pairs (set below, at least 3 fitting pairs)", "landmarks")
        self.levels = QtWidgets.QSpinBox()
        self.levels.setRange(1, 4)
        self.levels.setValue(3)
        self.iters = QtWidgets.QSpinBox()
        self.iters.setRange(10, 1000)
        self.iters.setValue(500)        # stops earlier when converged; small fields of view need 300-400
        form.addRow("Fixed image (stays)", self.fixed)
        form.addRow("Moving image (is moved)", self.moving)
        form.addRow("Estimate on fixed image", self.est_fixed)
        form.addRow("Estimate on moving image", self.moving_est)
        form.addRow("Engine", self.engine)
        form.addRow("Profile", self.profile)
        form.addRow("Transform", self.mode)
        form.addRow("Similarity metric", self.metric_box)
        form.addRow("Start", self.init)
        form.addRow("Resolution levels", self.levels)
        form.addRow("Iterations per level", self.iters)
        lay.addLayout(form)
        lm = QtWidgets.QGroupBox("Landmarks (optional): same anatomical point on both images")
        ll = QtWidgets.QGridLayout(lm)
        info2 = QtWidgets.QLabel("1) View fixed, move the cursor (Navigate tool) to a point, press Fixed point. "
                                 "2) View moving, put the cursor on the SAME anatomy, press Moving point. Repeat, "
                                 "spread widely (not on one line). Tick 'validation' for pairs that must NOT be used to fit "
                                 "but only to measure the error (TRE).")
        info2.setWordWrap(True)
        info2.setStyleSheet("color: gray; font-size: 9pt;")
        ll.addWidget(info2, 0, 0, 1, 4)
        self.lm_buttons = []
        for n_, (label, fn) in enumerate([("View fixed", lambda: self.view_lm("fixed")), ("View moving", lambda: self.view_lm("moving")),
                                          ("Fixed point", self.lm_fixed), ("Moving point", self.lm_moving),
                                          ("Remove last pair", self.lm_remove), ("Landmarks only", self.run_landmarks),
                                          ("Export pairs", self.lm_export)]):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(fn)
            b.setMinimumWidth(0)
            ll.addWidget(b, 1 + n_ // 4, n_ % 4)
            self.lm_buttons.append(b)
        self.lm_val = QtWidgets.QCheckBox("validation")
        ll.addWidget(self.lm_val, 3, 0)
        self.lm_text = QtWidgets.QLabel("No pairs yet.")
        self.lm_text.setWordWrap(True)
        ll.addWidget(self.lm_text, 3, 1, 1, 3)
        lay.addWidget(lm)
        row = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Run registration")
        self.run_btn.clicked.connect(lambda: self.run())
        self.accept_btn = QtWidgets.QPushButton("Accept (show and use in the viewer)")
        self.accept_btn.clicked.connect(self.accept_result)
        self.reject_btn = QtWidgets.QPushButton("Reject / undo")
        self.reject_btn.clicked.connect(self.reject_result)
        self.toggle_btn = QtWidgets.QPushButton("Before / after")
        self.toggle_btn.setToolTip("Switch the viewer between the moving image as it was and after registration; "
                                   "the cursor stays where it is, so shifts are easy to see.")
        self.toggle_btn.clicked.connect(self.toggle_before_after)
        self.save_btn = QtWidgets.QPushButton("Save transform…")
        self.save_btn.clicked.connect(self.save_result)
        for b in (self.run_btn, self.toggle_btn, self.accept_btn, self.reject_btn, self.save_btn):
            row.addWidget(b)
        lay.addLayout(row)
        srow = QtWidgets.QHBoxLayout()
        self.view_mode = QtWidgets.QComboBox()
        self.view_mode.addItem("Fusion (fixed grey, moving orange)", "fusion")
        self.view_mode.addItem("Checkerboard", "checker")
        self.view_mode.setToolTip("Fusion is easier for two different kinds of image (CT + SPECT/PET); the checkerboard "
                                  "for two images of the same kind: edges must continue across the squares.")
        self.view_mode.currentIndexChanged.connect(self._show_check)
        self.slice = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slice.valueChanged.connect(self._show_check)
        srow.addWidget(self.view_mode)
        srow.addWidget(self.slice, 1)
        lay.addLayout(srow)
        self.pic = QtWidgets.QLabel("Checkerboard of the fixed and the registered moving image appears here after a run.")
        self.pic.setAlignment(QtCore.Qt.AlignCenter)
        self.pic.setMinimumHeight(260)
        lay.addWidget(self.pic, 1)
        self.report = QtWidgets.QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(150)
        lay.addWidget(self.report)
        self._set_buttons()
        finish_scroll(self, content, 780, 700)

    # ------------------------------------------------------------------ helpers
    def _set_buttons(self):
        have = self.result is not None
        for b in (self.accept_btn, self.reject_btn, self.save_btn, self.toggle_btn):
            b.setEnabled(have)
        self.run_btn.setEnabled(self.moving.count() > 0)

    def _fill_moving(self, *_):
        names = [n for n in self.s.native if n != self.fixed.currentText()]
        self.moving.clear()
        self.moving.addItems(names)
        pref = next((n for n in names if self.s.native[n].kind in ("PET", "SPECT")), None)
        if pref:
            self.moving.setCurrentText(pref)

    def _est_fixed(self) -> str:
        t = self.est_fixed.currentText()
        return self.fixed.currentText() if t.startswith("(") else t

    def _moving_name(self) -> str:
        return self.moving.currentText()

    def _est_moving(self) -> str:
        t = self.moving_est.currentText()
        return self._moving_name() if t.startswith("(") else t


    # ------------------------------------------------------------------ landmarks
    def _cursor_mm(self):
        """Physical position (mm, LPS) of the viewer cursor on the reference grid."""
        s = self.s
        z, y, x = s.orient.to_original(self.main.cursor, s.ref.array.shape)
        return tuple(float(v) for v in s.geo.reference().TransformContinuousIndexToPhysicalPoint((float(x), float(y), float(z))))

    def _ensure_frame(self) -> bool:
        """Landmarks are recorded in the frame of the fixed image: make it the reference, forget pairs of another frame."""
        if self.fixed.currentText() != self.s.ref.name and not self._prepare_reference():
            return False
        if self.pairs_ref not in (None, self.s.ref.name):
            self.pairs, self.pending_fixed = [], None
            self._lm_summary()
        self.pairs_ref = self.s.ref.name
        return True

    def view_lm(self, which):
        if not self._ensure_frame():
            return
        name = self.fixed.currentText() if which == "fixed" else self._moving_name()
        if name:
            self.s.base_pick, self.s.over_pick = name, ""
            self.main._fill_display_boxes()
            self.main.viewer.refresh()

    def lm_fixed(self):
        if not self._ensure_frame():
            return
        self.pending_fixed = self._cursor_mm()
        self.lm_text.setText(f"Fixed point set at {np.round(self.pending_fixed, 1).tolist()} mm; now View moving and press Moving point.")

    def lm_moving(self):
        if self.pending_fixed is None:
            self.lm_text.setText("Set the Fixed point first.")
            return
        t = self.s.transforms.get(self._moving_name())          # frame -> native space of the moving image
        p = self._cursor_mm()
        m = tuple(t.TransformPoint(p)) if t is not None else p
        self.pairs.append({"fixed": self.pending_fixed, "moving": m, "validation": self.lm_val.isChecked()})
        self.pending_fixed = None
        self._lm_summary()

    def lm_remove(self):
        if self.pairs:
            self.pairs.pop()
        self.pending_fixed = None
        self._lm_summary()

    def _fit_pairs(self):
        return [q for q in self.pairs if not q["validation"]]

    def _lm_summary(self):
        fit, val = self._fit_pairs(), [q for q in self.pairs if q["validation"]]
        txt = f"{len(fit)} fitting pair(s), {len(val)} validation pair(s)."
        if len(fit) >= 3:
            try:
                t = rg.landmark_transform([q["fixed"] for q in fit], [q["moving"] for q in fit])
                txt += f" Rigid fit error (FRE) {self.lm_errors(t, fit)[0]:.2f} mm."
            except Exception as e:  # noqa: BLE001
                txt += f" ({e})"
        self.lm_text.setText(txt)

    @staticmethod
    def lm_errors(t, pairs):
        """Mean and maximum distance (mm) between the transformed fixed points and their moving partners."""
        d = [float(np.linalg.norm(np.array(t.TransformPoint(tuple(q["fixed"]))) - np.array(q["moving"]))) for q in pairs]
        return (float(np.mean(d)), float(np.max(d))) if d else (float("nan"), float("nan"))

    def lm_pairs_arrays(self):
        fit = self._fit_pairs()
        if len(fit) < 3:
            raise ValueError("at least three fitting landmark pairs are needed")
        f = np.array([q["fixed"] for q in fit], float)
        if np.linalg.matrix_rank(f - f.mean(axis=0), tol=5.0) < 2:      # within 5 mm of one line
            raise ValueError("the fixed points lie on one line; spread them over the body")
        return f.tolist(), [q["moving"] for q in fit]

    def run_landmarks(self):
        """Rigid transform from the landmark pairs alone (no image optimisation)."""
        if not self._moving_name():
            return None
        try:
            f, m = self.lm_pairs_arrays()
        except ValueError as e:
            self.report.setPlainText(str(e))
            return None
        if self.pairs_ref != self.s.ref.name or self.fixed.currentText() != self.s.ref.name:
            self.report.setPlainText("The pairs belong to another fixed image; press View fixed and set the pairs again.")
            return None
        res = rg.RegResult(rg.landmark_transform(f, m), "rigid", "landmarks", self.lm_errors(rg.landmark_transform(f, m), self._fit_pairs())[0], 0,
                           "landmark fit", [], None)
        self._done(res)
        return res

    def _prepare_reference(self):
        s = self.s
        if self.result is not None and not self._accepted:
            self._restore()
            self.result, self.run_name, self.previous = None, "", None
        if self.snapshot is None:
            self.snapshot = (s.ref.name, dict(s.transforms))
        ok = self.main.change_reference(self.fixed.currentText())
        self.previous = None
        return ok

    def lm_export(self):
        import pandas as pd
        if not self.pairs:
            self.lm_text.setText("No pairs to export.")
            return
        rows = []
        t = self.result.transform if self.result is not None else None
        for i, q in enumerate(self.pairs, 1):
            r = {"pair": i, "use": "validation" if q["validation"] else "fit"}
            r.update({f"fixed_{a}_mm": v for a, v in zip("xyz", q["fixed"])})
            r.update({f"moving_{a}_mm": v for a, v in zip("xyz", q["moving"])})
            r["error_mm_current_result"] = (float(np.linalg.norm(np.array(t.TransformPoint(tuple(q["fixed"]))) - np.array(q["moving"])))
                                            if t is not None else None)
            rows.append(r)
        self.main.export_table(pd.DataFrame(rows), "landmark_pairs")

    # ------------------------------------------------------------------ run
    def run(self, sync: bool = False):
        s = self.s
        if not self._moving_name():
            return None
        if self.result is not None and not self._accepted:          # a preview is pending: undo it in the old frame first
            self._restore()
            self.result, self.run_name, self.previous = None, "", None
        if self.fixed.currentText() != s.ref.name:
            if self.snapshot is None:                                 # remember the start state for Reject
                self.snapshot = (s.ref.name, dict(s.transforms))
            if not self.main.change_reference(self.fixed.currentText()):
                return None
            self.previous = None
        if self.init.currentData() == "landmarks" and self.pairs and self.pairs_ref != s.ref.name:
            self.report.setPlainText("The landmark pairs belong to another fixed image; set them again.")
            return None
        fixed = s.volumes[self._est_fixed()]                      # on the grid of the fixed image
        est = s.native[self._est_moving()]                       # own grid, never resampled before estimating
        if est.name == fixed.name:
            self.report.setPlainText("Fixed and moving image for the estimate are the same image.")
            return None
        kw = dict(mode=self.mode.currentData(), metric=self.metric_box.currentData(), init=self.init.currentData(),
                  levels=self.levels.value(), iterations=self.iters.value())
        if self.engine.currentData() == "elastix":
            if kw["init"] == "landmarks":
                self.report.setPlainText("A landmark start is available with the SimpleITK engine only.")
                return None
            kw.update(engine="elastix", profile=self.profile.currentData())
        if kw["init"] == "landmarks":
            try:
                kw["landmarks"] = self.lm_pairs_arrays()
            except ValueError as e:
                self.report.setPlainText(str(e))
                return None
        self.report.setPlainText("Running …")
        self.run_btn.setEnabled(False)
        if sync:
            res = rg.register(fixed, est, **kw)
            self._done(res)
            return res
        return self.main._run(lambda prog: rg.register(fixed, est, progress=lambda i, v: prog(i, kw["iterations"], f"metric {v:.4f}"),
                                                       **kw), self._done, "Registration")

    def _done(self, res: rg.RegResult):
        name = self._moving_name()
        if self.result is None or self.run_name != name:                   # first run for this image: remember what to restore
            if self.run_name and self.run_name != name:
                self._restore()
            self.previous = self.s.transforms.get(name)
        self.result, self.run_name = res, name
        same_kind = self.s.volumes[self.s.ref.name].kind == self.s.volumes[name].kind
        self.view_mode.blockSignals(True)
        self.view_mode.setCurrentIndex(1 if same_kind else 0)
        self.view_mode.blockSignals(False)
        self.s.set_transform(name, res.transform)          # preview
        est = self._est_moving()
        self.also_set = []
        if est != name and est != self.s.ref.name:         # hybrid: the study's other image moves with it
            self.also_prev = {est: self.s.transforms.get(est)}
            self.s.set_transform(est, res.transform)
            self.also_set = [est]
        self._showing_after = True
        self.main.show_pair(self.s.ref.name, name)          # fixed image as background, the moved image in colour
        self.main.opacity.setValue(max(self.main.opacity.value(), 55))
        self._centre_on(name)
        z = self.s.shape[0]
        self.slice.blockSignals(True)
        self.slice.setRange(0, z - 1)
        self.slice.setValue(z // 2)
        self.slice.blockSignals(False)
        self._show_check()
        v0, v1 = self.s.native[name], self.s.volumes[name]
        q = rg.quantification_check(v0, v1)
        lines = [f"Transform: {rg.describe(res.transform)}", f"Metric: {res.metric} = {res.metric_value:.4f}, "
                 f"{res.iterations} iterations, stop: {res.stop_reason}",
                 ("Total counts, before -> after:" if is_counts(v0.unit) else f"Total {v0.unit or 'value'} x volume, before -> after:") + f" {q['total_before']:.4g} -> {q['total_after']:.4g} "
                 f"({q['total_change_pct']:+.2f} %)"
                 + ("" if q.get("max_before") is None else f"; maximum {q['max_before']:.4g} -> {q['max_after']:.4g}")
                 + ". The total must stay (almost) the same; a loss means part of the image left the field of view."]
        if res.jacobian:
            lines.append(f"Jacobian determinant: min {res.jacobian['min']:.3f}, max {res.jacobian['max']:.3f}, "
                         f"{100 * res.jacobian['fraction_nonpositive']:.2f} % folded voxels")
        val = [q for q in self.pairs if q["validation"]]
        if val:
            mean, mx = self.lm_errors(res.transform, val)
            lines.append(f"Validation landmarks (not used for fitting), {len(val)} pairs: TRE mean {mean:.2f} mm, max {mx:.2f} mm")
        lines += [f"WARNING: {w}" for w in res.warnings]
        lines.append("Preview only: press Accept to keep it, Reject to go back. Always check the preview (fusion or checkerboard) by eye.")
        self.report.setPlainText("\n".join(lines))
        self._set_buttons()
        self.main.status("Registration finished: " + rg.describe(res.transform))

    def _centre_on(self, name):
        """Put the cursor in the middle of the moved image's content, so the slice shown contains it."""
        a = self.s.can[name]
        m = a > (a.min() + 0.25 * (a.max() - a.min()))
        if m.any():
            idx = np.round(np.argwhere(m).mean(axis=0)).astype(int)
            self.main.set_cursor(tuple(int(i) for i in idx))

    def toggle_before_after(self):
        if self.result is None:
            return
        name = self.run_name
        if getattr(self, "_showing_after", True):
            if self.previous is None:
                self.s.clear_transform(name)
            else:
                self.s.set_transform(name, self.previous)
            self._showing_after = False
            self.main.status(f"Showing {name} BEFORE registration")
        else:
            self.s.set_transform(name, self.result.transform)
            self._showing_after = True
            self.main.status(f"Showing {name} AFTER registration")
        self.main.viewer.refresh()
        self._show_check()

    def _show_check(self, *_):
        if self.result is None:
            return
        name, i = self.run_name, self.slice.value()
        can_f, can_m = self.s.can[self.s.ref.name], self.s.can[name]
        if self.view_mode.currentData() == "fusion":
            img = fusion(can_f[i], can_m[i])
            h, w = img.shape[:2]
            qimg = QtGui.QImage(img.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        else:
            img = np.ascontiguousarray(checkerboard(can_f[i], can_m[i]))
            h, w = img.shape
            qimg = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format_Grayscale8).copy()
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.pic.width(), 300, QtCore.Qt.KeepAspectRatio,
                                                   QtCore.Qt.SmoothTransformation)
        self.pic.setPixmap(pix)

    # ------------------------------------------------------------------ decisions
    def accept_result(self):
        if self.result is None:
            return
        self._accepted = True
        self.snapshot = None
        if not getattr(self, "_showing_after", True):
            self.s.set_transform(self.run_name, self.result.transform)
            self._showing_after = True
        self.main.status(f"Registration of {self.run_name} accepted ({rg.describe(self.result.transform)}). "
                         "Masks already drawn are not moved; draw or load masks now.")
        self.main.viewer.refresh()
        self.close()

    def _restore(self):
        self._showing_after = True
        name = self.run_name
        if name and name != self.s.ref.name:
            if self.previous is None:
                self.s.clear_transform(name)
            else:
                self.s.set_transform(name, self.previous)
        for e in self.also_set:
            if e != self.s.ref.name:
                old_t = getattr(self, "also_prev", {}).get(e)
                (self.s.clear_transform(e) if old_t is None else self.s.set_transform(e, old_t))
        self.also_set = []

    def _restore_reference(self):
        """Undo a change of the reference image made by this dialog (with all transforms as they were)."""
        if self.snapshot is None:
            return
        ref, tf = self.snapshot
        self.snapshot = None
        if self.s.ref.name != ref and self.main.change_reference(ref):
            pass
        if self.s.ref.name == ref:
            self.s.set_transforms(tf)

    def reject(self):
        """Closing with Esc or the window button without Accept discards the preview."""
        if self.result is not None and not self._accepted:
            self._restore()
        if not self._accepted:
            self._restore_reference()
        self.main.viewer.refresh()
        super().reject()

    def reject_result(self):
        self._restore()
        self._restore_reference()
        self.result = None
        self.main.viewer.refresh()
        self.report.setPlainText("Rejected; the previous view is restored.")
        self._set_buttons()

    def save_result(self, path: str | None = None):
        if self.result is None:
            return None
        if not path:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save transform", f"{self.run_name}_to_"
                                                            f"{self.s.ref.name}.tfm", "ITK transform (*.tfm)")
        if not path:
            return None
        out = rg.save_transform(self.result.transform, path)
        self.main.status(f"Saved transform {out}")
        return out
