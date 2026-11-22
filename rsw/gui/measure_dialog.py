"""Measurement window: table of the ROIs measured so far (mean, SD, max, peak, volume, total counts ...) with an
optional calibration that turns SPECT counts into activity (MBq) and concentration (MBq/mL). PET in SUV also gives the
activity concentration when dose and weight are known. Opens by itself when a ROI is placed."""
from __future__ import annotations

import pandas as pd
from PySide6 import QtWidgets

from rsw import APP_NAME, calibration, dosimetry, measure
from rsw.gui.widgets import PandasModel, finish_scroll, scroll_content, table_view


def _spin(lo, hi, val, dec=3, suffix=""):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setValue(val)
    s.setSuffix(suffix)
    s.setMinimumWidth(0)
    return s


class MeasureDialog(QtWidgets.QDialog):
    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.setWindowTitle(f"{APP_NAME} – measurements")
        content = scroll_content(self)
        lay = QtWidgets.QVBoxLayout(content)
        self.model = PandasModel()
        self.table = table_view(self.model)
        self.table.setMinimumHeight(130)
        lay.addWidget(self.table, 1)
        box = QtWidgets.QGroupBox("Counts to activity (SPECT in counts)")
        g = QtWidgets.QGridLayout(box)
        self.sens = _spin(0, 1e6, 0.0, 4, " cps/MBq")
        self.sens.setToolTip("System sensitivity of this SPECT (counts per second per MBq) for the isotope and window used, "
                             "from your own calibration (point or phantom source). 0 = no conversion.")
        self.time = _spin(0, 1e7, 0.0, 1, " s")
        self.time.setToolTip("Total acquisition time of the counts in the image (all projections together).")
        self.decay_on = QtWidgets.QCheckBox("Decay-correct")
        self.decay_on.setToolTip("Reports the activity at the reference moment. For REPORTING only: dosimetry needs the "
                                 "activity present at the scan, so the Dosimetry window always ignores this correction.")
        self.half = _spin(0.001, 1e6, 159.5, 2, " h")
        self.half.setToolTip("Half-life; 159.5 h = Lu-177. Tc-99m 6.01 h, I-131 192.5 h.")
        self.elapsed = _spin(-1e5, 1e5, 0.0, 2, " h")
        self.elapsed.setToolTip("Time from the reference moment (for example end of infusion) to the middle of the scan.")
        self.profile = QtWidgets.QComboBox()
        self.profile.setEditable(True)
        self.profile.setToolTip("Calibration profile of a radiopharmaceutical (isotope, energy window): type a name and press "
                                "Save profile. Profiles stay on this computer.")
        self.iso = QtWidgets.QComboBox()
        self.iso.addItems(list(dosimetry.ISOTOPES))
        self.iso.currentTextChanged.connect(lambda n: self.half.setValue(dosimetry.ISOTOPES[n]["half_life_h"]))
        sv, dl = QtWidgets.QPushButton("Save profile"), QtWidgets.QPushButton("Delete")
        sv.clicked.connect(self.save_profile)
        dl.clicked.connect(self.delete_profile)
        g.addWidget(QtWidgets.QLabel("Radiopharmaceutical"), 3, 0)
        g.addWidget(self.profile, 3, 1)
        g.addWidget(self.iso, 3, 2)
        g.addWidget(sv, 4, 1)
        g.addWidget(dl, 4, 2)
        g.addWidget(QtWidgets.QLabel("Sensitivity"), 0, 0)
        g.addWidget(self.sens, 0, 1)
        g.addWidget(QtWidgets.QLabel("Acquisition time"), 0, 2)
        g.addWidget(self.time, 0, 3)
        g.addWidget(self.decay_on, 1, 0)
        g.addWidget(self.half, 1, 1)
        g.addWidget(QtWidgets.QLabel("Elapsed"), 1, 2)
        g.addWidget(self.elapsed, 1, 3)
        self.hint = QtWidgets.QLabel("")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: gray; font-size: 9pt;")
        g.addWidget(self.hint, 2, 0, 1, 4)
        lay.addWidget(box)
        row = QtWidgets.QHBoxLayout()
        for label, fn in (("Mask → ROI", self.main.mask_to_roi), ("Refresh", self.refresh), ("Copy table", self.copy),
                          ("Export CSV", self.export)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        for w in (self.sens, self.time, self.half, self.elapsed):
            w.valueChanged.connect(self.refresh)
        self.decay_on.toggled.connect(self.refresh)
        self.profile.activated.connect(lambda i: self._profile_chosen(self.profile.itemText(i)))
        self.load_profiles()
        finish_scroll(self, content, 900, 520)

    def load_profiles(self):
        self.profile.blockSignals(True)
        self.profile.clear()
        self.profile.addItems(sorted(calibration.load()))
        self.profile.blockSignals(False)

    def _profile_chosen(self, name):
        p = calibration.load().get(name)
        if p:
            self.sens.setValue(p["sensitivity_cps_per_MBq"])
            if p.get("isotope") in dosimetry.ISOTOPES:
                self.iso.setCurrentText(p["isotope"])
            self.half.setValue(p.get("half_life_h", self.half.value()))

    def save_profile(self):
        try:
            calibration.save_profile(self.profile.currentText(), self.sens.value(), self.iso.currentText(), self.half.value())
        except ValueError as e:
            self.main.status(str(e))
            return
        self.load_profiles()
        self.main.status(f"Profile '{self.profile.currentText()}' saved: sensitivity {self.sens.value():g} cps/MBq")

    def delete_profile(self):
        calibration.delete_profile(self.profile.currentText())
        self.load_profiles()

    def calib(self) -> dict:
        d = {"sensitivity": self.sens.value(), "time_s": self.time.value()}
        d["decay"] = measure.decay_factor(self.half.value(), self.elapsed.value()) if self.decay_on.isChecked() else 1.0
        return d

    def frame(self) -> pd.DataFrame:
        """One row per ROI of the case, measured on the functional image shown (PET or SPECT)."""
        s = self.main.session
        if not s:
            return pd.DataFrame()
        rows = []
        for name in s.rois:
            _, rr = s.roi_details(name)
            rows += [r for r in rr if r["image"] == s.measure_name]
        return measure.roi_table(rows, self.calib())

    def refresh(self):
        s = self.main.session
        if s:
            info = s.volumes[s.measure_name].info
            txt = ""
            if "frame_duration_s" in info:
                n = info.get("NumberOfFramesInRotation") or info.get("NumberOfRotations") or 1
                txt = (f"File says {info['frame_duration_s']:g} s per projection and {n} projection(s) "
                       f"(about {info['frame_duration_s'] * n:g} s in total, to be verified against your protocol). ")
            self.hint.setText(txt + "Counts become MBq as: total counts / (sensitivity x time), optionally decay-corrected. "
                                    "Small ROIs underestimate activity (partial volume); no recovery correction is applied.")
        df = self.frame()
        self.model.set(df)
        self.table.resizeColumnsToContents()

    def copy(self):
        QtWidgets.QApplication.clipboard().setText(self.frame().to_csv(sep="\t", index=False))
        self.main.status("Table copied to the clipboard (paste into Excel).")

    def export(self):
        df = self.frame()
        self.main.export_table(df, "roi_measurements")
