"""Dosimetry window: activities at several times -> time-integrated activity -> absorbed dose. Every choice is
explained on the window itself (tooltips and the text box), so the method can be quoted in a thesis."""
from __future__ import annotations

import pandas as pd
from PySide6 import QtWidgets

from rsw import APP_NAME, dosimetry as ds, measure
from rsw.gui.widgets import PandasModel, finish_scroll, scroll_content, table_view

EXPLAIN = (
    "HOW IT IS CALCULATED\n"
    "1. Measure the activity (MBq) of a lesion or organ on each scan with an ROI (SPECT: needs the counts calibration; "
    "PET: from SUV or Bq/mL). It must be the activity PRESENT at the scan time, never decay-corrected back to the "
    "injection: physical decay is part of the curve ('Selected ROI → row' takes care of this). Enter it here with the time of the scan in hours after the reference moment (for example the "
    "end of the infusion). One target = one name; two or more time points per target.\n"
    "2. Time-integrated activity A~ (MBq·h) = area under the activity-time curve = total number of decays / 3.6e9. "
    "Method 'Trapezoids + effective tail': straight lines between the points, then exponential decay after the last point "
    "using the rate between the last two points. 'Physical tail': decay only by the half-life of the isotope (assumes no "
    "biological clearance, an upper bound). 'One exponential': a straight line through ln(activity) of all points. "
    "The activity is assumed to rise linearly from zero at time 0 to the first point (option).\n"
    "3. Dose (Gy) = decays x absorbed energy per decay / mass (local deposition: the energy stays in the target). "
    "Absorbed energy per decay = alpha energy + electron energy (beta, conversion, Auger) + photon energy x absorbed "
    "fraction. Alpha and electrons are absorbed in the target itself; photons mostly leave it, so their fraction is a "
    "number YOU give (0 = ignore them; for large organs or imaging isotopes such as Tc-99m, F-18, I-131 it is not 0). "
    "ALPHA CHAINS (Ac-225, Ra-223): the daughters add most of the energy; 'Daughters retained' is the fraction that stays "
    "in the target after the recoil (1 = all, 0 = none; kidneys and bone differ from tumours). The daughters are assumed "
    "to be in equilibrium with the parent at the target. 'RBE alpha' (for example 5) gives a second, RBE-weighted dose "
    "in Gy(RBE); the absorbed dose in Gy is always shown separately. Mass = volume of the ROI x density; the density comes "
    "from the CT inside the ROI when a CT is open (column density_g_ml), else the global value. The energies "
    "are indicative table values: replace them by the numbers of your own reference (MIRD / ICRP 107). "
    "Or enter an S-value (Gy per MBq·h) from OLINDA/MIRD to replace this step.\n"
    "4. With the administered activity you also get the residence time (A~ / A administered) and Gy per GBq.\n"
    "LIMITS: activity must come from a validated calibration; partial volume lowers the activity of small lesions; "
    "two or three points give a rough curve; check the notes column.")


class DosimetryDialog(QtWidgets.QDialog):
    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.setWindowTitle(f"{APP_NAME} – dosimetry")
        content = scroll_content(self)
        lay = QtWidgets.QVBoxLayout(content)
        self.points = QtWidgets.QTableWidget(0, 5)
        self.points.setHorizontalHeaderLabels(["target", "time_h", "activity_MBq", "volume_ml", "density_g_ml"])
        self.points.setMinimumHeight(150)
        lay.addWidget(self.points)
        row = QtWidgets.QHBoxLayout()
        for label, fn, tip in (("Add row", lambda: self.add_row(), "Adds an empty row to fill in"),
                               ("Selected ROI → row", self.from_roi, "Takes activity (MBq) and volume (mL) of the ROI selected "
                                "in the ROI tab; give target name and time first (boxes at the right)"),
                               ("Remove row", self.remove_row, "Removes the selected row"),
                               ("Load CSV", self.load_csv, "Columns: target, time_h, activity_MBq, volume_ml"),
                               ("Save CSV", self.save_csv, "Saves the time points (needs an activated licence)")):
            b = QtWidgets.QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        self.target = QtWidgets.QLineEdit("lesion1")
        self.target.setMaximumWidth(110)
        self.time_h = self._spin(0, 5000, 24.0, 1, " h")
        row.addWidget(self.target)
        row.addWidget(self.time_h)
        lay.addLayout(row)
        form = QtWidgets.QGridLayout()
        self.method = QtWidgets.QComboBox()
        for k, v in ds.METHODS.items():
            self.method.addItem(v, k)
        self.method.setToolTip("How the area under the activity-time curve is obtained, see the text below")
        self.iso = QtWidgets.QComboBox()
        self.iso.addItems(list(ds.ISOTOPES) + ["custom"])
        self.half = self._spin(0.001, 1e6, 159.53, 2, " h")
        self.half.setToolTip("Physical half-life of the isotope (Lu-177: 159.53 h)")
        self.e_alpha = self._spin(0, 100, 0.0, 4, " MeV")
        self.e_alpha.setToolTip("Alpha energy per decay, including the chain daughters that stay (see 'Daughters retained')")
        self.e_electron = self._spin(0, 100, ds.ISOTOPES["Lu-177"]["electron"], 4, " MeV")
        self.e_electron.setToolTip("Beta, conversion-electron and Auger energy per decay (absorbed in the target)")
        self.e_photon = self._spin(0, 100, ds.ISOTOPES["Lu-177"]["photon"], 4, " MeV")
        self.e_photon.setToolTip("Photon energy per decay; only the absorbed fraction below counts")
        self.daug = self._spin(0, 1, 1.0, 2)
        self.daug.setToolTip("Alpha chains: fraction of the daughter nuclides that stays in the target (1 = all)")
        self.phot_frac = self._spin(0, 1, 0.0, 3)
        self.phot_frac.setToolTip("Fraction of the photon energy absorbed in the target (0 = ignore photons)")
        self.rbe = self._spin(1, 20, 1.0, 1)
        self.rbe.setToolTip("Relative biological effectiveness of alpha particles (for example 5); 1 = absorbed dose only")
        self.iso_note = QtWidgets.QLabel("")
        self.iso_note.setWordWrap(True)
        self.iso_note.setStyleSheet("color: gray; font-size: 9pt;")
        self.daug.valueChanged.connect(lambda _v: self._iso(self.iso.currentText()))
        self.density = self._spin(0.1, 5, 1.0, 2, " g/mL")
        self.admin = self._spin(0, 1e6, 0.0, 1, " MBq")
        self.admin.setToolTip("Administered activity; 0 = skip residence time and Gy/GBq")
        self.sval = self._spin(0, 1e6, 0.0, 5, " Gy/MBq·h")
        self.sval.setToolTip("S-value (self-dose) you trust; 0 = use local deposition with the mass of the ROI")
        self.zero = QtWidgets.QCheckBox("Rise from zero at time 0")
        self.zero.setChecked(True)
        for i, (lab, w) in enumerate((("Method", self.method), ("Isotope", self.iso), ("Half-life", self.half),
                                      ("Alpha/decay", self.e_alpha), ("Electron/decay", self.e_electron),
                                      ("Photon/decay", self.e_photon), ("Daughters retained", self.daug),
                                      ("Photon absorbed", self.phot_frac), ("RBE alpha", self.rbe),
                                      ("Density", self.density), ("Administered", self.admin), ("S-value", self.sval))):
            form.addWidget(QtWidgets.QLabel(lab), i // 2, 2 * (i % 2))
            form.addWidget(w, i // 2, 2 * (i % 2) + 1)
        form.addWidget(self.zero, 5, 1)
        form.addWidget(self.iso_note, 6, 0, 1, 4)
        lay.addLayout(form)
        run = QtWidgets.QPushButton("Calculate")
        run.clicked.connect(self.calculate)
        self.wb = self._spin(0, 1e5, 0.0, 2, " h")
        self.wb.setToolTip("Whole-body residence time (hours) for the 'Remainder of body' source: administered activity x this "
                           "time minus the organs entered. 0 = no excretion (T1/2 / ln 2).")
        self.scheme = QtWidgets.QComboBox()
        self.scheme.addItems(["ICRP103", "ICRP60", "no effective dose"])
        self.scheme.setToolTip("Tissue weighting factors for the effective dose (simplified, beta/gamma emitters only)")
        mird = QtWidgets.QPushButton("Organ doses from S-value matrix…")
        mird.setToolTip("MIRD method: dose to every target organ = sum of A~(source) x S(target <- source). Needs an S-value "
                        "CSV (targets as rows, sources as columns) exported from OLINDA/EXM, IDAC or MIRDcalc, and a "
                        "calculated table whose target names equal the source organ names. Optional second file: tissue weights.")
        mird.clicked.connect(self.mird)
        exp = QtWidgets.QPushButton("Export results CSV")
        exp.clicked.connect(self.export)
        r2 = QtWidgets.QHBoxLayout()
        r2.addWidget(run)
        r2.addWidget(mird)
        r2.addWidget(QtWidgets.QLabel("WB residence"))
        r2.addWidget(self.wb)
        r2.addWidget(self.scheme)
        r2.addWidget(exp)
        lay.addLayout(r2)
        self.model = PandasModel()
        lay.addWidget(table_view(self.model), 1)
        txt = QtWidgets.QPlainTextEdit(EXPLAIN)
        txt.setReadOnly(True)
        txt.setMaximumHeight(210)
        lay.addWidget(txt)
        self.iso.currentTextChanged.connect(self._iso)
        self._iso(self.iso.currentText())
        self.df = pd.DataFrame()
        finish_scroll(self, content, 940, 720)

    @staticmethod
    def _spin(lo, hi, val, dec, suffix=""):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(dec)
        s.setValue(val)
        s.setSuffix(suffix)
        s.setMinimumWidth(0)
        return s

    def _iso(self, name):
        if name in ds.ISOTOPES:
            i, f = ds.ISOTOPES[name], self.daug.value()
            self.half.setValue(i["half_life_h"])
            self.e_alpha.setValue(i["alpha"] + f * i["alpha_d"])
            self.e_electron.setValue(i["electron"] + f * i["electron_d"])
            self.e_photon.setValue(i["photon"] + f * i["photon_d"])
            self.iso_note.setText(i["note"])
        else:
            self.iso_note.setText("custom: enter half-life and energies per decay from your reference")

    # ------------------------------------------------------------------ table
    def add_row(self, target="", time_h="", act="", vol="", dens=""):
        r = self.points.rowCount()
        self.points.insertRow(r)
        for c, v in enumerate((target, time_h, act, vol, dens)):
            self.points.setItem(r, c, QtWidgets.QTableWidgetItem(str(v)))

    def remove_row(self):
        r = self.points.currentRow()
        if r >= 0:
            self.points.removeRow(r)

    def from_roi(self):
        """Activity (MBq) and volume of the ROI selected in the ROI tab (the last drawn one if none is selected)."""
        name = self.main._selected_roi()
        if not name:
            return
        s = self.main.session
        _, rows = s.roi_details(name)
        row = [r for r in rows if r["image"] == s.measure_name]
        calib = dict(self.main.measure_dlg.calib()) if self.main.measure_dlg else {}
        calib["decay"] = 1.0                       # dosimetry needs the activity present at the scan, never decay-corrected
        df = measure.roi_table(row, calib)
        if "activity_MBq" not in df or df["activity_MBq"].isna().all():
            self.main.status("This ROI has no activity: set the counts calibration in the ROI table window first.")
            return
        ct = [r for r in rows if r.get("density_g_ml") is not None]          # mass density from the CT, if one is open
        dens = f"{ct[0]['density_g_ml']:.4g}" if ct else ""
        self.add_row(self.target.text() or name, self.time_h.value(), f"{df['activity_MBq'][0]:.6g}", f"{df['volume_ml'][0]:.6g}", dens)

    def collect(self) -> list[dict]:
        out = []
        for r in range(self.points.rowCount()):
            cell = [(self.points.item(r, c).text() if self.points.item(r, c) else "").strip() for c in range(5)]
            if not cell[0] and not cell[1] and not cell[2]:
                continue
            try:
                out.append({"target": cell[0] or "target", "time_h": float(cell[1]), "activity_MBq": float(cell[2]),
                            "volume_ml": float(cell[3]) if cell[3] else None,
                            "density_g_ml": float(cell[4]) if cell[4] else None})
            except ValueError:
                raise ValueError(f"row {r + 1}: time and activity must be numbers")
        return out

    def load_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Time points", "", "CSV (*.csv)")
        if not path:
            return
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            self.add_row(r.get("target", "target"), r["time_h"], r["activity_MBq"],
                         "" if pd.isna(r.get("volume_ml", float("nan"))) else r["volume_ml"],
                         "" if pd.isna(r.get("density_g_ml", float("nan"))) else r["density_g_ml"])

    def save_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save time points", "time_points.csv", "CSV (*.csv)")
        if path:
            pd.DataFrame(self.collect()).to_csv(path, index=False)

    # ------------------------------------------------------------------ result
    def calculate(self):
        try:
            pts = self.collect()
        except ValueError as e:
            self.main.status(str(e))
            return
        iso = {"alpha": self.e_alpha.value(), "electron": self.e_electron.value(), "photon": self.e_photon.value()}
        e_abs, e_rbe = ds.energy_per_decay(iso, 1.0, self.phot_frac.value(), self.rbe.value())
        self.df = ds.dosimetry_table(pts, self.method.currentData(), self.half.value(), e_abs,
                                     self.density.value(), self.admin.value() or None, self.sval.value() or None,
                                     self.zero.isChecked(), weighted_energy_mev=e_rbe)
        self.model.set(self.df)

    def mird(self):
        if self.df.empty or "tia_MBq_h" not in self.df:
            self.calculate()
        if self.df.empty or "tia_MBq_h" not in self.df:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "S-value matrix (targets x sources)", "", "CSV (*.csv)")
        if not path:
            return
        unit, ok = QtWidgets.QInputDialog.getItem(self, APP_NAME, "Unit of the S-values:", list(ds.S_UNITS), 0, False)
        if not ok:
            return
        tia = {r["target"]: r["tia_MBq_h"] for _, r in self.df.iterrows() if pd.notna(r.get("tia_MBq_h"))}
        matrix = pd.read_csv(path, index_col=0)
        rem = next((c for c in matrix.columns if c.strip().lower() in ("remainder of body", "remainder of the body", "remainder")), None)
        if rem and self.admin.value() > 0:                      # remainder of body = whole body minus the organs entered
            tau = self.wb.value() or self.half.value() / 0.693147
            tia[rem] = ds.remainder_tia(self.admin.value(), tau, tia)
        scale = None
        if "mass_g" in self.df and QtWidgets.QMessageBox.question(
                self, APP_NAME, "Scale the self-dose of each organ by reference mass / patient mass?\nYou then pick a CSV with "
                "columns organ, model_mass_g (patient mass = ROI volume x density of the table).") == QtWidgets.QMessageBox.Yes:
            mp, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Reference organ masses", "", "CSV (*.csv)")
            if mp:
                ref = pd.read_csv(mp).set_index("organ")["model_mass_g"].to_dict()
                scale = {r["target"]: ref[r["target"]] / r["mass_g"] for _, r in self.df.iterrows()
                         if r["target"] in ref and pd.notna(r.get("mass_g")) and r["mass_g"] > 0}
        try:
            out = ds.mird_organ_doses(tia, matrix, unit, None, scale)
        except (ValueError, KeyError) as e:
            self.main.status(f"MIRD calculation failed: {e}")
            return
        miss = out.attrs.get("sources_missing_in_matrix")
        if self.scheme.currentText() != "no effective dose":
            e, notes = ds.effective_dose_sv(dict(zip(out["target"], out["dose_Gy"])), self.scheme.currentText())
            out = pd.concat([out, pd.DataFrame({"target": [f"EFFECTIVE DOSE ({self.scheme.currentText()}) Sv"], "dose_Gy": [e]})],
                            ignore_index=True)
            if self.e_alpha.value() > 0:
                notes = ["NOT valid for alpha emitters: the alpha radiation weighting factor (20) is not applied"] + notes
            if notes:
                self.main.status("Effective dose: " + "; ".join(notes))
        self.df = out
        self.model.set(out)
        self.main.status("Organ doses calculated" + (f"; sources not in the matrix and NOT counted: {', '.join(miss)}" if miss else ""))

    def export(self):
        if self.df.empty:
            self.calculate()
        self.main.export_table(self.df, "dosimetry")
