"""ROI details window: opens when an ROI is clicked (Navigate tool) or double-clicked in the ROI list.
Shows the ROI geometry, the statistics on every open image (CT in HU, PET in SUV, SPECT in counts), activity when a
counts calibration is set, and the histogram of the functional image inside the ROI."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets

from rsw import APP_NAME, measure
from rsw.gui.widgets import PandasModel, finish_scroll, scroll_content, table_view


class RoiDetailsDialog(QtWidgets.QDialog):
    def __init__(self, main):
        super().__init__(main)
        self.main, self.name = main, None
        self.setWindowTitle(f"{APP_NAME} – ROI details")
        content = scroll_content(self)
        lay = QtWidgets.QVBoxLayout(content)
        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        lay.addWidget(self.info)
        row = QtWidgets.QHBoxLayout()
        for label, fn in (("Copy", self.copy), ("Export CSV", self.export), ("Save ROI…", lambda: self.main.save_roi()),
                          ("Calibration / table", self.main.show_measure_table), ("Close", self.close)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        self.model = PandasModel()
        self.table = table_view(self.model)
        self.table.setMinimumHeight(260)
        lay.addWidget(self.table)
        self.note = QtWidgets.QLabel()
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: gray; font-size: 9pt;")
        lay.addWidget(self.note)
        try:
            import pyqtgraph as pg
            self.plot = pg.PlotWidget()
            self.plot.setMinimumHeight(150)
            self.plot.setMaximumHeight(200)
            lay.addWidget(self.plot)
        except ImportError:                              # pragma: no cover
            self.plot = None
        finish_scroll(self, content, 760, 620)
        self.df = pd.DataFrame()

    def calib(self) -> dict:
        d = getattr(self.main, "measure_dlg", None)
        return d.calib() if d is not None else {}

    def show_roi(self, name: str):
        s = self.main.session
        self.name = name
        info, rows = s.roi_details(name)
        self.df = measure.roi_table(rows, self.calib())          # one row per image (kept for Copy / Export)
        self.model.set(self.display_frame(self.df))
        self.table.resizeColumnsToContents()
        ext = " x ".join(f"{v:.1f}" for v in info["extent_mm"])
        lines = [f"<b>{name}</b> &nbsp; shape: {info['shape']}, plane: {info['plane']}"
                 + (f", slice {info['slice']}" if info.get("slice") else ""),
                 f"Volume {info['volume_ml']:.3f} mL &nbsp; voxels {info['voxels']} &nbsp; extent (x, y, z) {ext} mm",
                 f"Centre (LPS) {info['centre_mm_LPS']} mm" + (f" &nbsp; area {info['area_mm2']:.1f} mm²" if "area_mm2" in info else ""),
                 f"File: {info['file']}"]
        self.info.setText("<br>".join(lines))
        cnt = self.df[self.df["unit"].map(measure.is_counts_unit)] if "unit" in self.df else self.df.iloc[0:0]
        uncal = len(cnt) and ("activity_MBq" not in cnt or cnt["activity_MBq"].isna().any())
        self.note.setText(("SPECT in counts: set sensitivity and acquisition time in the Calibration / table window to get MBq. "
                           if uncal else "")
                          + "Intensity = image value (HU for CT, SUV for PET, counts per voxel for SPECT). peak_1ml = highest "
                          "mean in a 1 mL sphere centred in the ROI. Small ROIs are affected by partial volume.")
        self._histogram(s, name)
        self.setWindowTitle(f"{APP_NAME} – ROI details – {name}")

    LABELS = [("unit", "Unit"), ("mean_intensity", "Mean"), ("sd_intensity", "SD"), ("min_intensity", "Min"),
              ("max_intensity", "Max"), ("median_intensity", "Median"), ("peak_1ml_intensity", "Peak (1 mL sphere)"),
              ("volume_ml", "Volume (mL)"), ("voxels", "Voxels"), ("total_counts", "Total counts"),
              ("activity_MBq", "Activity (MBq)"), ("concentration_MBq_per_ml", "Concentration (MBq/mL)"),
              ("concentration_kBq_per_ml", "Concentration (kBq/mL)")]

    @classmethod
    def display_frame(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Images as columns and the quantities as rows, so everything is readable without scrolling sideways."""
        if df.empty:
            return df
        out = {}
        for _, r in df.iterrows():
            col = []
            for key, _label in cls.LABELS:
                v = r.get(key) if key in df else None
                if isinstance(v, float):
                    v = "" if np.isnan(v) else (f"{v:.6g}" if key != "voxels" else f"{v:.0f}")
                col.append("" if v is None else v)
            out[r["image"]] = col
        t = pd.DataFrame(out, index=[lab for _k, lab in cls.LABELS])
        keep = [i for i in t.index if (t.loc[i] != "").any()]
        return t.loc[keep].reset_index().rename(columns={"index": "Quantity"})

    def _histogram(self, s, name):
        if self.plot is None:
            return
        self.plot.clear()
        img = s.measure_name
        v = s.can[img][s.rois[name].array > 0]
        if v.size == 0:
            return
        y, x = np.histogram(v, bins=min(60, max(10, int(np.sqrt(v.size)))))
        self.plot.plot(x, y, stepMode="center", fillLevel=0, brush=(255, 220, 0, 120))
        self.plot.setLabel("bottom", f"{img} ({s.volumes[img].unit or 'value'})")
        self.plot.setLabel("left", "voxels")

    def copy(self):
        QtWidgets.QApplication.clipboard().setText(self.df.to_csv(sep="\t", index=False))
        self.main.status("ROI table copied (paste into Excel).")

    def export(self):
        self.main.export_table(self.df, f"roi_{self.name}")
