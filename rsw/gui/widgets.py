"""Small reusable pieces: pandas table model, background jobs, licence dialog."""
from __future__ import annotations

import traceback

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

from rsw import APP_LONG_NAME, APP_NAME, APP_VERSION, AUTHOR, COPYRIGHT


class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, df: pd.DataFrame | None = None):
        super().__init__()
        self.df = df if df is not None else pd.DataFrame()

    def set(self, df: pd.DataFrame):
        self.beginResetModel()
        self.df = df
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.df)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.df.columns)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole or not index.isValid():
            return None
        v = self.df.iat[index.row(), index.column()]
        if isinstance(v, float):
            return "" if pd.isna(v) else f"{v:.4g}"
        return str(v)

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return None
        return str(self.df.columns[section]) if orientation == QtCore.Qt.Horizontal else str(section + 1)


def table_view(model: PandasModel) -> QtWidgets.QTableView:
    t = QtWidgets.QTableView()
    t.setModel(model)
    t.setAlternatingRowColors(True)
    t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    t.horizontalHeader().setDefaultSectionSize(90)
    t.verticalHeader().setDefaultSectionSize(20)
    return t


class _Signals(QtCore.QObject):
    progress = QtCore.Signal(int, int, str)
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)


class Job(QtCore.QThread):
    """Runs fn(progress) in a background thread; progress(i, n, text) may be called from fn."""

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self.sig = _Signals()

    def run(self):
        try:
            res = self.fn(lambda i, n, t: self.sig.progress.emit(int(i), int(n), str(t)))
            self.sig.done.emit(res)
        except Exception:  # noqa: BLE001
            self.sig.failed.emit(traceback.format_exc())


def about_text() -> str:
    return (f"<h3>{APP_NAME} {APP_VERSION}</h3><p>{APP_LONG_NAME}</p><p>{COPYRIGHT}<br>Author: {AUTHOR}</p>"
            "<p><b>Licence:</b> MIT (open source)</p>"
            "<p>For research use only. Not a medical device; not for diagnosis or treatment decisions.</p>"
            "<p>Uses open-source components (Qt for Python / PySide6 under LGPLv3, pyqtgraph, NumPy, SciPy, "
            "SimpleITK, elastix, PyRadiomics, pandas, pydicom); see THIRD_PARTY_NOTICES.txt.</p>")


class WheelGuard(QtCore.QObject):
    """Spin boxes, combo boxes and sliders ignore the mouse wheel until they have been clicked (have focus); the wheel
    scrolls the surrounding panel instead, so scrolling a long tool panel never changes a value by accident."""

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Wheel and not obj.hasFocus():
            w = obj.parentWidget()
            while w is not None and not isinstance(w, QtWidgets.QAbstractScrollArea):
                w = w.parentWidget()
            if w is not None:
                QtWidgets.QApplication.sendEvent(w.verticalScrollBar(), ev)
            return True
        return False


_GUARD = None


def guard_wheel(root: QtWidgets.QWidget) -> int:
    """Install the guard on every spin box, combo box and slider below ``root``; returns how many."""
    global _GUARD
    if _GUARD is None:
        _GUARD = WheelGuard()
    n = 0
    for cls in (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox, QtWidgets.QSlider):
        for w in root.findChildren(cls):
            w.setFocusPolicy(QtCore.Qt.StrongFocus)
            w.installEventFilter(_GUARD)
            n += 1
    return n


def polygon_mask(shape, points) -> np.ndarray:
    """Bool mask (rows, cols) of the polygon through ``points`` given as (row, col) in voxel units (voxel centres
    at integer + 0.5 are inside when the centre lies within the outline)."""
    h, w = shape
    img = QtGui.QImage(w, h, QtGui.QImage.Format_Grayscale8)
    img.fill(0)
    p = QtGui.QPainter(img)
    p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))      # the outline voxels (the clicked ones) belong to the mask
    p.setBrush(QtGui.QColor(255, 255, 255))
    p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(c, r) for r, c in points]))
    p.end()
    buf = np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine())[:, :w]
    return buf > 127


def scroll_content(dialog) -> QtWidgets.QWidget:
    """Widget to build a dialog's layout in; finish with ``finish_scroll`` so the dialog scrolls on small screens."""
    return QtWidgets.QWidget(dialog)


def finish_scroll(dialog, content, width: int, height: int):
    """Put ``content`` in a scroll area and open the dialog no larger than the screen, so no part is cut off."""
    sa = QtWidgets.QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QtWidgets.QFrame.NoFrame)
    sa.setWidget(content)
    outer = QtWidgets.QVBoxLayout(dialog)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.addWidget(sa)
    screen = QtGui.QGuiApplication.primaryScreen()
    geo = screen.availableGeometry() if screen else QtCore.QRect(0, 0, width, height)
    dialog.resize(min(width, int(geo.width() * 0.92)), min(height, int(geo.height() * 0.85)))
