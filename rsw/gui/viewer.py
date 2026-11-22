"""Slice views (axial, coronal, sagittal) and a PET MIP, drawn with pyqtgraph.

Mouse: wheel = next / previous slice, Ctrl+wheel = zoom, right drag = zoom, middle drag = pan.
Left button depends on the tool: navigate (click moves the cursor, drag pans), paint / erase (drag),
grow (click on a lesion). Shift+left drag adjusts the window in any tool.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from scipy import ndimage

from rsw.orient import VIEW_AXIS, pixel_size, plane_to_index, view_slice

pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)

VIEW_TITLES = {"axial": "Axial", "coronal": "Coronal", "sagittal": "Sagittal"}


def _lut(name: str, invert: bool = False) -> np.ndarray:
    if name == "gray":
        lut = np.repeat(np.arange(256, dtype=np.ubyte)[:, None], 3, axis=1)
    else:
        lut = pg.colormap.get(name).getLookupTable(nPts=256, alpha=False)
    return lut[::-1].copy() if invert else lut


def outline(m: np.ndarray) -> np.ndarray:
    return m & ~ndimage.binary_erosion(m, border_value=0)


def overlay_rgba(layers: list, getter, active: str | None, fill_scale: float = 1.0) -> np.ndarray | None:
    """RGBA image of the visible mask layers for one slice: light fill + solid outline."""
    rgba = None
    for layer in layers:
        if not layer.visible:
            continue
        m = getter(layer) > 0
        if not m.any():
            continue
        if rgba is None:
            rgba = np.zeros(m.shape + (4,), np.ubyte)
        fill = int((90 if layer.name == active else 45) * fill_scale)
        if fill:
            rgba[m] = (*layer.color, fill)
        rgba[outline(m)] = (*layer.color, 255)
    return rgba


def with_rois(rgba, session, getter):
    """Draw the measurement ROIs on top of the mask overlay: solid outline, faint fill for the selected ROI."""
    rois = list(getattr(session, "rois", {}).values())
    if not rois:
        return rgba
    r2 = overlay_rgba(rois, getter, getattr(session, "roi_selected", None), 0.35)
    if r2 is None:
        return rgba
    if rgba is None:
        return r2
    on = r2[..., 3] > 0
    rgba[on] = r2[on]
    return rgba


class SliceViewBox(pg.ViewBox):
    def __init__(self, owner: "SliceView"):
        super().__init__(lockAspect=True, invertY=True, enableMenu=False)
        self.owner = owner
        self._wl = None

    def _rc(self, ev_pos):
        p = self.mapSceneToView(ev_pos)
        rh, cw = self.owner.pixel
        return int(np.floor(p.y() / rh)), int(np.floor(p.x() / cw))

    def wheelEvent(self, ev, axis=None):
        if ev.modifiers() & QtCore.Qt.ControlModifier:
            return super().wheelEvent(ev, axis)
        delta = ev.delta() if hasattr(ev, "delta") else ev.angleDelta().y()
        if ev.modifiers() & QtCore.Qt.AltModifier:                    # Alt + wheel: brush size
            self.owner.handler._brush_step(1 if delta > 0 else -1)
            ev.accept()
            return
        self.owner.step(1 if delta > 0 else -1)
        ev.accept()

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton and self.owner.handler.tool == "label":
            hit = self.owner.label_hit(ev.scenePos())
            if hit:
                self.owner.handler.select_label(hit[0], edit=ev.double())
                ev.accept()
                return
        if ev.button() == QtCore.Qt.LeftButton:
            r, c = self._rc(ev.scenePos())
            self.owner.handler.on_click(self.owner.view, r, c, ev.double())
            ev.accept()
        else:
            ev.ignore()

    def mouseDragEvent(self, ev, axis=None):
        tool = self.owner.handler.tool
        if ev.button() == QtCore.Qt.LeftButton and ev.modifiers() & QtCore.Qt.ShiftModifier:
            if ev.isStart():
                self._wl = ev.buttonDownScenePos()
            d = ev.scenePos() - ev.lastScenePos()
            self.owner.handler.on_window_drag(float(d.x()), float(d.y()))
            ev.accept()
        elif ev.button() == QtCore.Qt.LeftButton and tool == "label":
            if ev.isStart():
                self._drag = self.owner.label_hit(ev.buttonDownScenePos())
            if getattr(self, "_drag", None):
                p = self.mapSceneToView(ev.scenePos())
                self.owner.handler.move_label(self.owner, *self._drag, p.x(), p.y())
            if ev.isFinish():
                self._drag = None
            ev.accept()
        elif ev.button() == QtCore.Qt.LeftButton and tool in ("paint", "erase") and self.owner.paintable:
            r, c = self._rc(ev.scenePos())
            self.owner.handler.on_paint(self.owner.view, r, c, ev.isStart(), ev.isFinish())
            ev.accept()
        elif (ev.button() == QtCore.Qt.LeftButton and tool == "roi" and self.owner.paintable
              and self.owner.handler.roi_shape.currentData() in ("freehand", "ellipse")):
            r, c = self._rc(ev.scenePos())
            self.owner.handler.on_roi_drag(self.owner.view, r, c, ev.isStart(), ev.isFinish())
            ev.accept()
        elif ev.button() == QtCore.Qt.MiddleButton:
            ev.accept()
            tr = self.mapToView(ev.pos()) - self.mapToView(ev.lastPos())
            self.translateBy(x=-tr.x(), y=-tr.y())
        else:
            super().mouseDragEvent(ev, axis)


class SliceView(QtWidgets.QWidget):
    """One orthogonal view; `handler` is the main window (tool, cursor, painting)."""

    def __init__(self, view: str, handler, parent=None):
        super().__init__(parent)
        self.view, self.handler = view, handler
        self.paintable = True
        self.session = None
        self.index = 0
        self.pixel = (1.0, 1.0)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("k")
        self.glw.ci.setContentsMargins(0, 0, 0, 0)
        self.vb = SliceViewBox(self)
        self.glw.addItem(self.vb)
        lay.addWidget(self.glw)
        self.base = pg.ImageItem()
        self.over = pg.ImageItem()
        self.mask = pg.ImageItem()
        for it in (self.base, self.over, self.mask):
            self.vb.addItem(it)
        pen = pg.mkPen((0, 220, 120, 150), width=1)
        self.hline = pg.InfiniteLine(angle=0, pen=pen, movable=False)
        self.vline = pg.InfiniteLine(angle=90, pen=pen, movable=False)
        self.vb.addItem(self.hline)
        self.vb.addItem(self.vline)
        self.label = pg.TextItem("", color=(235, 235, 235), anchor=(0, 0), fill=pg.mkBrush(0, 0, 0, 160))
        self.label.setParentItem(self.vb)
        self.label.setPos(4, 2)
        # brush outline in true size (view coordinates are millimetres); shown for paint / erase only
        self.ring = QtWidgets.QGraphicsEllipseItem()
        self.ring.setPen(pg.mkPen((255, 255, 0, 230), width=1, cosmetic=True))
        self.ring.setBrush(pg.mkBrush(None))
        self.ring.setZValue(50)
        self.ring.setVisible(False)
        self.vb.addItem(self.ring, ignoreBounds=True)
        self.polyitem = QtWidgets.QGraphicsPathItem()          # polygon tool preview
        self.polyitem.setPen(pg.mkPen((0, 255, 255, 255), width=1, cosmetic=True))
        self.polyitem.setZValue(51)
        self.polyitem.setVisible(False)
        self.vb.addItem(self.polyitem, ignoreBounds=True)
        self.glw.scene().sigMouseMoved.connect(self._moved)

    # ------------------------------------------------------------------ data
    def set_session(self, session):
        self.session = session
        self.pixel = pixel_size(self.view, session.orient.spacing_zyx)
        self.index = session.shape[VIEW_AXIS[self.view]] // 2
        self.refresh()
        self.reset_zoom()

    def reset_zoom(self):
        if self.session is not None:
            self.vb.autoRange(padding=0.02)

    def n_slices(self) -> int:
        return self.session.shape[VIEW_AXIS[self.view]] if self.session else 1

    def step(self, d: int):
        if self.session is None:
            return
        self.index = int(np.clip(self.index + d, 0, self.n_slices() - 1))
        self.handler.on_slice_changed(self.view, self.index)

    def _rect(self, arr2d):
        rh, cw = self.pixel
        return QtCore.QRectF(0, 0, arr2d.shape[1] * cw, arr2d.shape[0] * rh)

    def refresh(self):
        s, h = self.session, self.handler
        if s is None:
            return
        base_name, over_name = h.display_names()
        b = np.ascontiguousarray(view_slice(s.can[base_name], self.view, self.index))
        self.base.setImage(b, autoLevels=False, levels=h.levels(base_name), lut=h.lut(base_name, base=True))
        self.base.setRect(self._rect(b))
        if over_name:
            o = np.ascontiguousarray(view_slice(s.can[over_name], self.view, self.index))
            self.over.setImage(o, autoLevels=False, levels=h.levels(over_name), lut=h.lut(over_name, base=False))
            self.over.setRect(self._rect(o))
            self.over.setOpacity(h.overlay_opacity)
            self.over.setVisible(True)
        else:
            self.over.setVisible(False)
        rgba = with_rois(overlay_rgba(list(s.layers.values()), lambda l: view_slice(l.array, self.view, self.index), s.active,
                                      h.mask_fill_scale()), s, lambda r: view_slice(r.array, self.view, self.index))
        if rgba is not None:
            self.mask.setImage(rgba, autoLevels=False)
            self.mask.setRect(self._rect(b))
            self.mask.setVisible(True)
        else:
            self.mask.setVisible(False)
        self.label.setText(f"{VIEW_TITLES[self.view]}  {self.index + 1}/{self.n_slices()}")
        self.draw_labels()
        self._draw_poly()
        self.set_cursor(h.cursor)

    def draw_labels(self):
        """Text labels with optional arrows (Labels tab) and, if switched on, the names of the masks and ROIs next
        to them; they are scene items, so 'Save view as image' exports them with the image. A label is stored in
        view millimetres: tip (x, y) and the offset of its text (dx, dy); both can be dragged with the Label tool."""
        for it in getattr(self, "_labels", []):
            self.vb.removeItem(it)
        self._labels, self._hits = [], []
        s, h = self.session, self.handler

        def text(t, x, y, color, anchor, size, selected=False):
            it = pg.TextItem(t, color=color, anchor=anchor,
                             fill=pg.mkBrush(40, 60, 170, 210) if selected else pg.mkBrush(0, 0, 0, 140))
            font = QtGui.QFont()
            font.setPointSize(int(size))
            font.setBold(True)
            it.setFont(font)
            it.setPos(x, y)
            it.setZValue(60)
            self.vb.addItem(it, ignoreBounds=True)
            self._labels.append(it)
            return it

        for i, a in enumerate(s.annotations):
            if a["view"] != self.view or a["index"] != self.index:
                continue
            x, y, tx, ty = a["x"], a["y"], a["x"] + a["dx"], a["y"] + a["dy"]
            color, sel = tuple(a["color"]), i == getattr(h, "label_selected", None)
            if a["arrow"]:
                line = QtWidgets.QGraphicsLineItem(tx, ty, x, y)
                line.setPen(pg.mkPen(color, width=2, cosmetic=True))
                line.setZValue(60)
                ang = float(np.degrees(np.arctan2(-(y - ty), -(x - tx)))) if (x, y) != (tx, ty) else 0.0
                head = pg.ArrowItem(pos=(x, y), angle=ang, headLen=14, tailLen=0, pen=pg.mkPen(color),
                                    brush=pg.mkBrush(color))
                head.setZValue(61)
                for it in (line, head):
                    self.vb.addItem(it, ignoreBounds=True)
                    self._labels.append(it)
                anchor = (1.0 if a["dx"] < 0 else 0.0, 1.0 if a["dy"] < 0 else 0.0)
            else:
                anchor = (0.5, 0.5)
            self._hits.append((i, text(a["text"], tx, ty, color, anchor, a.get("size", 12), sel), (x, y)))
        if getattr(h, "show_names", False):
            rh, cw = self.pixel
            named = [(l.name, l.array, l.color) for l in s.layers.values() if l.visible]
            named += [(r.name, r.array, r.color) for r in s.rois.values()]
            for name, arr, color in named:
                rr, cc = np.nonzero(view_slice(arr, self.view, self.index))
                if rr.size:
                    text(name, (cc.max() + 1.5) * cw, rr.min() * rh, tuple(color), (0, 1), 11)

    def label_hit(self, scene_pos):
        """(label number, 'tip' or 'text') under a scene position, or None."""
        for i, _it, (x, y) in getattr(self, "_hits", []):
            q = self.vb.mapViewToScene(QtCore.QPointF(x, y))
            if (q.x() - scene_pos.x()) ** 2 + (q.y() - scene_pos.y()) ** 2 <= 100:
                return i, "tip"
        for i, it, _xy in getattr(self, "_hits", []):
            if it.sceneBoundingRect().contains(scene_pos):
                return i, "text"
        return None

    def _draw_poly(self):
        poly = getattr(self.handler, "poly", None)
        if not poly or poly["view"] != self.view or poly["index"] != self.index or not poly["pts"]:
            self.polyitem.setVisible(False)
            return
        rh, cw = self.pixel
        path = QtGui.QPainterPath()
        pts = [((c + 0.5) * cw, (r + 0.5) * rh) for r, c in poly["pts"]]
        path.moveTo(*pts[0])
        for x, y in pts[1:]:
            path.lineTo(x, y)
        if "start" in poly:                                        # freehand / ellipse ROI being dragged: closed outline
            path.lineTo(*pts[0])
        else:
            for x, y in pts:                                       # small squares at the clicked corners
                path.addRect(x - 0.6 * cw, y - 0.6 * rh, 1.2 * cw, 1.2 * rh)
        self.polyitem.setPath(path)
        self.polyitem.setVisible(True)

    def set_cursor(self, cur):
        a = {"axial": (cur[1], cur[2]), "coronal": (cur[0], cur[2]), "sagittal": (cur[0], cur[1])}[self.view]
        rh, cw = self.pixel
        self.hline.setPos((a[0] + 0.5) * rh)
        self.vline.setPos((a[1] + 0.5) * cw)
        self.hline.setVisible(self.handler.show_cursor)
        self.vline.setVisible(self.handler.show_cursor)

    def _moved(self, pos):
        if self.session is None or not self.vb.sceneBoundingRect().contains(pos):
            return
        p = self.vb.mapSceneToView(pos)
        rh, cw = self.pixel
        r, c = int(np.floor(p.y() / rh)), int(np.floor(p.x() / cw))
        idx = plane_to_index(self.view, self.index, r, c)
        inside = all(0 <= i < n for i, n in zip(idx, self.session.shape))
        self._place_ring(r, c, inside)
        if inside:
            self.handler.on_hover(idx)

    def _place_ring(self, r: int, c: int, inside: bool = True):
        """Outline of the brush at the voxel under the mouse: the radius in mm, and the one-voxel footprint
        (the smallest stroke possible) when the radius is below half a voxel."""
        h = self.handler
        rad = h.ring_radius() if hasattr(h, "ring_radius") else None
        if not (self.paintable and inside and rad is not None):
            self.ring.setVisible(False)
            return
        rh, cw = self.pixel
        if rad < 0.5 * min(rh, cw):
            rad = 0.5 * min(rh, cw)          # a stroke always paints at least the voxel under the mouse
        cx, cy = (c + 0.5) * cw, (r + 0.5) * rh
        self.ring.setRect(cx - rad, cy - rad, 2 * rad, 2 * rad)
        self.ring.setVisible(True)


class MipView(SliceView):
    """Coronal maximum-intensity projection of the PET (inverted grey); a click jumps to the hottest voxel
    along the projection line."""

    def __init__(self, handler, parent=None):
        super().__init__("coronal", handler, parent)
        self.paintable = False
        self.mip = None
        self.vb.mouseClickEvent = self._click
        self.vb.wheelEvent = lambda ev, axis=None: pg.ViewBox.wheelEvent(self.vb, ev, axis)

    def set_session(self, session):
        self.session = session
        self.pixel = pixel_size("coronal", session.orient.spacing_zyx)
        name = session.pet_name or session.ref.name
        self.mip = session.can[name].max(axis=1)
        self.refresh()
        self.reset_zoom()

    def refresh(self):
        s, h = self.session, self.handler
        if s is None or self.mip is None:
            return
        name = s.pet_name or s.ref.name
        self.base.setImage(self.mip, autoLevels=False, levels=h.levels(name), lut=_lut("gray", invert=True))
        self.base.setRect(self._rect(self.mip))
        self.over.setVisible(False)
        rgba = with_rois(overlay_rgba(list(s.layers.values()), lambda l: l.array.max(axis=1), s.active, h.mask_fill_scale()),
                         s, lambda r: r.array.max(axis=1))
        self.mask.setVisible(rgba is not None)
        if rgba is not None:
            self.mask.setImage(rgba, autoLevels=False)
            self.mask.setRect(self._rect(self.mip))
        self.label.setText("MIP")
        self.set_cursor(h.cursor)

    def _click(self, ev):
        if ev.button() != QtCore.Qt.LeftButton or self.session is None:
            return ev.ignore()
        p = self.vb.mapSceneToView(ev.scenePos())
        rh, cw = self.pixel
        a0, a2 = int(p.y() // rh), int(p.x() // cw)
        if 0 <= a0 < self.session.shape[0] and 0 <= a2 < self.session.shape[2]:
            name = self.session.pet_name or self.session.ref.name
            a1 = int(np.argmax(self.session.can[name][a0, :, a2]))
            self.handler.set_cursor((a0, a1, a2), jump=True)
        ev.accept()

    def _moved(self, pos):
        pass


class Viewer(QtWidgets.QWidget):
    LAYOUTS = ("2 × 2", "Axial", "Coronal", "Axial + MIP")

    def __init__(self, handler, parent=None):
        super().__init__(parent)
        self.views = {v: SliceView(v, handler) for v in ("axial", "coronal", "sagittal")}
        self.mipview = MipView(handler)
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(2)
        self.set_layout("2 × 2")

    def all(self):
        return [*self.views.values(), self.mipview]

    def set_layout(self, name: str):
        for w in self.all():
            self.grid.removeWidget(w)
            w.hide()
        place = {"2 × 2": [("axial", 0, 0), ("coronal", 0, 1), ("sagittal", 1, 0), ("mip", 1, 1)],
                 "Axial": [("axial", 0, 0)],
                 "Coronal": [("coronal", 0, 0)],
                 "Axial + MIP": [("axial", 0, 0), ("mip", 0, 1)]}[name]
        for key, r, c in place:
            w = self.mipview if key == "mip" else self.views[key]
            self.grid.addWidget(w, r, c)
            w.show()

    def set_session(self, s):
        for w in self.all():
            w.set_session(s)

    def refresh(self, only_masks: bool = False):
        for w in self.all():
            w.refresh()
