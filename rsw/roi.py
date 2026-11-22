"""Regions of interest (ROIs) for measurement, kept apart from the segmentation masks: an ROI is drawn by hand
(freehand, polygon, ellipse, circle) on one slice or as a 3D sphere, or copied from a mask; it is saved as a label
file (NIfTI on the reference grid) with a small JSON file holding its name and shape, and it can be loaded again.
No Qt here; the window draws and forwards the clicks."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROI_COLORS = [(255, 220, 0), (0, 230, 255), (255, 120, 220), (140, 255, 90), (255, 160, 60), (170, 150, 255)]
SHAPES = ("freehand", "polygon", "ellipse", "circle", "sphere", "mask")
PLANE = {"axial": 0, "coronal": 1, "sagittal": 2}            # canonical axis held fixed by each view


@dataclass
class Roi:
    name: str
    array: np.ndarray                 # canonical view (z, y, x) of the base array, uint8
    color: tuple
    shape: str = "freehand"
    view: str = ""                    # plane of a 2D ROI ("" for 3D)
    slice_index: int | None = None
    path: str = ""
    saved: bool = False
    visible: bool = True


def plane_box(view: str, index: int):
    return {"axial": (index, slice(None), slice(None)), "coronal": (slice(None), index, slice(None)),
            "sagittal": (slice(None), slice(None), index)}[view]


def plane_shape(shape3d, view: str):
    return {"axial": shape3d[1:], "coronal": (shape3d[0], shape3d[2]), "sagittal": shape3d[:2]}[view]


def ellipse_region(shape2d, r0: float, c0: float, r1: float, c1: float) -> np.ndarray:
    """Filled ellipse inscribed in the box between two corners (row, col), at least one pixel."""
    rc, cc = (r0 + r1) / 2.0, (c0 + c1) / 2.0
    ar, ac = max(abs(r1 - r0) / 2.0, 0.5), max(abs(c1 - c0) / 2.0, 0.5)
    rr, cc_ = np.ogrid[:shape2d[0], :shape2d[1]]
    return ((rr - rc) / ar) ** 2 + ((cc_ - cc) / ac) ** 2 <= 1.0


def circle_region(shape2d, r: int, c: int, radius_mm: float, pixel_mm) -> np.ndarray:
    """Disc of a radius in millimetres (the pixel may be non-square) around pixel (r, c)."""
    rr, cc = np.ogrid[:shape2d[0], :shape2d[1]]
    return ((rr - r) * pixel_mm[0]) ** 2 + ((cc - c) * pixel_mm[1]) ** 2 <= max(radius_mm, 0.5 * min(pixel_mm)) ** 2


def write_sidecar(path: str | Path, roi: Roi) -> Path:
    p = sidecar_path(path)
    p.write_text(json.dumps({"name": roi.name, "shape": roi.shape, "view": roi.view, "slice_index": roi.slice_index,
                             "color": list(roi.color), "kind": "PySegment ROI"}, indent=1), encoding="utf-8")
    return p


def read_sidecar(path: str | Path) -> dict:
    p = sidecar_path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def sidecar_path(path: str | Path) -> Path:
    p = Path(path)
    stem = p.name[:-7] if p.name.endswith(".nii.gz") else p.stem
    return p.with_name(stem + ".roi.json")
