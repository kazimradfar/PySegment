"""Display orientation without copying or resampling.

Images and masks are kept on their own voxel grid (so masks are saved on exactly the grid of the image).
For display and editing, `canonical()` returns a numpy *view* whose axes are (inferior, posterior, left)
for (axis 0, 1, 2), using the direction cosines of the image. Because it is a view, edits made on it are
made on the original array. With this, every view shows the usual radiological orientation:

    axial    canonical[k]          rows anterior → posterior, columns patient right → left
    coronal  canonical[:, j, :]    rows superior → inferior,  columns patient right → left
    sagittal canonical[:, :, i]    rows superior → inferior,  columns anterior → posterior
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# target canonical numpy axes: (physical LPS axis, sign along which the index grows)
_TARGET = ((2, -1), (1, +1), (0, +1))     # axis0 → inferior (-S), axis1 → posterior (+P), axis2 → left (+L)


@dataclass(frozen=True)
class Orientation:
    order: tuple          # numpy axes of the original array taken in canonical order
    flips: tuple          # flip each canonical axis?
    spacing_zyx: tuple    # canonical spacing (mm) for axes 0, 1, 2

    @classmethod
    def from_direction(cls, direction, spacing_xyz) -> "Orientation":
        d = np.asarray(direction, float).reshape(3, 3)       # column c = index axis c (x, y, z)
        phys = [int(np.argmax(np.abs(d[:, c]))) for c in range(3)]
        sign = [int(np.sign(d[phys[c], c])) or 1 for c in range(3)]
        if sorted(phys) != [0, 1, 2]:                         # degenerate (very oblique): keep array order
            return cls((0, 1, 2), (False, False, False), (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]))
        order, flips, sp = [], [], []
        for p, s in _TARGET:
            c = phys.index(p)                                 # index axis (x=0, y=1, z=2) along physical axis p
            order.append(2 - c)                               # numpy axis of that index axis
            flips.append(sign[c] != s)
            sp.append(spacing_xyz[c])
        return cls(tuple(order), tuple(flips), tuple(sp))

    def canonical(self, arr: np.ndarray) -> np.ndarray:
        v = arr.transpose(self.order)
        return v[tuple(slice(None, None, -1) if f else slice(None) for f in self.flips)]

    @property
    def spacing_xyz(self) -> tuple:
        """Canonical spacing in (x, y, z) order, as the editing tools expect."""
        return (self.spacing_zyx[2], self.spacing_zyx[1], self.spacing_zyx[0])

    def to_original(self, idx_c, shape_original) -> tuple:
        """Canonical index (a0, a1, a2) → index in the original numpy array."""
        out = [0, 0, 0]
        for t, (ax, f) in enumerate(zip(self.order, self.flips)):
            n = shape_original[ax]
            out[ax] = (n - 1 - idx_c[t]) if f else idx_c[t]
        return tuple(int(v) for v in out)


def view_slice(can: np.ndarray, view: str, index: int) -> np.ndarray:
    if view == "axial":
        return can[index]
    if view == "coronal":
        return can[:, index, :]
    return can[:, :, index]


VIEW_AXIS = {"axial": 0, "coronal": 1, "sagittal": 2}


def plane_to_index(view: str, index: int, row: int, col: int) -> tuple:
    """(row, column) in a view → canonical (a0, a1, a2)."""
    if view == "axial":
        return (index, row, col)
    if view == "coronal":
        return (row, index, col)
    return (row, col, index)


def pixel_size(view: str, spacing_zyx) -> tuple:
    """(row height, column width) in mm of a view."""
    z, y, x = spacing_zyx
    return {"axial": (y, x), "coronal": (z, x), "sagittal": (z, y)}[view]
