"""Mask layers and editing tools (independent of the user interface, so they are unit-tested):
brush paint / erase (2D disk or 3D sphere, optionally limited to voxels above a threshold), PET region
growing from a click (fixed value or % of the local maximum), removal of small islands, connected
lesions, and undo / redo."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

AXES = {"axial": 0, "coronal": 1, "sagittal": 2}       # array axis that is fixed in each view (z, y, x)


@dataclass
class Edit:
    box: tuple            # slices (z, y, x)
    before: np.ndarray


@dataclass
class MaskLayer:
    name: str
    array: np.ndarray                      # (z, y, x) uint8
    color: tuple = (230, 40, 40)
    path: str = ""
    reader: str = ""
    visible: bool = True
    locked: bool = False
    saved: bool = True
    _undo: list = field(default_factory=list, repr=False)
    _redo: list = field(default_factory=list, repr=False)
    _group: Edit | None = field(default=None, repr=False)
    max_undo: int = 30

    # ------------------------------------------------------------------ strokes
    def begin_group(self, box) -> None:
        """One undo step for a whole brush stroke: everything changed until end_group() must lie in box."""
        self._group = Edit(box, self.array[box].copy())

    def end_group(self) -> bool:
        g, self._group = self._group, None
        if g is None or np.array_equal(g.before, self.array[g.box]):
            return False
        self._undo.append(g)
        if len(self._undo) > self.max_undo:
            self._undo.pop(0)
        self._redo.clear()
        return True

    # ------------------------------------------------------------------ undo
    def _record(self, box) -> None:
        self._undo.append(Edit(box, self.array[box].copy()))
        if len(self._undo) > self.max_undo:
            self._undo.pop(0)
        self._redo.clear()
        self.saved = False

    def undo(self) -> bool:
        if not self._undo:
            return False
        e = self._undo.pop()
        self._redo.append(Edit(e.box, self.array[e.box].copy()))
        self.array[e.box] = e.before
        self.saved = False
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        e = self._redo.pop()
        self._undo.append(Edit(e.box, self.array[e.box].copy()))
        self.array[e.box] = e.before
        self.saved = False
        return True

    def apply(self, region: np.ndarray, box, value: int) -> int:
        """Set voxels of `region` (bool, shape of box) to value; returns the number of changed voxels."""
        if self.locked:
            raise PermissionError(f"layer '{self.name}' is locked")
        cur = self.array[box]
        change = region & (cur != value)
        n = int(change.sum())
        if n:
            if self._group is None:
                self._record(box)
            else:
                self.saved = False
            cur[change] = value
        return n

    @property
    def voxels(self) -> int:
        return int((self.array > 0).sum())


# ---------------------------------------------------------------------- geometry helpers

def _box(center, radius_vox, shape):
    return tuple(slice(max(0, int(c - r)), min(s, int(c + r) + 1)) for c, r, s in zip(center, radius_vox, shape))


def brush_region(shape, center_zyx, radius_mm: float, spacing_xyz, view: str = "axial", sphere: bool = False,
                 image: np.ndarray | None = None, low: float | None = None, high: float | None = None):
    """Boolean region and its box for a brush stroke at center (z, y, x).
    2D: a disk in the plane of the view; sphere: a 3D ball. Optional intensity limits (e.g. SUV >= 2.5)."""
    sz = np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], float)
    r = radius_mm / sz
    if not sphere:
        r[AXES[view]] = 0
    box = _box(center_zyx, np.ceil(r), shape)
    grids = np.ogrid[box[0], box[1], box[2]]
    d = np.zeros([b.stop - b.start for b in box], float)
    for g, c, rr in zip(grids, center_zyx, r):
        if rr > 0:
            d = d + ((g - c) / rr) ** 2
        else:
            d = d + np.where(g == c, 0.0, np.inf)
    region = d <= 1.0
    if image is not None:
        sub = image[box]
        if low is not None:
            region &= sub >= low
        if high is not None:
            region &= sub <= high
    return region, box


class RegionTooLarge(ValueError):
    pass


def region_grow(image: np.ndarray, seed_zyx, spacing_xyz, mode: str = "percent", value: float = 41.0,
                search_mm: float = 10.0, max_extent_mm: float = 150.0, max_ml: float = 300.0):
    """PET lesion from one click: threshold = value (absolute, e.g. SUV 2.5) or value % of the local maximum
    found within search_mm of the click; the connected component (26-neighbourhood) containing the hottest
    voxel near the click is returned, limited to a cube of max_extent_mm. Returns (region, box, threshold)."""
    sz = np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], float)
    sbox = _box(seed_zyx, np.ceil(search_mm / sz), image.shape)
    local = image[sbox]
    peak_local = np.unravel_index(int(np.argmax(local)), local.shape)
    peak = tuple(int(b.start + p) for b, p in zip(sbox, peak_local))
    peak_val = float(image[peak])
    thr = value if mode == "absolute" else peak_val * value / 100.0
    box = _box(peak, np.ceil(max_extent_mm / 2 / sz), image.shape)
    sub = image[box] >= thr
    lab, _ = ndimage.label(sub, structure=np.ones((3, 3, 3), bool))
    p_in = tuple(p - b.start for p, b in zip(peak, box))
    if not sub[p_in]:
        return np.zeros_like(sub), box, thr
    region = lab == lab[p_in]
    if region.sum() * float(np.prod(spacing_xyz)) / 1000.0 > max_ml:
        raise RegionTooLarge(f"the region would be larger than {max_ml:g} mL (threshold {thr:.2f}); "
                             "click on the lesion itself or use a higher / absolute threshold")
    bkg = local_background(image[box], region, sz) if mode == "percent" else None
    if mode == "percent" and bkg is not None and bkg > 0 and thr < LOW_CONTRAST * bkg:
        raise RegionTooLarge(f"the threshold ({thr:.2f}) is too close to the background around the lesion ({bkg:.2f}): "
                             "the region would leak into normal tissue. The lesion contrast is too low for this "
                             "percentage; use an absolute threshold, e.g. half-way between background and lesion")
    return region, box, thr


LOW_CONTRAST = 1.5       # threshold / local background below this: the region leaks into the background (QIBA DROmtv: every
                         # leak had a ratio < 1.5, every non-leaking region >= 1.64; validation/qiba_dro/)


def local_background(sub: np.ndarray, region: np.ndarray, spacing_zyx, inner_mm: float = 4.0, outer_mm: float = 12.0):
    """Median of the voxels in a shell 4-12 mm around the region (voxels of value 0, outside the body, ignored);
    None when the shell is empty."""
    def grow(mm):
        st = ndimage.generate_binary_structure(3, 1)
        it = max(1, int(round(mm / float(min(spacing_zyx)))))
        return ndimage.binary_dilation(region, st, iterations=it)
    shell = grow(outer_mm) & ~grow(inner_mm)
    v = sub[shell]
    v = v[v > 0]
    return float(np.median(v)) if v.size >= 10 else None


def remove_islands(layer: MaskLayer, min_voxels: int) -> int:
    """Remove connected components smaller than min_voxels; returns the number removed."""
    lab, n = ndimage.label(layer.array > 0, structure=np.ones((3, 3, 3), bool))
    if n == 0:
        return 0
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    small = np.isin(lab, np.nonzero(sizes < min_voxels)[0] + 1)
    if small.any():
        full = tuple(slice(0, s) for s in layer.array.shape)
        layer.apply(small, full, 0)
    return int((sizes < min_voxels).sum())


def _replace_mask(layer: MaskLayer, new: np.ndarray) -> int:
    """Set the whole layer to ``new`` (bool) as ONE undo step; returns the number of changed voxels."""
    old = layer.array > 0
    full = tuple(slice(0, n) for n in layer.array.shape)
    add, drop = new & ~old, old & ~new
    n = int(add.sum() + drop.sum())
    if n:
        layer.begin_group(full)
        layer.apply(add, full, 1)
        layer.apply(drop, full, 0)
        layer.end_group()
    return n


def smooth_mask(layer: MaskLayer, sigma_mm: float, spacing_xyz) -> int:
    """Smooth the outline of the layer: Gaussian blur of the binary mask (sigma in mm, per axis) re-thresholded at
    0.5. Removes jagged edges and thin spurs; a convex blob keeps its volume to within about a voxel layer."""
    sz = np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], float)
    sig = sigma_mm / sz
    m = layer.array > 0
    if not m.any() or sigma_mm <= 0:
        return 0
    blurred = ndimage.gaussian_filter(m.astype(np.float32), sig, mode="constant")
    return _replace_mask(layer, blurred >= 0.5)


def margin_mask(layer: MaskLayer, margin_mm: float, spacing_xyz) -> int:
    """Grow (margin > 0) or shrink (margin < 0) the mask by a distance in mm, measured in real millimetres even
    when the voxels are not cubic (Euclidean distance transform with the voxel spacing)."""
    sz = np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], float)
    m = layer.array > 0
    if not m.any() or margin_mm == 0:
        return 0
    if margin_mm > 0:
        new = ndimage.distance_transform_edt(~m, sampling=sz) <= margin_mm
    else:
        new = ndimage.distance_transform_edt(m, sampling=sz) > -margin_mm
    return _replace_mask(layer, new)


def fill_holes(layer: MaskLayer, per_slice: bool = True) -> int:
    """Fill enclosed holes. per_slice=True fills inside each axial slice (closed outlines drawn slice by slice become
    solid); False fills 3D cavities only."""
    m = layer.array > 0
    if per_slice:
        new = np.stack([ndimage.binary_fill_holes(sl) for sl in m])
    else:
        new = ndimage.binary_fill_holes(m)
    return _replace_mask(layer, new)


def keep_largest(layer: MaskLayer) -> int:
    """Keep only the largest connected component (26-connectivity)."""
    m = layer.array > 0
    lab, n = ndimage.label(m, structure=np.ones((3, 3, 3), bool))
    if n <= 1:
        return 0
    sizes = ndimage.sum(m, lab, index=np.arange(1, n + 1))
    return _replace_mask(layer, lab == (int(np.argmax(sizes)) + 1))


def clear_mask(layer: MaskLayer) -> int:
    return _replace_mask(layer, np.zeros(layer.array.shape, bool))


def combine(layer: MaskLayer, other: np.ndarray, op: str) -> int:
    """layer = layer OP other, with OP union / intersect / subtract (one undo step)."""
    a, b = layer.array > 0, other > 0
    new = {"union": a | b, "intersect": a & b, "subtract": a & ~b}.get(op)
    if new is None:
        raise ValueError(f"unknown operation {op!r}")
    return _replace_mask(layer, new)


def interpolate_slices(layer: MaskLayer, spacing_xyz) -> int:
    """Fill the axial slices between drawn ones: for every pair of neighbouring non-empty slices with empty slices
    in between, the outline is interpolated from the signed distance maps of the two slices (shape-based
    interpolation), so a lesion outlined on every 3rd or 5th slice becomes a solid volume."""
    m = layer.array > 0
    filled = np.nonzero(m.reshape(m.shape[0], -1).any(axis=1))[0]
    sy, sx = spacing_xyz[1], spacing_xyz[0]
    new = m.copy()

    def sdist(sl):
        return ndimage.distance_transform_edt(sl, sampling=(sy, sx)) - ndimage.distance_transform_edt(~sl, sampling=(sy, sx))
    for a, b in zip(filled[:-1], filled[1:]):
        if b - a < 2:
            continue
        da, db = sdist(m[a]), sdist(m[b])
        for k in range(a + 1, b):
            w = (k - a) / (b - a)
            new[k] = ((1 - w) * da + w * db) > 0
    return _replace_mask(layer, new)


def threshold_mask(layer: MaskLayer, image: np.ndarray, mode: str, low: float = 2.5, high: float | None = None,
                   percent: float = 41.0) -> int:
    """Threshold segmentation as ONE undo step; returns the number of changed voxels.
      add          : every voxel of the whole image with low <= value (<= high) joins the mask (normal uptake of
                     bladder, kidneys, brain... will join too; refine afterwards or work inside a region)
      refine_abs   : voxels of the existing mask below ``low`` (or above ``high``) are removed
      refine_rel   : per connected lesion of the existing mask, voxels below ``percent`` % of that lesion's own
                     maximum are removed (the usual 41 % / 50 % of SUVmax rule)"""
    m = layer.array > 0
    if mode == "add":
        new = m | ((image >= low) & ((image <= high) if high is not None else True))
    elif mode == "refine_abs":
        new = m & (image >= low) & ((image <= high) if high is not None else True)
    elif mode == "refine_rel":
        lab, n = ndimage.label(m, structure=np.ones((3, 3, 3), bool))
        new = np.zeros_like(m)
        if n:
            peaks = ndimage.maximum(image, lab, index=np.arange(1, n + 1))
            thr = np.concatenate([[np.inf], np.asarray(peaks, float) * percent / 100.0])
            new = m & (image >= thr[lab])
    else:
        raise ValueError(f"unknown threshold mode {mode!r}")
    return _replace_mask(layer, new)


def lesions(mask: np.ndarray, image: np.ndarray | None, spacing_xyz, min_ml: float = 0.0) -> list[dict]:
    """Connected lesions with volume (mL) and, with an image, max / mean value (e.g. SUVmax, SUVmean)."""
    lab, n = ndimage.label(mask > 0, structure=np.ones((3, 3, 3), bool))
    vox_ml = float(np.prod(spacing_xyz)) / 1000.0
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        m = lab[sl] == i
        vol = float(m.sum()) * vox_ml
        if vol < min_ml:
            continue
        rec = {"lesion": i, "voxels": int(m.sum()), "volume_ml": vol,
               "center_zyx": tuple(int(s.start + c) for s, c in zip(sl, ndimage.center_of_mass(m)))}
        if image is not None:
            v = image[sl][m]
            rec.update({"max": float(v.max()), "mean": float(v.mean())})
        out.append(rec)
    return out
