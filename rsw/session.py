"""The open case: images on the reference grid, mask layers, the reader, and the editing operations.
No Qt here, so the whole editing logic is unit-tested; the window only displays and forwards clicks."""
from __future__ import annotations

import re

from pathlib import Path

import numpy as np

from rsw import io, seg
from rsw.orient import Orientation, plane_to_index

PLANE_AXIS = {"axial": 0, "coronal": 1, "sagittal": 2}
from rsw.project import Case, Project

PALETTE = [(230, 40, 40), (40, 140, 255), (250, 200, 0), (60, 200, 90), (200, 80, 220), (255, 140, 0)]


class Session:
    def __init__(self, project: Project | None, case: Case | None, volumes: list[io.Volume]):
        if not volumes:
            raise ValueError("no image")
        self.project, self.case = project, case
        ref = next((v for v in volumes if case and v.name == case.reference), None) or \
            next((v for v in volumes if v.kind == "PET"), volumes[0])
        self.ref = ref
        self.geo = ref.geometry
        self.orient = Orientation.from_direction(self.geo.direction, self.geo.spacing)
        self.native = {v.name: v for v in volumes}     # as loaded (own grid)
        self.base_pick: str | None = None              # image shown as base / overlay; None = automatic
        self.over_pick: str | None = None              # "" = no overlay
        self.transforms: dict[str, object] = {}        # registration per image (reference to image space); none = identity
        self.volumes = {v.name: io.resample_to(v, self.geo) for v in volumes}
        self.can = {n: self.orient.canonical(v.array) for n, v in self.volumes.items()}
        self.layers: dict[str, seg.MaskLayer] = {}
        self._base: dict[str, np.ndarray] = {}         # original-orientation arrays of the layers
        self.rois: dict = {}                           # measurement ROIs (rsw.roi.Roi), apart from the masks
        self.annotations: list[dict] = []              # labels on the image: view, index, r, c, text, arrow
        self._roi_base: dict[str, np.ndarray] = {}
        self._roi_undo: dict[str, list] = {}
        self.roi_selected: str | None = None
        self.active: str | None = None
        self.reader = ""
        self.blind = False
        self.shape = self.can[ref.name].shape

    # ------------------------------------------------------------------ construction
    @classmethod
    def open_case(cls, project: Project, case: Case, reader: str = "", blind: bool = False) -> "Session":
        vols = [io.load_volume(project.path(p), name=n) for n, p in case.images.items()]
        s = cls(project, case, vols)
        s.reader, s.blind = reader, blind
        for name, rel in project.visible_masks(case, reader, blind).items():
            s.add_layer_from_file(name, project.path(rel))
        return s

    @classmethod
    def open_files(cls, paths: list[str]) -> "Session":
        return cls(None, None, [io.load_volume(p) for p in paths])

    @property
    def pet_name(self) -> str | None:
        """First functional image (PET, else SPECT)."""
        return next((n for k in ("PET", "SPECT") for n, v in self.volumes.items() if v.kind == k), None)

    @property
    def ct_name(self) -> str | None:
        return next((n for n, v in self.volumes.items() if v.kind == "CT"), None)

    # ------------------------------------------------------------------ measurements and filtering
    @property
    def measure_name(self) -> str:
        """Image used for ROI statistics and profiles: the functional image (SUV), else the reference."""
        for pick in (self.over_pick, self.base_pick):             # the functional image the user is looking at
            if pick and pick in self.volumes and self.volumes[pick].kind in ("PET", "SPECT"):
                return pick
        return self.pet_name or self.ref.name

    def mask_stats(self, layer_name: str | None = None, name: str | None = None) -> dict:
        """Statistics of the voxels inside a mask layer (all lesions together) on the measured image."""
        from rsw import measure
        layer = self.layers[layer_name or self.active]
        name = name or self.measure_name
        return measure.roi_stats(self.can[name], layer.array > 0, self.orient.spacing_zyx)

    # ------------------------------------------------------------------ measurement ROIs
    def _new_roi(self, name: str, base: np.ndarray, shape: str, view: str = "", index=None, color=None, path: str = ""):
        from rsw.roi import ROI_COLORS, Roi
        if not name or name in self.rois:
            i = len(self.rois) + 1
            while f"ROI {i}" in self.rois:
                i += 1
            name = name if name and name not in self.rois else f"ROI {i}"
        r = Roi(name, self.orient.canonical(base), tuple(color or ROI_COLORS[len(self.rois) % len(ROI_COLORS)]), shape,
                view, index, path)
        self.rois[name], self._roi_base[name] = r, base
        self.roi_selected = name
        return name

    def add_roi(self, region, shape: str, view: str = "", index=None, name: str = "") -> str:
        """New ROI from a boolean region on the canonical grid (z, y, x). Returns its name."""
        base = np.zeros(tuple(self.geo.size[::-1]), np.uint8)
        self.orient.canonical(base)[...] = np.asarray(region, bool)          # canonical is a view: writes into base
        if not base.any():
            raise ValueError("the ROI is empty")
        return self._new_roi(name, base, shape, view, index)

    def add_plane_roi(self, view: str, index: int, region2d, shape: str, name: str = "") -> str:
        from rsw.roi import plane_box
        reg = np.zeros(self.shape, bool)
        reg[plane_box(view, index)] = region2d
        return self.add_roi(reg, shape, view, index, name)

    def edit_roi(self, name: str, region, op: str = "add") -> int:
        """Add a region to an ROI or remove it ('add' / 'remove'); one undo step. Returns the new voxel count.
        Removing everything is refused (delete the ROI instead)."""
        roi = self.rois[name]
        new = roi.array > 0
        new = (new | np.asarray(region, bool)) if op == "add" else (new & ~np.asarray(region, bool))
        if not new.any():
            raise ValueError("this would remove the whole ROI; use Delete instead")
        self._roi_undo.setdefault(name, []).append(self._roi_base[name].copy())
        del self._roi_undo[name][:-20]
        roi.array[...] = new                                  # canonical view: writes into the base array
        roi.saved = False
        if roi.view and op == "add":
            roi.view, roi.slice_index = ("", None) if self._spans_slices(new, roi.view) else (roi.view, roi.slice_index)
        return int(new.sum())

    @staticmethod
    def _spans_slices(region, view):
        ax = PLANE_AXIS[view]
        return np.unique(np.argwhere(region)[:, ax]).size > 1

    def undo_roi(self, name: str) -> bool:
        stack = self._roi_undo.get(name)
        if not stack:
            return False
        self._roi_base[name][...] = stack.pop()
        self.rois[name].saved = False
        return True

    def roi_from_layer(self, layer_name: str | None = None, name: str = "") -> str:
        layer = self.layers[layer_name or self.active]
        return self.add_roi(layer.array > 0, "mask", name=name or f"{layer.name} (mask)")

    def remove_roi(self, name: str) -> None:
        self.rois.pop(name, None)
        self._roi_base.pop(name, None)
        self._roi_undo.pop(name, None)
        if self.roi_selected == name:
            self.roi_selected = next(reversed(self.rois), None) if self.rois else None

    def rename_roi(self, old: str, new: str) -> str:
        if not new or new == old:
            return old
        if new in self.rois:
            raise ValueError(f"an ROI called '{new}' exists already")
        self.rois = {(new if k == old else k): v for k, v in self.rois.items()}
        self._roi_base[new] = self._roi_base.pop(old)
        if old in self._roi_undo:
            self._roi_undo[new] = self._roi_undo.pop(old)
        self.rois[new].name, self.rois[new].saved = new, False
        if self.roi_selected == old:
            self.roi_selected = new
        return new

    def roi_hit(self, idx_zyx) -> str | None:
        """Visible ROI containing the voxel; the most recently drawn one wins when ROIs overlap."""
        for n in reversed(list(self.rois)):
            r = self.rois[n]
            if r.visible and r.array[tuple(idx_zyx)] > 0:
                return n
        return None

    def roi_details(self, name: str) -> tuple[dict, list[dict]]:
        """Geometry of an ROI and one statistics row per open image (CT in HU, PET in SUV, SPECT in counts ...)."""
        from rsw import measure
        roi = self.rois[name]
        region = roi.array > 0
        sp = self.orient.spacing_zyx
        idx = np.argwhere(region)
        lo, hi = idx.min(axis=0), idx.max(axis=0)
        centre_c = idx.mean(axis=0)
        orig = self.orient.to_original(tuple(int(round(v)) for v in centre_c), self.ref.array.shape)
        ref = self.geo.reference()
        centre_mm = ref.TransformContinuousIndexToPhysicalPoint((float(orig[2]), float(orig[1]), float(orig[0])))
        vox_ml = float(np.prod(sp)) / 1000.0
        info = {"name": name, "shape": roi.shape, "plane": roi.view or "3D",
                "slice": (roi.slice_index + 1) if roi.slice_index is not None else None,
                "voxels": int(region.sum()), "volume_ml": float(region.sum() * vox_ml),
                "slices_z": int(hi[0] - lo[0] + 1),
                "extent_mm": tuple(float((h - l + 1) * s) for h, l, s in zip(hi[::-1], lo[::-1], sp[::-1])),
                "centre_mm_LPS": tuple(round(float(v), 1) for v in centre_mm), "file": roi.path or "(not saved)"}
        if roi.view:
            a = [i for i in range(3) if i != PLANE_AXIS[roi.view]]
            info["area_mm2"] = float(region.sum() * sp[a[0]] * sp[a[1]])
        rows = []
        for n, v in self.volumes.items():
            functional = v.kind in ("PET", "SPECT")
            st = measure.roi_stats(self.can[n], region, sp) if functional else self._plain_stats(self.can[n], region, sp)
            row = dict(st, kind="roi", roi=name, image=n, unit=v.unit, weight_kg=v.info.get("weight_kg"),
                       dose_MBq=v.info.get("dose_at_scan_MBq"))      # SUV refers to the dose decayed to the scan start
            if v.kind == "CT":
                rho = measure.ct_density(self.can[n][region])
                row.update({"density_g_ml": float(rho.mean()), "mass_g": float(rho.sum() * vox_ml)})
            rows.append(row)
        return info, rows

    @staticmethod
    def _plain_stats(image, region, sp):
        from rsw import measure
        v = image[region].astype(np.float64)
        st = measure.roi_stats(image[:1, :1, :1], np.zeros((1, 1, 1), bool), sp)        # the empty template
        if v.size:
            st.update({"voxels": int(v.size), "volume_ml": v.size * float(np.prod(sp)) / 1000.0, "mean": float(v.mean()),
                       "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0, "min": float(v.min()), "max": float(v.max()),
                       "median": float(np.median(v)), "peak": None, "total": None})
        return st

    def save_roi(self, name: str, path: str | Path) -> Path:
        from rsw.roi import write_sidecar
        roi = self.rois[name]
        io.save_mask(self._roi_base[name], self.geo, path, note=f"ROI {name}")
        write_sidecar(path, roi)
        roi.path, roi.saved = str(path), True
        return Path(path)

    def load_roi(self, path: str | Path) -> str:
        from rsw.roi import read_sidecar
        arr, resampled = io.load_mask(path, self.geo)
        meta = read_sidecar(path)
        p = Path(path)
        stem = p.name[:-7] if p.name.endswith(".nii.gz") else p.stem
        base = (arr > 0).astype(np.uint8)
        if not base.any():
            raise ValueError(f"{p.name}: the ROI file is empty on this image grid")
        name = self._new_roi(meta.get("name") or stem, base, meta.get("shape", "mask"), meta.get("view", ""),
                             meta.get("slice_index"), meta.get("color"), str(path))
        self.rois[name].saved = not resampled
        return name

    def roi_at(self, idx_zyx, radius_mm: float, name: str | None = None) -> tuple[np.ndarray, dict]:
        """Spherical ROI centred on a canonical voxel; returns (boolean region on the full grid, statistics)."""
        from rsw import measure
        name = name or self.measure_name
        sp = self.orient.spacing_zyx
        region = np.zeros(self.shape, bool)
        r, box = seg.brush_region(self.shape, tuple(idx_zyx), radius_mm, (sp[2], sp[1], sp[0]), "axial", True)
        region[box] = r
        return region, measure.roi_stats(self.can[name], region, sp)

    def add_filtered(self, src: str, sigma_mm: float) -> str:
        """Add a Gaussian-smoothed copy of image ``src`` (sigma in mm, FWHM = 2.355 sigma) as a new image on the
        reference grid. Used to see lesions through the noise or to mimic a lower resolution; the source is unchanged."""
        import SimpleITK as sitk
        vol = self.volumes[src]
        img = self.geo.apply(sitk.GetImageFromArray(vol.array.astype(np.float32)))
        arr = sitk.GetArrayFromImage(sitk.SmoothingRecursiveGaussian(img, float(sigma_mm)))
        name, i = f"{src}_g{sigma_mm:g}", 2
        while name in self.native:
            name, i = f"{src}_g{sigma_mm:g}_{i}", i + 1
        new = io.Volume(name, arr.astype(np.float32), self.geo, vol.kind, vol.unit, "", dict(vol.info, filtered_sigma_mm=sigma_mm))
        self.native[name] = self.volumes[name] = new
        self.can[name] = self.orient.canonical(new.array)
        return name

    # ------------------------------------------------------------------ registration
    def set_transform(self, name: str, transform) -> None:
        """Show image ``name`` through a registration transform (maps reference-grid points to the image's space;
        the output of rsw.registration.register with the session reference as fixed image). The native image is
        never modified. Mask layers are not moved: draw or load masks after accepting the registration."""
        if name == self.ref.name:
            raise ValueError("the reference image cannot be registered onto itself")
        self.transforms[name] = transform
        v = io.resample_to(self.native[name], self.geo, transform=transform)
        self.volumes[name] = v
        self.can[name] = self.orient.canonical(v.array)

    @staticmethod
    def _inverse(t, geo):
        """Inverse of a registration transform. Linear transforms are inverted exactly (one affine); a deformable
        transform is inverted numerically on the grid ``geo`` (the grid it was estimated on)."""
        import SimpleITK as sitk
        from rsw import registration as rg
        m = rg.transform_matrix(t)
        if m is not None:
            return Session._affine(np.linalg.inv(m))
        ref = geo.reference()
        disp = sitk.TransformToDisplacementField(t, sitk.sitkVectorFloat64, ref.GetSize(), ref.GetOrigin(),
                                                 ref.GetSpacing(), ref.GetDirection())
        inv = sitk.InvertDisplacementField(disp, 20, 0.1, 0.1)
        return sitk.DisplacementFieldTransform(sitk.Cast(inv, sitk.sitkVectorFloat64))

    @staticmethod
    def _affine(m):
        import SimpleITK as sitk
        a = sitk.AffineTransform(3)
        a.SetMatrix([float(x) for x in np.asarray(m)[:3, :3].ravel()])
        a.SetTranslation([float(x) for x in np.asarray(m)[:3, 3]])
        return a

    @staticmethod
    def _then(first, second):
        """Transform p -> second(first(p)); a single affine when both are linear, else a flattened composite."""
        import SimpleITK as sitk
        from rsw import registration as rg
        if first is None:
            return second
        if second is None:
            return first
        m1, m2 = rg.transform_matrix(first), rg.transform_matrix(second)
        if m1 is not None and m2 is not None:
            return Session._affine(m2 @ m1)
        c = sitk.CompositeTransform(3)
        c.AddTransform(second)
        c.AddTransform(first)                 # the last added is applied first
        c.FlattenTransform()
        return c

    def set_transforms(self, transforms: dict) -> None:
        """Replace all registration transforms at once and rebuild the displayed images."""
        self.transforms = dict(transforms)
        self.volumes = {n: io.resample_to(v, self.geo, transform=self.transforms.get(n)) for n, v in self.native.items()}
        self.can = {n: self.orient.canonical(v.array) for n, v in self.volumes.items()}

    def set_reference(self, name: str) -> None:
        """Make ``name`` the reference (fixed) image: every image and mask layer is shown on its grid.
        A transform maps points of the current frame to the space of its image. When the new reference had its own
        transform T_R (it was registered before), the frame changes to R's own space, so every other transform
        becomes T_n(T_R^-1(p)) and the old reference becomes T_R^-1: nothing registered earlier is lost or moved.
        Mask layers are resampled through T_R^-1 (nearest neighbour); their undo history is cleared and they count
        as unsaved (their file on disk is untouched)."""
        if name == self.ref.name:
            return
        import SimpleITK as sitk
        old_geo, old = self.geo, {n: (self._base[n], l) for n, l in self.layers.items()}
        old_rois = {n: (self._roi_base[n], r) for n, r in self.rois.items()}
        t_r = self.transforms.get(name)
        rinv = self._inverse(t_r, old_geo) if t_r is not None else None
        new_tf = {}
        for n in self.native:
            if n == name:
                continue
            new_tf[n] = self._then(rinv, self.transforms.get(n)) if rinv is not None else self.transforms.get(n)
        if rinv is not None:
            new_tf[self.ref.name] = rinv                       # the old reference had the identity
        new_tf = {n: t for n, t in new_tf.items() if t is not None}
        self.ref = self.native[name]
        self.geo = self.ref.geometry
        self.orient = Orientation.from_direction(self.geo.direction, self.geo.spacing)
        self.set_transforms(new_tf)
        self.shape = self.can[name].shape
        self.layers, self._base = {}, {}
        for n, (base, l) in old.items():
            img = old_geo.apply(sitk.GetImageFromArray(base))
            arr = sitk.GetArrayFromImage(sitk.Resample(img, self.geo.reference(), rinv or sitk.Transform(),
                                                       sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8))
            new = self._register(n, arr, l.path, l.reader)
            new.visible, new.locked, new.color, new.saved = l.visible, l.locked, l.color, False
        self.active = self.active if self.active in self.layers else next(iter(self.layers), None)
        sel, self.rois, self._roi_base = self.roi_selected, {}, {}
        for n, (base, r) in old_rois.items():
            img = old_geo.apply(sitk.GetImageFromArray(base))
            arr = sitk.GetArrayFromImage(sitk.Resample(img, self.geo.reference(), rinv or sitk.Transform(),
                                                       sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8))
            if arr.any():
                self._new_roi(n, arr, r.shape, "", None, r.color, r.path)     # a 2D ROI is no longer on one slice
                self.rois[n].saved, self.rois[n].visible = False, r.visible
        self.roi_selected = sel if sel in self.rois else (next(iter(self.rois), None))

    def clear_transform(self, name: str) -> None:
        self.transforms.pop(name, None)
        v = io.resample_to(self.native[name], self.geo)
        self.volumes[name] = v
        self.can[name] = self.orient.canonical(v.array)

    # ------------------------------------------------------------------ layers
    def _register(self, name: str, base: np.ndarray, path: str = "", reader: str = "") -> seg.MaskLayer:
        color = PALETTE[len(self.layers) % len(PALETTE)]
        layer = seg.MaskLayer(name, self.orient.canonical(base), color, path, reader)
        self.layers[name] = layer
        self._base[name] = base
        self.active = self.active or name
        return layer

    def add_layer_from_file(self, name: str, path: str | Path) -> seg.MaskLayer:
        arr, resampled = io.load_mask(path, self.geo)
        layer = self._register(name, arr, str(path))
        layer.saved = not resampled
        return layer

    def add_empty_layer(self, name: str, path: str = "", reader: str = "") -> seg.MaskLayer:
        return self._register(name, np.zeros(tuple(self.geo.size[::-1]), np.uint8), path, reader)

    def remove_layer(self, name: str) -> None:
        self.layers.pop(name, None)
        self._base.pop(name, None)
        if self.active == name:
            self.active = next(iter(self.layers), None)

    def start_reader(self, reader: str) -> tuple[str, str]:
        """Layer for a reader's correction: continue the reader's saved mask, else start from the
        automatic mask, else empty. With blinding, the other readers' layers are removed from the view."""
        if self.project is None or self.case is None:
            raise RuntimeError("open a project case to work as a reader")
        self.reader = reader
        if self.blind:
            for n in list(self.layers):
                if any(n.lower().endswith(o.lower()) for o in self.project.readers if o != reader):
                    self.remove_layer(n)
        source, rel = self.project.start_mask(self.case, reader)
        target = self.project.reader_mask_rel(self.case, reader)
        name = Path(target).name.split(".")[0]
        self.remove_layer(name)
        if rel:
            arr, _ = io.load_mask(self.project.path(rel), self.geo)
            layer = self._register(name, (arr > 0).astype(np.uint8), str(self.project.path(target)), reader)
        else:
            layer = self.add_empty_layer(name, str(self.project.path(target)), reader)
        layer.reader = reader
        layer.saved = source == "continue"
        self.active = name
        return name, source

    def layer(self) -> seg.MaskLayer | None:
        return self.layers.get(self.active) if self.active else None

    def unsaved(self) -> list[str]:
        return [n for n, l in self.layers.items() if not l.saved]

    def save_layer(self, name: str | None = None, path: str | None = None) -> dict:
        name = name or self.active
        layer = self.layers[name]
        path = path or layer.path
        if not path:
            raise ValueError("no file name for this layer")
        res = io.save_mask(self._base[name], self.geo, path, reader=layer.reader)
        layer.path, layer.saved = str(path), True
        return res

    # ------------------------------------------------------------------ editing (canonical indices)
    def export_images(self, folder: str | Path, on_reference_grid: bool = False) -> list[Path]:
        """Save every image of the case as NIfTI (<name>.nii.gz; PET in SUV). on_reference_grid=True
        writes all images on the grid of the reference image (normally PET), as shown in the viewer;
        False keeps each image on its own grid (CT at its original resolution)."""
        out = []
        for name in self.native:
            vol = self.volumes[name] if on_reference_grid else self.native[name]
            stem = re.sub(r"[^\w.-]+", "_", name).strip("_") or "image"
            if name in self.transforms and on_reference_grid:
                stem_extra = "_registered"
            else:
                stem_extra = ""
            if vol.kind == "PET" and vol.unit == "SUV" and "suv" not in stem.lower():
                stem += "_SUV"
            stem += stem_extra
            out.append(io.export_volume(vol, Path(folder) / f"{stem}.nii.gz"))
        return out

    def image_for_tools(self) -> np.ndarray:
        return self.can[self.pet_name or self.ref.name]

    def paint(self, view: str, index: int, row: int, col: int, radius_mm: float, value: int = 1,
              sphere: bool = False, low: float | None = None, high: float | None = None) -> int:
        layer = self.layer()
        if layer is None:
            raise RuntimeError("no active mask layer")
        c = plane_to_index(view, index, row, col)
        img = self.image_for_tools() if (low is not None or high is not None) else None
        region, box = seg.brush_region(self.shape, c, radius_mm, self.orient.spacing_xyz, view, sphere, img, low, high)
        return layer.apply(region, box, value)

    def duplicate_layer(self, name: str | None = None) -> str:
        name = name or self.active
        src = self.layers[name]
        new = f"{name}_copy"
        i = 2
        while new in self.layers:
            new, i = f"{name}_copy{i}", i + 1
        layer = self._register(new, self._base[name].copy(), "", "")
        layer.color = PALETTE[len(self.layers) % len(PALETTE)]
        layer.saved = False
        self.active = new
        return new

    def edit_active(self, fn) -> int:
        layer = self.layer()
        if layer is None:
            raise RuntimeError("no active mask layer")
        return fn(layer)

    def combine_with(self, other: str, op: str) -> int:
        layer = self.layer()
        if layer is None or other not in self.layers:
            raise RuntimeError("select an active layer and another layer")
        return seg.combine(layer, self.layers[other].array, op)

    def smooth(self, sigma_mm: float) -> int:
        layer = self.layer()
        if layer is None:
            raise RuntimeError("no active mask layer")
        return seg.smooth_mask(layer, sigma_mm, self.orient.spacing_xyz)

    def margin(self, margin_mm: float) -> int:
        layer = self.layer()
        if layer is None:
            raise RuntimeError("no active mask layer")
        return seg.margin_mask(layer, margin_mm, self.orient.spacing_xyz)

    def threshold(self, mode: str, low: float = 2.5, high: float | None = None, percent: float = 41.0) -> int:
        layer = self.layer()
        if layer is None:
            raise RuntimeError("no active mask layer")
        return seg.threshold_mask(layer, self.image_for_tools(), mode, low, high, percent)

    def grow(self, view: str, index: int, row: int, col: int, mode: str = "percent", value: float = 41.0) -> tuple[int, float]:
        layer = self.layer()
        if layer is None:
            raise RuntimeError("no active mask layer")
        c = plane_to_index(view, index, row, col)
        region, box, thr = seg.region_grow(self.image_for_tools(), c, self.orient.spacing_xyz, mode, value)
        return layer.apply(region, box, 1), thr

    def lesions(self, name: str | None = None) -> list[dict]:
        layer = self.layers[name or self.active]
        return seg.lesions(layer.array, self.image_for_tools(), self.orient.spacing_xyz)

    def value_at(self, a0: int, a1: int, a2: int) -> dict:
        return {n: float(c[a0, a1, a2]) for n, c in self.can.items()}
