"""Images and masks: loading (NIfTI, NRRD, MHA, DICOM series with PET SUV conversion), resampling to a
reference grid, and saving masks with exactly the reference geometry (with backup and history)."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk

IMAGE_SUFFIXES = (".nii.gz", ".nii", ".nrrd", ".nhdr", ".mha", ".mhd")


@dataclass
class Geometry:
    size: tuple          # (x, y, z) voxels
    spacing: tuple       # (x, y, z) mm
    origin: tuple
    direction: tuple

    @classmethod
    def of(cls, img: sitk.Image) -> "Geometry":
        return cls(tuple(img.GetSize()), tuple(img.GetSpacing()), tuple(img.GetOrigin()), tuple(img.GetDirection()))

    def apply(self, img: sitk.Image) -> sitk.Image:
        img.SetSpacing(self.spacing)
        img.SetOrigin(self.origin)
        img.SetDirection(self.direction)
        return img

    def reference(self) -> sitk.Image:
        ref = sitk.Image([int(v) for v in self.size], sitk.sitkUInt8)
        return self.apply(ref)

    def same_grid(self, other: "Geometry", tol: float = 1e-3) -> bool:
        return (tuple(self.size) == tuple(other.size)
                and np.allclose(self.spacing, other.spacing, atol=tol)
                and np.allclose(self.origin, other.origin, atol=tol)
                and np.allclose(self.direction, other.direction, atol=1e-4))

    @property
    def voxel_ml(self) -> float:
        return float(np.prod(self.spacing)) / 1000.0


@dataclass
class Volume:
    name: str
    array: np.ndarray            # (z, y, x) float32
    geometry: Geometry
    kind: str = "other"          # "PET", "CT", "MR", "other"
    unit: str = ""               # "SUV", "HU", ...
    path: str = ""
    info: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.array.shape


def guess_kind(name: str) -> str:
    n = name.lower()
    if "spect" in n or n.startswith("nm") or "_nm" in n:
        return "SPECT"
    if "suv" in n or "pet" in n or n.startswith("pt"):
        return "PET"
    if n.startswith("ct") or "_ct" in n or n.endswith("ct") or "ct." in n:
        return "CT"
    if any(t in n for t in ("t1", "t2", "flair", "adc", "dwi", "mr")):
        return "MR"
    return "other"


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.name.lower().endswith(IMAGE_SUFFIXES)


def stem(p: Path) -> str:
    name = p.name
    for s in IMAGE_SUFFIXES:
        if name.lower().endswith(s):
            return name[: -len(s)]
    return p.stem


def _peek_files(files) -> str:
    import pydicom
    for f in files:
        try:
            return str(pydicom.dcmread(f, force=True, stop_before_pixels=True).Modality).upper()
        except Exception:  # noqa: BLE001
            continue
    return ""


def is_planar_nm(ds) -> bool:
    """Planar NM image (static or whole-body anterior/posterior views), not a reconstructed SPECT volume."""
    it = [str(v).upper() for v in (getattr(ds, "ImageType", []) or [])]
    return str(getattr(ds, "Modality", "")).upper() == "NM" and not any("TOMO" in v for v in it)


def load_planar(files) -> list:
    """Planar NM (one frame per view, e.g. whole-body ANT and POST) as thin coronal 'volumes' of shape (rows, 1,
    cols), so that the coronal view shows them head up; the posterior view is mirrored left-right (direction -x),
    so both views are shown in the same anatomical orientation, as for conjugate-view quantification."""
    import pydicom
    out = []
    for f in files:
        ds = pydicom.dcmread(f, force=True)
        if "PixelData" not in ds or not is_planar_nm(ds):
            continue
        a = ds.pixel_array.astype(np.float32)
        a = a[None] if a.ndim == 2 else a
        a = a * float(getattr(ds, "RescaleSlope", 1.0) or 1.0) + float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        det = list(getattr(ds, "DetectorInformationSequence", []) or [])
        sp = [float(v) for v in getattr(ds, "PixelSpacing", [1.0, 1.0])]
        for k, frame in enumerate(a):
            view = ""
            if k < len(det) and "ViewCodeSequence" in det[k]:
                view = str(det[k].ViewCodeSequence[0].get("CodeMeaning", ""))
            name = "NM POST" if "poster" in view.lower() else "NM ANT" if "anter" in view.lower() else f"NM frame {k + 1}"
            img = sitk.GetImageFromArray(frame[:, None, :].copy())
            img.SetSpacing((sp[1], 1.0, sp[0]))
            img.SetDirection((-1.0 if name == "NM POST" else 1.0, 0, 0, 0, 1, 0, 0, 0, -1))
            if name == "NM POST":                       # mirrored view covers the same physical field as ANT
                img.SetOrigin(((frame.shape[1] - 1) * sp[1], 0.0, 0.0))
            info = {"planar": True, "view": view or f"frame {k + 1}", "frame_duration_s":
                    float(getattr(ds, "ActualFrameDuration", 0) or 0) / 1000.0}
            out.append(Volume(name, sitk.GetArrayFromImage(img).astype(np.float32), Geometry.of(img), "SPECT",
                              "counts", str(f), info))
    if not out:
        raise ValueError("no planar NM image found")
    return out


def load_volume(path: str | Path, kind: str | None = None, name: str | None = None, files: list | None = None) -> Volume:
    """Load an image file or a DICOM series folder as float32. PET DICOM in Bq/mL is converted to
    body-weight SUV; a file whose name contains 'SUV' is taken as SUV already."""
    path = Path(path)
    info: dict = {}
    if path.is_dir():
        from rsw.suv import dicom_modality, peek_modality, read_dicom_series, read_nm_series, suv_bw_factor
        nm = (peek_modality(path) if files is None else _peek_files(files)) == "NM"
        img, dss = read_nm_series(path, files) if nm else read_dicom_series(path, files)
        mod = dicom_modality(dss)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        if mod == "PT":
            factor, info = suv_bw_factor(dss)
            arr *= factor
            kind, unit = "PET", "SUV"
        elif mod == "CT":
            kind, unit = "CT", "HU"
        elif mod == "NM":
            # SPECT: counts (or Bq/mL when stated); never converted to SUV here
            kind, unit = "SPECT", str(getattr(dss[0], "Units", "") or "counts")
            d0 = dss[0]
            if getattr(d0, "ActualFrameDuration", None):          # ms per projection, as stated in the file
                info["frame_duration_s"] = float(d0.ActualFrameDuration) / 1000.0
            for kw in ("NumberOfFramesInRotation", "NumberOfRotations"):
                if getattr(d0, kw, None):
                    info[kw] = int(d0[kw].value)
        else:
            kind, unit = (kind or ("MR" if mod == "MR" else "other")), ""
        geo = Geometry.of(img)
    else:
        img = sitk.ReadImage(str(path))
        if img.GetDimension() != 3 or img.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError(f"{path.name}: only 3D single-channel images are supported")
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        geo = Geometry.of(img)
        kind = kind or guess_kind(path.name)
        unit = {"PET": "SUV" if "suv" in path.name.lower() else "", "CT": "HU"}.get(kind, "")
    return Volume(name or (path.name if path.is_dir() else stem(path)), arr, geo, kind, unit, str(path), info)


def is_counts(unit: str) -> bool:
    """Per-voxel counts (not a concentration): resampling must conserve the sum, so values are scaled by the voxel volume ratio."""
    return str(unit).strip().upper() in ("CNTS", "COUNTS", "CPS")


def resample_to(vol: Volume, ref: Geometry, linear: bool = True, default: float | None = None,
                transform: sitk.Transform | None = None) -> Volume:
    """Volume on the reference grid (for fusion display); returns vol unchanged if already on it and no transform.
    ``transform`` (optional) maps reference-grid points to the volume's space, as SimpleITK's resample expects
    (the result of rsw.registration.register(fixed=reference, moving=vol)); None means identity."""
    if transform is None and vol.geometry.same_grid(ref):
        return vol
    src = vol.geometry.apply(sitk.GetImageFromArray(vol.array))
    fill = float(vol.array.min()) if default is None else default
    out = sitk.Resample(src, ref.reference(), transform if transform is not None else sitk.Transform(),
                        sitk.sitkLinear if linear else sitk.sitkNearestNeighbor, fill, sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(out)
    if is_counts(vol.unit):
        arr = arr * np.float32(ref.voxel_ml / vol.geometry.voxel_ml)      # counts per voxel: conserve the total
        if transform is not None:
            arr = arr * jacobian_on(ref, transform)                        # local volume change of the transform
    info = dict(vol.info)
    if transform is not None:
        info["registered"] = True
    return Volume(vol.name, arr, ref, vol.kind, vol.unit, vol.path, info)


def jacobian_on(ref: Geometry, transform: sitk.Transform):
    """Determinant of the Jacobian of ``transform`` (reference grid -> image space) on the reference grid.
    A voxel of the reference grid covers J times its volume in the image, so counts per voxel are multiplied by J
    to keep the total; concentrations (SUV, Bq/mL) are not multiplied. Constant for a linear transform."""
    from rsw.registration import transform_matrix
    m = transform_matrix(transform)
    if m is not None:
        return np.float32(abs(np.linalg.det(m[:3, :3])))
    r = ref.reference()
    disp = sitk.TransformToDisplacementField(transform, sitk.sitkVectorFloat64, r.GetSize(), r.GetOrigin(),
                                             r.GetSpacing(), r.GetDirection())
    return sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(disp)).astype(np.float32)


def export_volume(vol: Volume, path: str | Path) -> Path:
    """Write an image (for example a DICOM series read and converted to SUV) as NIfTI with its geometry.
    CT is stored as 16-bit integers (HU), everything else as 32-bit floats (PET in SUV)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(vol.array)
    if vol.kind == "CT" and np.all(np.isfinite(arr)) and arr.min() >= -32768 and arr.max() <= 32767:
        img = sitk.GetImageFromArray(np.round(arr).astype(np.int16))
    else:
        img = sitk.GetImageFromArray(arr.astype(np.float32))
    sitk.WriteImage(vol.geometry.apply(img), str(path), True)
    return path


def load_mask(path: str | Path, ref: Geometry, transform: sitk.Transform | None = None) -> tuple[np.ndarray, bool]:
    """Mask as uint8 on the reference grid (nearest neighbour, labels are never interpolated). ``transform`` as in
    resample_to; a mask drawn on a registered image is brought to the reference through it. Returns (array, resampled?)."""
    img = sitk.ReadImage(str(path))
    geo = Geometry.of(img)
    if transform is None and geo.same_grid(ref):
        return sitk.GetArrayFromImage(img).astype(np.uint8), False
    out = sitk.Resample(img, ref.reference(), transform if transform is not None else sitk.Transform(),
                        sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(out).astype(np.uint8), True


def backup_existing(path: Path, now: datetime | None = None) -> Path | None:
    if not path.exists():
        return None
    st = stem(path)
    dest = path.parent / "backup" / f"{st}_{(now or datetime.now()).strftime('%Y%m%d_%H%M%S')}{path.name[len(st):]}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(dest))
    return dest


def save_mask(array: np.ndarray, ref: Geometry, path: str | Path, reader: str = "", note: str = "") -> dict:
    """Write a mask (uint8; label values kept) with exactly the reference geometry. An existing file is
    moved to backup/, and every save is logged in mask_history.jsonl next to the mask."""
    path = Path(path)
    arr = np.asarray(array).astype(np.uint8)
    if tuple(arr.shape[::-1]) != tuple(ref.size):
        raise ValueError(f"mask grid {arr.shape[::-1]} differs from the reference grid {ref.size}")
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(path)
    img = ref.apply(sitk.GetImageFromArray(arr))
    sitk.WriteImage(img, str(path), True)
    rec = {"time": datetime.now().isoformat(timespec="seconds"), "file": path.name, "reader": reader,
           "voxels": int((arr > 0).sum()), "volume_ml": round(float((arr > 0).sum()) * ref.voxel_ml, 3),
           "note": note}
    with open(path.parent / "mask_history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return {**rec, "path": str(path), "backup": str(backup) if backup else ""}
