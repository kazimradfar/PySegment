"""DICOM series reading and body-weight SUV for PET (QIBA / Kinahan & Fletcher 2010)."""
from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def list_series(folder: Path) -> list[dict]:
    """Image series under a folder (any depth, several series may share a folder): uid, modality, description,
    number of files, the files themselves. Headers only, so it is quick; non-image objects are skipped."""
    import pydicom
    found: dict = {}
    for f in sorted(Path(folder).rglob("*")):
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(f, force=True, stop_before_pixels=True)
        except Exception:  # noqa: BLE001 - not a DICOM file
            continue
        uid, mod = str(getattr(ds, "SeriesInstanceUID", "")), str(getattr(ds, "Modality", "")).upper()
        if not uid or mod in ("RTSTRUCT", "RTPLAN", "RTDOSE", "SR", "PR", "REG", "SEG", "KO"):
            continue
        if "ImagePositionPatient" not in ds and mod != "NM":
            continue
        it = [str(v).upper() for v in (getattr(ds, "ImageType", []) or [])]
        planar = mod == "NM" and not any("TOMO" in v for v in it)
        e = found.setdefault(uid, {"uid": uid, "modality": mod, "description": str(getattr(ds, "SeriesDescription", "")),
                                   "planar": planar, "files": []})
        e["files"].append(f)
    out = list(found.values())
    for e in out:
        e["n"] = len(e["files"])
    return sorted(out, key=lambda e: ({"PT": 0, "NM": 1, "CT": 2, "MR": 3}.get(e["modality"], 9), -e["n"]))


def read_dicom_series(folder: Path, files: list | None = None) -> tuple[sitk.Image, list]:
    """Read a single-series DICOM folder (or the given files of one series) into a float32 image, slices sorted
    along the normal and the per-slice RescaleSlope / RescaleIntercept applied (PET slices often have different
    slopes)."""
    import pydicom
    datasets = []
    for f in (files if files is not None else sorted(Path(folder).rglob("*"))):
        if not Path(f).is_file():
            continue
        try:
            ds = pydicom.dcmread(f, force=True)
        except Exception:  # noqa: BLE001 - not a DICOM file
            continue
        if "PixelData" in ds and "ImagePositionPatient" in ds:
            datasets.append(ds)
    if not datasets:
        raise FileNotFoundError(f"No DICOM images found in {folder}")
    uids = {getattr(ds, "SeriesInstanceUID", "") for ds in datasets}
    if len(uids) > 1:
        raise ValueError(f"{folder} contains {len(uids)} series; keep one series per folder")
    row = np.array(datasets[0].ImageOrientationPatient[:3], float)
    col = np.array(datasets[0].ImageOrientationPatient[3:], float)
    normal = np.cross(row, col)
    datasets.sort(key=lambda d: float(np.dot(normal, np.array(d.ImagePositionPatient, float))))
    vol = np.stack([d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1.0))
                    + float(getattr(d, "RescaleIntercept", 0.0)) for d in datasets], axis=0)
    pos = [np.array(d.ImagePositionPatient, float) for d in datasets]
    if len(pos) > 1:
        gaps = np.diff([np.dot(normal, p) for p in pos])
        dz = float(np.median(gaps))
        if np.ptp(gaps) > 0.01 * abs(dz):
            warnings.warn(f"Non-uniform slice spacing in {folder}")
    else:
        dz = float(getattr(datasets[0], "SliceThickness", 1.0))
    px = [float(v) for v in datasets[0].PixelSpacing]
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing((px[1], px[0], dz))
    img.SetOrigin(tuple(pos[0]))
    img.SetDirection(tuple(np.column_stack([row, col, normal]).ravel()))
    return img, datasets


def _dt(date, time) -> datetime | None:
    if not date or not time:
        return None
    return datetime.strptime(f"{date}{str(time).split('.')[0].ljust(6, '0')}", "%Y%m%d%H%M%S")


def _full_dt(value) -> datetime | None:
    if not value:
        return None
    value = str(value).split(".")[0].split("+")[0].split("-")[0]
    return datetime.strptime(value[:14].ljust(14, "0"), "%Y%m%d%H%M%S")


def suv_bw_factor(datasets: list) -> tuple[float, dict]:
    """Factor converting Bq/mL to body-weight SUV and the values used:
    SUVbw = C * weight(g) / (injected activity * 2^(-uptake / T1/2))."""
    ds = datasets[0]
    if str(getattr(ds, "Units", "")).upper() != "BQML":
        raise ValueError(f"PET units are '{getattr(ds, 'Units', '')}', expected BQML")
    if str(getattr(ds, "DecayCorrection", "START")).upper() != "START":
        raise ValueError("Only DecayCorrection = START is supported")
    rp = ds.RadiopharmaceuticalInformationSequence[0]
    dose, half_life, weight = float(rp.RadionuclideTotalDose), float(rp.RadionuclideHalfLife), float(ds.PatientWeight)
    injection = _full_dt(getattr(rp, "RadiopharmaceuticalStartDateTime", None)) or _dt(
        ds.SeriesDate, getattr(rp, "RadiopharmaceuticalStartTime", None))
    acq = [t for t in (_dt(getattr(d, "AcquisitionDate", None) or d.SeriesDate, getattr(d, "AcquisitionTime", None))
                       for d in datasets) if t]
    start = min(acq) if acq else _dt(ds.SeriesDate, ds.SeriesTime)
    series = _dt(ds.SeriesDate, ds.SeriesTime)
    if series and start and series < start:
        start = series
    if injection is None or start is None:
        raise ValueError("Could not determine injection or scan time")
    dt = (start - injection).total_seconds()
    factor = weight * 1000.0 / (dose * 2 ** (-dt / half_life))
    return factor, {"injected_MBq": dose / 1e6, "weight_kg": weight, "uptake_min": dt / 60.0,
                    "half_life_s": half_life, "dose_at_scan_MBq": dose / 1e6 * 2 ** (-dt / half_life),
                    "radiopharmaceutical": str(getattr(rp, "Radiopharmaceutical", ""))}


def dicom_modality(datasets: list) -> str:
    return str(getattr(datasets[0], "Modality", "")).upper()


def peek_modality(folder: Path) -> str:
    """Modality of the first readable DICOM image under ``folder`` ('' if none)."""
    import pydicom
    for f in sorted(Path(folder).rglob("*")):
        if f.is_file():
            try:
                ds = pydicom.dcmread(f, force=True, stop_before_pixels=True)
            except Exception:  # noqa: BLE001
                continue
            if getattr(ds, "Modality", ""):
                return str(ds.Modality).upper()
    return ""


def _nm_geometry(ds):
    """Orientation, first-slice position, pixel spacing and slice step of an NM multi-frame image. The geometry sits
    in the top-level attributes or in the DetectorInformationSequence (common for SPECT)."""
    src = ds
    if "ImageOrientationPatient" not in ds or "ImagePositionPatient" not in ds:
        seq = getattr(ds, "DetectorInformationSequence", None)
        if not seq:
            raise ValueError("NM image without orientation/position (needs ImageOrientationPatient and "
                             "ImagePositionPatient or a DetectorInformationSequence)")
        src = seq[0]
    iop = [float(v) for v in src.ImageOrientationPatient]
    ipp = np.array(src.ImagePositionPatient, float)
    px = [float(v) for v in getattr(ds, "PixelSpacing", None) or src.PixelSpacing]
    dz = getattr(ds, "SpacingBetweenSlices", None) or getattr(src, "SpacingBetweenSlices", None) or \
        getattr(ds, "SliceThickness", None) or getattr(src, "SliceThickness", None)
    if dz is None:
        raise ValueError("NM image without slice spacing (SpacingBetweenSlices / SliceThickness)")
    return iop, ipp, px, abs(float(dz))


def read_nm_series(folder: Path, files: list | None = None) -> tuple[sitk.Image, list]:
    """Read a reconstructed SPECT (Modality NM) image: one multi-frame file. Pixel values are kept as stored times
    RescaleSlope/Intercept (counts, or Bq/mL if the Units say so); no SUV is computed because that needs a
    calibrated, quantitative reconstruction that the file alone cannot guarantee."""
    import pydicom
    found = []
    for f in (files if files is not None else sorted(Path(folder).rglob("*"))):
        if Path(f).is_file():
            try:
                ds = pydicom.dcmread(f, force=True)
            except Exception:  # noqa: BLE001
                continue
            if "PixelData" in ds and str(getattr(ds, "Modality", "")).upper() == "NM":
                found.append(ds)
    if len(found) != 1:
        raise ValueError(f"{folder}: expected one multi-frame NM file, found {len(found)}")
    ds = found[0]
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim != 3:
        raise ValueError("NM image is not a 3D volume (static or single-frame image)")
    arr = arr * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
    iop, ipp, px, dz = _nm_geometry(ds)
    row, col = np.array(iop[:3]), np.array(iop[3:])
    normal = np.cross(row, col)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((px[1], px[0], dz))
    img.SetOrigin(tuple(ipp))
    img.SetDirection(tuple(np.column_stack([row, col, normal]).ravel()))
    return img, [ds]
