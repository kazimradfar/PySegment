"""DICOM helpers for a complete workflow in one program: header summary, 'is SUV possible?' check, anonymisation
that keeps everything SUV needs (weight, dose, half-life, times) and shifts all dates by one common offset if asked."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

HEADER_KEYWORDS = [
    "Modality", "SeriesDescription", "StudyDate", "AcquisitionDate", "AcquisitionTime", "Manufacturer",
    "ManufacturerModelName", "Units", "DecayCorrection", "CorrectedImage", "ReconstructionMethod", "Rows", "Columns",
    "PixelSpacing", "SliceThickness", "SpacingBetweenSlices", "NumberOfFrames", "PatientWeight", "PatientSize", "PatientSex",
    "RescaleSlope", "RescaleIntercept", "KVP", "ConvolutionKernel", "ImageOrientationPatient", "ImagePositionPatient",
    "FrameOfReferenceUID", "SeriesInstanceUID"]
RADIO_KEYWORDS = ["Radiopharmaceutical", "RadionuclideTotalDose", "RadionuclideHalfLife",
                  "RadiopharmaceuticalStartDateTime", "RadiopharmaceuticalStartTime"]


def _files(folder: Path):
    for f in sorted(Path(folder).rglob("*")):
        if f.is_file():
            yield f


def _read(f, pixels=False):
    import pydicom
    try:
        ds = pydicom.dcmread(f, force=True, stop_before_pixels=not pixels)
    except Exception:  # noqa: BLE001
        return None
    return ds if ("SOPClassUID" in ds or "Modality" in ds) else None


def list_series(folder) -> list[dict]:
    """One row per series: modality, description, number of files, date."""
    rows: dict[str, dict] = {}
    for f in _files(folder):
        ds = _read(f)
        if ds is None:
            continue
        uid = str(getattr(ds, "SeriesInstanceUID", f.parent))
        r = rows.setdefault(uid, {"series_uid": uid, "modality": str(getattr(ds, "Modality", "")),
                                  "description": str(getattr(ds, "SeriesDescription", "")),
                                  "date": str(getattr(ds, "StudyDate", "")), "files": 0, "folder": str(f.parent)})
        r["files"] += 1
    return list(rows.values())


def read_header(folder_or_file) -> list[tuple[str, str]]:
    """Key technical tags of the first image (patient name and ID are deliberately not listed)."""
    p = Path(folder_or_file)
    first = p if p.is_file() else next((f for f in _files(p) if _read(f) is not None), None)
    if first is None:
        raise FileNotFoundError(f"No DICOM image found in {p}")
    ds = _read(first)
    out = [("File", first.name)]
    for kw in HEADER_KEYWORDS:
        if kw in ds:
            out.append((kw, str(ds.data_element(kw).value)))
    seq = getattr(ds, "RadiopharmaceuticalInformationSequence", None)
    if seq:
        for kw in RADIO_KEYWORDS:
            if kw in seq[0]:
                out.append((kw, str(seq[0].data_element(kw).value)))
    return out


def suv_readiness(folder) -> tuple[bool, str]:
    """Can body-weight SUV be computed from this PET series? Returns (ok, explanation)."""
    from rsw.suv import read_dicom_series, suv_bw_factor
    try:
        _, dss = read_dicom_series(Path(folder))
        factor, info = suv_bw_factor(dss)
    except Exception as e:  # noqa: BLE001
        return False, f"SUV not possible: {e}"
    return True, (f"SUV possible: dose {info['injected_MBq']:.1f} MBq, weight {info['weight_kg']:.1f} kg, "
                  f"uptake time {info['uptake_min']:.0f} min (factor {factor:.4g})")


_NAME_TAGS = ["PatientName", "PatientID", "OtherPatientIDs", "OtherPatientNames", "PatientBirthDate", "PatientBirthTime",
              "PatientAddress", "PatientTelephoneNumbers", "PatientMotherBirthName", "InstitutionName", "InstitutionAddress",
              "ReferringPhysicianName", "PerformingPhysicianName", "OperatorsName", "PhysiciansOfRecord",
              "NameOfPhysiciansReadingStudy", "RequestingPhysician", "AccessionNumber", "StudyID", "InstitutionalDepartmentName",
              "StationName", "DeviceSerialNumber"]
_UID_TAGS = ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "FrameOfReferenceUID", "MediaStorageSOPInstanceUID"]


def _shift_date(value: str, days: int) -> str:
    if not value or len(value) < 8 or not value[:8].isdigit():
        return value
    d = datetime.strptime(value[:8], "%Y%m%d") + timedelta(days=days)
    return d.strftime("%Y%m%d") + value[8:]


def anonymize_folder(src, dst, code: str, shift_days: int | None = None, keep_private: bool = False) -> dict:
    """Copy every DICOM file under ``src`` to ``dst`` (same relative layout) without patient identity.
      * names, IDs, birth date, address, institution, physicians, accession and study number are replaced by ``code``
        or emptied; private tags are removed unless keep_private;
      * UIDs are replaced consistently (images of one series stay one series, frame of reference stays shared);
      * weight, dose, half-life and all times stay, so SUV can still be computed; if shift_days is given every date is
        moved by that many days (time differences are unchanged);
      * burned-in text inside the pixels cannot be removed: check the images before sharing.
    Returns counts."""
    import pydicom
    from pydicom.uid import generate_uid
    src, dst = Path(src), Path(dst)
    uid_map: dict[str, str] = {}
    n_ok = n_skip = 0

    def remap(u):
        return uid_map.setdefault(str(u), str(generate_uid()))
    for f in _files(src):
        ds = _read(f, pixels=True)
        if ds is None:
            n_skip += 1
            continue
        for kw in _NAME_TAGS:
            if kw in ds:
                el = ds.data_element(kw)
                el.value = code if kw in ("PatientName", "PatientID") else ("" if el.VR != "SQ" else el.value)
        if not keep_private:
            ds.remove_private_tags()
        for kw in _UID_TAGS:
            if kw in ds:
                ds.data_element(kw).value = remap(ds.data_element(kw).value)
        if hasattr(ds, "file_meta") and "MediaStorageSOPInstanceUID" in ds.file_meta and "SOPInstanceUID" in ds:
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        if shift_days:
            for el in ds.iterall():
                if el.VR in ("DA", "DT") and el.value:
                    el.value = _shift_date(str(el.value), shift_days)
        ds.PatientIdentityRemoved = "YES"
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        ds.save_as(str(out), enforce_file_format=True)
        n_ok += 1
    return {"written": n_ok, "skipped_non_dicom": n_skip, "uids_replaced": len(uid_map)}
