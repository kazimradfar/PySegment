"""Synthetic but anatomically plausible PSMA patient for trying PySegment (NOT real data, NOT a finding).

Writes, for one patient, to a folder OUTSIDE data/ and results/:
  baseline/PET   DICOM, Modality PT, Bq/mL with injection data (SUV works), 4 x 4 x 4 mm, 68Ga-PSMA-11 like pattern
  baseline/CT    DICOM, Modality CT, HU, 2 x 2 x 3 mm
  followup/PET, followup/CT   the same patient later: re-positioned (rigid + smooth local deformation), lesions changed
  followup/SPECT DICOM NM multi-frame, counts, 4.8 mm, 177Lu-PSMA-like pattern (kidneys, salivary glands, lesions)
  truth/*.nii.gz  lesion masks and the true transform (for later validation only)
Usage: python tools/make_realistic_demo.py D:/public_demo/synthetic_pt01 [--seed 1]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk

SP_CT = (2.0, 2.0, 3.0)          # mm, (x, y, z)
SP_PET = (4.0, 4.0, 4.0)
SP_SPECT = (4.8, 4.8, 4.8)
NX, NY, NZ = 200, 140, 260       # CT grid -> 400 x 280 x 780 mm
ORIGIN = (-200.0, -140.0, 0.0)   # LPS mm of the first voxel (feet end z = 0, head at z = 780)

BODY, FAT, LUNG, BONE, MARROW, LIVER, SPLEEN, KIDNEY, BLADDER, HEART, SALIV, BRAIN, PROSTATE, BOWEL, MUSCLE = range(1, 16)
#             HU     68Ga SUV  177Lu relative uptake (arbitrary units)
TABLE = {0: (-1000, 0.0, 0.0), BODY: (40, 1.0, 0.5), FAT: (-100, 0.3, 0.2), LUNG: (-800, 0.3, 0.1),
         BONE: (900, 0.8, 0.3), MARROW: (250, 1.6, 0.6), LIVER: (60, 5.5, 3.0), SPLEEN: (50, 5.0, 2.5),
         KIDNEY: (32, 22.0, 18.0), BLADDER: (8, 24.0, 6.0), HEART: (45, 2.6, 1.2), SALIV: (45, 13.0, 9.0),
         BRAIN: (35, 0.5, 0.1), PROSTATE: (50, 3.0, 1.5), BOWEL: (30, 2.8, 1.6), MUSCLE: (55, 1.0, 0.5)}


def grids():
    z, y, x = np.mgrid[:NZ, :NY, :NX].astype(np.float32)
    return (ORIGIN[0] + x * SP_CT[0], ORIGIN[1] + y * SP_CT[1], ORIGIN[2] + z * SP_CT[2])


def ell(X, Y, Z, c, r):
    return ((X - c[0]) / r[0]) ** 2 + ((Y - c[1]) / r[1]) ** 2 + ((Z - c[2]) / r[2]) ** 2 <= 1.0


def anatomy():
    X, Y, Z = grids()
    L = np.zeros((NZ, NY, NX), np.uint8)
    # body: trunk (z 0-600) with a slight taper, neck, head
    taper = np.clip((Z - 120) / 80.0, 0, 1)                    # legs below the pelvis, joined above
    legs = sum((((X - s * 88) / 72) ** 2 + (Y / 78) ** 2 <= 1) for s in (-1, 1)) > 0
    trunk = np.where(Z < 120, legs, ((X / 165) ** 2 + (Y / (105 + 0.01 * (Z - 300))) ** 2 <= 1) | (legs & (Z < 200))) & (Z <= 610)
    neck = ((X / 58) ** 2 + (Y / 62) ** 2 <= 1) & (Z > 600) & (Z <= 665)
    head = ell(X, Y, Z, (0, 5, 725), (82, 98, 58)) | (((X / 78) ** 2 + ((Y - 5) / 92) ** 2 <= 1) & (Z > 660) & (Z < 725))
    body = trunk | neck | head
    L[body] = FAT
    legs_in = sum((((X - s * 88) / 60) ** 2 + (Y / 66) ** 2 <= 1) for s in (-1, 1)) > 0
    inner = np.where(Z < 120, legs_in, ((X / 148) ** 2 + (Y / (90 + 0.01 * (Z - 300))) ** 2 <= 1) | (legs_in & (Z < 200))) & (Z <= 610) | neck | head
    L[inner & body] = MUSCLE
    L[body & (Z > 660)] = BODY
    L[ell(X, Y, Z, (0, 5, 725), (68, 84, 45))] = BRAIN
    # skull, spine (posterior, y > 0), pelvis, ribs, sternum
    skull = ell(X, Y, Z, (0, 5, 725), (82, 98, 58)) & ~ell(X, Y, Z, (0, 5, 725), (72, 88, 48))
    L[skull] = BONE
    for zc in np.arange(40, 640, 26):                                        # vertebral bodies with marrow
        vb = ell(X, Y, Z, (0, 62 + 6 * np.sin(zc / 130.0), zc), (24, 20, 11))
        L[vb] = BONE
        L[ell(X, Y, Z, (0, 62 + 6 * np.sin(zc / 130.0), zc), (17, 14, 8))] = MARROW
    for s in (-1, 1):
        pel = ell(X, Y, Z, (s * 95, 40, 90), (42, 38, 78)) & ~ell(X, Y, Z, (s * 95, 40, 90), (34, 30, 70))
        L[pel] = BONE
        L[ell(X, Y, Z, (s * 95, 40, 90), (34, 30, 70))] = MARROW
        L[ell(X, Y, Z, (s * 88, 5, 40), (14, 14, 90))] = BONE                # femoral heads / proximal femora
        for zc in np.arange(430, 600, 22):                                    # ribs as thin shells
            rib = ell(X, Y, Z, (s * 5, 5, zc), (150, 92, 8)) & ~ell(X, Y, Z, (s * 5, 5, zc), (144, 86, 9))
            L[rib & (np.abs(X) > 60)] = BONE
        L[ell(X, Y, Z, (s * 90, -5, 560), (75, 70, 110)) & (Z > 440) & (L == MUSCLE)] = LUNG   # lungs
    L[ell(X, Y, Z, (0, -75, 540), (14, 6, 60))] = BONE                         # sternum
    L[ell(X, Y, Z, (-5, -35, 520), (48, 40, 45))] = HEART
    L[ell(X, Y, Z, (75, -10, 395), (58, 55, 52)) & (L != BONE)] = LIVER      # liver (patient right = x < 0? keep simple)
    L[ell(X, Y, Z, (-85, 40, 395), (28, 22, 40)) & (L != BONE)] = SPLEEN
    for s, zc in ((-1, 335), (1, 325)):
        L[ell(X, Y, Z, (s * 60, 52, zc), (22, 17, 45)) & (L != BONE)] = KIDNEY
    L[ell(X, Y, Z, (0, -45, 155), (40, 36, 36))] = BLADDER
    L[ell(X, Y, Z, (0, -5, 100), (23, 20, 18))] = PROSTATE
    for k in range(7):                                                          # bowel loops
        L[ell(X, Y, Z, (-70 + 25 * k, -50, 215 + 12 * (k % 3)), (20, 17, 25)) & (L == MUSCLE)] = BOWEL
    for s in (-1, 1):                                                           # parotid / submandibular glands
        L[ell(X, Y, Z, (s * 66, 25, 700), (13, 16, 22)) & (L == BODY)] = SALIV
        L[ell(X, Y, Z, (s * 36, -10, 668), (14, 12, 10)) & (L == BODY)] = SALIV
    return L, (X, Y, Z)


def lesions(rng, scale):
    """(name, centre mm, radius mm, 68Ga SUV, 177Lu uptake) : prostate primary, nodes, bone and liver mets."""
    base = [("prostate", (6, -4, 102), 9, 16.0, 9.0), ("node_iliac_L", (-60, -5, 178), 8, 12.0, 7.0),
            ("node_iliac_R", (58, 0, 172), 6, 9.0, 5.0), ("node_paraaortic", (-6, 20, 285), 7, 14.0, 8.0),
            ("node_mediastinal", (30, -10, 520), 6, 10.0, 6.0), ("bone_L3", (0, 65, 205), 9, 20.0, 10.0),
            ("bone_pelvis_R", (96, 40, 120), 12, 25.0, 12.0), ("bone_rib_L", (-150, 20, 480), 7, 15.0, 8.0),
            ("liver_met", (80, 0, 400), 14, 8.5, 4.0)]
    out = []
    for n, c, r, suv, lu in base:
        jit = rng.normal(0, 3, 3)
        out.append((n, tuple(np.array(c) + jit), r * scale.get(n, 1.0), suv * (0.7 + 0.6 * rng.random()), lu))
    return out


def fields(L, XYZ, les, with_lesions=True):
    X, Y, Z = XYZ
    hu = np.zeros(L.shape, np.float32)
    suv = np.zeros_like(hu)
    lu = np.zeros_like(hu)
    for k, (h, s, l) in TABLE.items():
        m = L == k
        hu[m], suv[m], lu[m] = h, s, l
    lab = np.zeros(L.shape, np.uint8)
    if with_lesions:
        for i, (n, c, r, s, l) in enumerate(les, start=1):
            m = ell(X, Y, Z, c, (r, r, r)) & (L > 0)
            lab[m] = i
            suv[m], lu[m] = s, l
            if n.startswith("bone"):
                hu[m] = np.maximum(hu[m], 350)
    return hu, suv, lu, lab


def to_img(arr, sp, origin=ORIGIN):
    im = sitk.GetImageFromArray(arr.astype(np.float32))
    im.SetSpacing(sp)
    im.SetOrigin(origin)
    return im


def smooth(im, fwhm):
    return sitk.SmoothingRecursiveGaussian(im, fwhm / 2.355)


def resample(im, sp, tfm=None, interp=sitk.sitkLinear, default=0.0):
    size = [int(round(NX * SP_CT[0] / sp[0])), int(round(NY * SP_CT[1] / sp[1])), int(round(NZ * SP_CT[2] / sp[2]))]
    return sitk.Resample(im, size, tfm or sitk.Transform(), interp, ORIGIN, sp, (1, 0, 0, 0, 1, 0, 0, 0, 1), default)


def write_dicom_series(arr, sp, folder: Path, modality: str, uid_study, uid_for, desc, extra, when: datetime, pid):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    folder.mkdir(parents=True, exist_ok=True)
    series = generate_uid()
    sop_class = {"PT": "1.2.840.10008.5.1.4.1.1.128", "CT": "1.2.840.10008.5.1.4.1.1.2"}[modality]
    slope = float(np.abs(arr).max()) / 30000.0 if modality == "PT" else 1.0
    for k in range(arr.shape[0]):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID = sop_class, generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(folder / f"{k:04d}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID, ds.Modality = sop_class, meta.MediaStorageSOPInstanceUID, modality
        ds.PatientName, ds.PatientID, ds.PatientSex = "SYNTHETIC^DEMO", pid, "M"
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = uid_study, series, uid_for
        ds.SeriesDescription = desc
        ds.StudyDate = ds.SeriesDate = ds.AcquisitionDate = when.strftime("%Y%m%d")
        ds.StudyTime = ds.SeriesTime = ds.AcquisitionTime = when.strftime("%H%M%S")
        ds.InstanceNumber = k + 1
        ds.ImagePositionPatient = [ORIGIN[0], ORIGIN[1], ORIGIN[2] + k * sp[2]]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [sp[1], sp[0]]
        ds.SliceThickness = sp[2]
        ds.Rows, ds.Columns = arr.shape[1], arr.shape[2]
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        if modality == "CT":
            ds.PixelRepresentation = 1
            ds.RescaleSlope, ds.RescaleIntercept = 1, 0
            ds.PixelData = np.clip(np.round(arr[k]), -2000, 30000).astype("<i2").tobytes()
        else:
            ds.PixelRepresentation = 0
            ds.RescaleSlope, ds.RescaleIntercept = slope, 0
            ds.PixelData = np.clip(np.round(arr[k] / slope), 0, 65535).astype("<u2").tobytes()
            for key, val in extra.items():
                setattr(ds, key, val)
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.save_as(str(folder / f"{k:04d}.dcm"), enforce_file_format=True)


def pet_extra(weight_kg, dose_mbq, uptake_min, when):
    from pydicom.dataset import Dataset
    inj = datetime.fromtimestamp(when.timestamp() - uptake_min * 60)
    rp = Dataset()
    rp.Radiopharmaceutical = "Ga-68 PSMA-11 (synthetic)"
    rp.RadionuclideTotalDose = dose_mbq * 1e6
    rp.RadionuclideHalfLife = 4062.6
    rp.RadiopharmaceuticalStartDateTime = inj.strftime("%Y%m%d%H%M%S")
    return {"Units": "BQML", "DecayCorrection": "START", "PatientWeight": weight_kg,
            "RadiopharmaceuticalInformationSequence": [rp]}


def write_nm(arr, sp, folder: Path, uid_study, uid_for, when, pid):
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    folder.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID = "1.2.840.10008.5.1.4.1.1.20", generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(folder / "spect.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID, ds.Modality = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID, "NM"
    ds.PatientName, ds.PatientID = "SYNTHETIC^DEMO", pid
    ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = uid_study, generate_uid(), uid_for
    ds.SeriesDescription = "Lu-177 SPECT recon (synthetic)"
    ds.StudyDate = ds.SeriesDate = when.strftime("%Y%m%d")
    ds.Rows, ds.Columns, ds.NumberOfFrames = arr.shape[1], arr.shape[2], arr.shape[0]
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit, ds.PixelRepresentation = 15, 0
    ds.PixelSpacing, ds.SpacingBetweenSlices = [sp[1], sp[0]], sp[2]
    ds.Units = "CNTS"
    det = Dataset()
    det.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    det.ImagePositionPatient = list(ORIGIN)
    ds.DetectorInformationSequence = [det]
    ds.PixelData = np.clip(np.round(arr), 0, 65535).astype("<u2").tobytes()
    ds.is_little_endian, ds.is_implicit_VR = True, False
    ds.save_as(str(folder / "spect.dcm"), enforce_file_format=True)


def followup_transform(rng):
    """Re-positioning between visits: rigid (a few degrees / cm) plus a smooth local deformation (breathing, bladder, bowel)."""
    rig = sitk.Euler3DTransform()
    rig.SetCenter((0.0, 0.0, 390.0))
    rig.SetRotation(*np.radians([2.5, -1.5, 3.0]))
    rig.SetTranslation((8.0, -6.0, 14.0))
    ref = to_img(np.zeros((NZ, NY, NX), np.float32), SP_CT)
    bs = sitk.BSplineTransformInitializer(ref, [6, 5, 9], order=3)
    p = np.zeros(len(bs.GetParameters()))
    p[:] = rng.uniform(-4.0, 4.0, p.shape)
    bs.SetParameters(p.tolist())
    comp = sitk.CompositeTransform(3)
    comp.AddTransform(rig)
    comp.AddTransform(bs)
    return rig, bs, comp


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    if any(p in ("data", "results") for p in out.parts):
        sys.exit("Refusing to write synthetic data inside a data/ or results/ folder; choose another folder.")
    rng = np.random.default_rng(a.seed)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    from pydicom.uid import generate_uid
    pid = f"SYNTH{a.seed:03d}"
    L, XYZ = anatomy()
    les0 = lesions(rng, {})
    les1 = [(n, c, r, s, l) for (n, c, r, s, l) in
            lesions(np.random.default_rng(a.seed), {"prostate": 0.55, "node_iliac_L": 0.5, "bone_L3": 1.15, "liver_met": 1.3})]
    les1 = [(n, c, r, s * (0.45 if n in ("prostate", "node_iliac_L") else 1.1), l) for n, c, r, s, l in les1]
    when0, when1 = datetime(2026, 1, 12, 10, 30), datetime(2026, 4, 20, 11, 0)
    study0, study1, for0, for1 = generate_uid(), generate_uid(), generate_uid(), generate_uid()
    rig, bs, comp = followup_transform(rng)
    for tag, les, tfm, when, st, fr in (("baseline", les0, None, when0, study0, for0), ("followup", les1, comp, when1, study1, for1)):
        hu, suv, lu, lab = fields(L, XYZ, les)
        hu_i, suv_i, lu_i, lab_i = (to_img(v, SP_CT) for v in (hu, suv, lu, lab))
        t = tfm        # follow-up image(x) = baseline(T(x)): T maps follow-up space to baseline space
        if t is not None:
            hu_i = sitk.Resample(hu_i, hu_i, t, sitk.sitkLinear, -1000.0)
            suv_i = sitk.Resample(suv_i, suv_i, t, sitk.sitkLinear, 0.0)
            lu_i = sitk.Resample(lu_i, lu_i, t, sitk.sitkLinear, 0.0)
            lab_i = sitk.Resample(lab_i, lab_i, t, sitk.sitkNearestNeighbor, 0.0)
        ct = sitk.GetArrayFromImage(smooth(hu_i, 1.2)) + rng.normal(0, 11, (NZ, NY, NX)).astype(np.float32)
        write_dicom_series(ct, SP_CT, out / tag / "CT", "CT", st, fr, "CT (synthetic)", {}, when, pid)
        sig = smooth(suv_i, 5.5)                                   # PET resolution ~ 5.5 mm FWHM
        pet_sp = resample(sig, SP_PET)
        pet = sitk.GetArrayFromImage(pet_sp)
        pet = pet * (1 + rng.normal(0, 0.07, pet.shape)) * (1 + 0.10 * np.clip(pet, 0, 6) / 6 * rng.normal(0, 1, pet.shape))
        pet = np.clip(pet, 0, None)
        weight, dose, uptake = 82.0, 148.0, 62.0
        decay = 2 ** (-uptake * 60 / 4062.6)
        bqml = pet * dose * 1e6 * decay / (weight * 1000.0)
        write_dicom_series(bqml, SP_PET, out / tag / "PET", "PT", st, fr, "PSMA PET (synthetic)",
                           pet_extra(weight, dose, uptake, when), when, pid)
        sitk.WriteImage(sitk.Cast(resample(lab_i, SP_PET, None, sitk.sitkNearestNeighbor), sitk.sitkUInt8),
                        str(out / "truth" / f"{tag}_lesion_labels_on_PET_grid.nii.gz"))
        if tag == "followup":
            spect = sitk.GetArrayFromImage(resample(smooth(lu_i, 11.0), SP_SPECT))
            counts = np.clip(spect / max(spect.max(), 1e-6) * 900.0, 0, None)
            counts = rng.poisson(counts).astype(np.float32)
            write_nm(counts, SP_SPECT, out / tag / "SPECT", st, fr, when, pid)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(comp, str(out / "truth" / "followup_true_transform.tfm"))
    (out / "README.txt").write_text(
        "Synthetic demo patient (random anatomy-like phantom). Not real data; numbers have no clinical meaning.\n"
        "Open baseline/PET and baseline/CT (or followup/*) in PySegment. SPECT: followup/SPECT.\n"
        "Registration test: fixed = FOLLOWUP CT, moving = BASELINE CT. truth/followup_true_transform.tfm maps follow-up\n"
        "points to baseline space, which is exactly what the registration returns for that choice (a transform always\n"
        "maps fixed-image points to moving-image points). Compare it at the follow-up lesion centroids for validation.\n")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
