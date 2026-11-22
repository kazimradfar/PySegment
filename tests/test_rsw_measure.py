"""ROI statistics, line profiles, DICOM header / SUV check / anonymisation (synthetic data only)."""
import numpy as np
import pytest

from rsw import dicomtools, measure


def sphere_image(sp=(2.0, 2.0, 2.0), shape=(40, 40, 40)):
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    img = np.full(shape, 1.0, np.float32)
    ball = ((z - 20) * sp[0]) ** 2 + ((y - 20) * sp[1]) ** 2 + ((x - 20) * sp[2]) ** 2 <= 10.0 ** 2
    img[ball] = 10.0
    return img, ball


def test_roi_stats_values_and_volume():
    img, ball = sphere_image()
    st = measure.roi_stats(img, ball, (2.0, 2.0, 2.0))
    assert st["mean"] == pytest.approx(10.0) and st["max"] == 10.0 and st["sd"] == pytest.approx(0.0)
    assert st["volume_ml"] == pytest.approx(ball.sum() * 8e-3)
    assert st["volume_ml"] == pytest.approx(4 / 3 * np.pi * 1.0 ** 3, rel=0.12)         # r = 10 mm = 1 cm -> 4.19 mL
    assert st["total"] == pytest.approx(10.0 * st["volume_ml"])
    assert st["peak"] == pytest.approx(10.0, rel=0.05)                                  # 1 mL sphere fits inside the 4.2 mL ball


def test_suv_peak_is_lower_than_max_for_a_hot_voxel():
    img = np.ones((30, 30, 30), np.float32)
    img[15, 15, 15] = 100.0
    region = np.zeros(img.shape, bool)
    region[13:18, 13:18, 13:18] = True
    st = measure.roi_stats(img, region, (2.0, 2.0, 2.0))
    assert st["max"] == 100.0 and 1.0 < st["peak"] < 100.0 / 20                       # one voxel spread over the ~1 mL sphere


def test_percist_threshold():
    assert measure.percist_threshold(2.0, 0.3) == pytest.approx(3.6)


def test_line_profile_and_fwhm_of_gaussian():
    sp = (1.0, 1.0, 1.0)
    x = np.arange(101)
    img = np.exp(-0.5 * ((x - 50) / 5.0) ** 2)[None, None, :] * np.ones((3, 3, 1))
    d, v = measure.line_profile(img, (1, 1, 0), (1, 1, 100), sp)
    assert d[-1] == pytest.approx(100.0) and v.max() == pytest.approx(1.0, abs=1e-3)
    assert measure.fwhm(d, v) == pytest.approx(2.355 * 5.0, rel=0.02)
    assert np.isnan(measure.fwhm(d, np.ones_like(v)))


def make_series(folder):
    from datetime import datetime
    from tools import make_realistic_demo as m
    from pydicom.uid import generate_uid
    arr = (np.random.default_rng(0).random((5, 16, 16)) * 2e4).astype(np.float32)
    when = datetime(2026, 3, 3, 9, 0)
    m.write_dicom_series(arr, (4.0, 4.0, 4.0), folder, "PT", generate_uid(), generate_uid(), "PSMA PET",
                         m.pet_extra(80.0, 150.0, 60.0, when), when, "REALNAME123")
    return arr


def test_header_series_and_suv_readiness(tmp_path):
    make_series(tmp_path / "pet")
    rows = dicomtools.list_series(tmp_path)
    assert len(rows) == 1 and rows[0]["modality"] == "PT" and rows[0]["files"] == 5
    hdr = dict(dicomtools.read_header(tmp_path / "pet"))
    assert hdr["Units"] == "BQML" and hdr["PatientWeight"] == "80.0" and "RadionuclideTotalDose" in hdr
    assert "REALNAME123" not in " ".join(hdr.values())                                   # identity is not shown
    ok, msg = dicomtools.suv_readiness(tmp_path / "pet")
    assert ok and "150.0 MBq" in msg


def test_anonymise_keeps_suv_and_removes_identity(tmp_path):
    import pydicom
    from rsw.suv import read_dicom_series, suv_bw_factor
    make_series(tmp_path / "pet")
    for f in (tmp_path / "pet").glob("*.dcm"):                    # add identifying fields to every file
        ds = pydicom.dcmread(f)
        ds.InstitutionName, ds.ReferringPhysicianName, ds.AccessionNumber = "Some Hospital", "Dr X", "ACC77"
        ds.add_new(0x00091001, "LO", "private identity")
        ds.save_as(str(f), enforce_file_format=True)
    res = dicomtools.anonymize_folder(tmp_path / "pet", tmp_path / "anon", "ANON01", shift_days=100)
    assert res["written"] == 5 and res["uids_replaced"] >= 3
    first = pydicom.dcmread(next((tmp_path / "anon").glob("*.dcm")))
    assert str(first.PatientName) == "ANON01" and first.PatientID == "ANON01" and first.InstitutionName == ""
    assert first.AccessionNumber == "" and 0x00091001 not in first and first.PatientIdentityRemoved == "YES"
    assert first.StudyDate != "20260303"                                                  # dates shifted
    _, d0 = read_dicom_series(tmp_path / "pet")
    _, d1 = read_dicom_series(tmp_path / "anon")                                          # still ONE series, still readable
    f0, _ = suv_bw_factor(d0)
    f1, _ = suv_bw_factor(d1)
    assert f0 == pytest.approx(f1)                                                        # SUV unchanged: times shifted together
    assert len({str(d.SeriesInstanceUID) for d in d1}) == 1 and str(d1[0].SeriesInstanceUID) != str(d0[0].SeriesInstanceUID)


def test_session_roi_and_filter():
    from rsw import io
    from rsw.session import Session
    img, ball = sphere_image()
    geo = io.Geometry((40, 40, 40), (2.0, 2.0, 2.0), (0.0, 0.0, 0.0), (1, 0, 0, 0, 1, 0, 0, 0, 1))
    v = io.Volume("PET", img, geo, "PET", "SUV")
    s = Session(None, None, [v])
    region, st = s.roi_at(tuple(int(i) for i in np.array(np.nonzero(ball)).mean(axis=1).round()), 8.0)
    assert st["mean"] == pytest.approx(10.0) and st["voxels"] > 100
    name = s.add_filtered("PET", 4.0)
    assert name in s.can and s.can[name].max() < 10.0 and s.can[name].shape == s.can["PET"].shape
    assert s.can[name].sum() == pytest.approx(s.can["PET"].sum(), rel=0.02)         # smoothing keeps the total counts


def test_roi_table_counts_to_activity_suv_and_bqml():
    import pandas as pd
    base = {"kind": "roi", "image": "x", "radius_mm": 15, "volume_ml": 2.0, "voxels": 10, "sd": 1.0, "min": 1.0, "max": 9.0,
            "median": 5.0, "peak": 8.0}
    spect = dict(base, unit="CNTS", mean=500.0, total=500.0 * 10 * 0.2)           # 10 voxels of 0.2 mL, mean 500 counts
    df = measure.roi_table([spect], {"sensitivity": 10.0, "time_s": 100.0, "decay": 1.0})
    assert df["total_counts"][0] == pytest.approx(5000.0)
    assert df["activity_MBq"][0] == pytest.approx(5.0)                             # 5000 / (10 cps/MBq x 100 s)
    assert df["concentration_MBq_per_ml"][0] == pytest.approx(2.5) and df["mean_intensity"][0] == 500.0
    assert "activity_MBq" not in measure.roi_table([spect], {})                    # no calibration: counts only
    pet = dict(base, unit="SUV", mean=4.0, total=8.0, dose_MBq=200.0, weight_kg=80.0)
    d2 = measure.roi_table([pet])
    assert d2["concentration_kBq_per_ml"][0] == pytest.approx(10.0)               # 4 x 200/80
    assert d2["activity_MBq"][0] == pytest.approx(0.02)
    bq = measure.roi_table([dict(base, unit="BQML", mean=5000.0, total=0.0)])
    assert bq["concentration_kBq_per_ml"][0] == pytest.approx(5.0)
    assert measure.decay_factor(159.5, 159.5) == pytest.approx(2.0)


def test_mask_stats_and_measured_image_follows_the_display():
    from rsw import io
    from rsw.session import Session
    img, ball = sphere_image()
    geo = io.Geometry((40, 40, 40), (2.0, 2.0, 2.0), (0.0, 0.0, 0.0), (1, 0, 0, 0, 1, 0, 0, 0, 1))
    pet = io.Volume("PET", img, geo, "PET", "SUV")
    spect = io.Volume("SPECT", img * 100, geo, "SPECT", "CNTS")
    s = Session(None, None, [pet, spect])
    assert s.measure_name == "PET"
    s.over_pick = "SPECT"                                   # looking at the SPECT: measurements are on the SPECT
    assert s.measure_name == "SPECT"
    s.add_empty_layer("les")
    can_ball = s.can["PET"] == 10.0
    s.layers["les"].apply(can_ball, tuple(slice(0, n) for n in s.shape), 1)
    st = s.mask_stats()
    assert st["mean"] == pytest.approx(1000.0) and st["voxels"] == int(ball.sum())


def _two_image_session():
    from rsw import io
    from rsw.session import Session
    img, ball = sphere_image()
    geo = io.Geometry((40, 40, 40), (2.0, 2.0, 2.0), (0.0, 0.0, 0.0), (1, 0, 0, 0, 1, 0, 0, 0, 1))
    ct = io.Volume("CT", img * 10 - 50, geo, "CT", "HU")
    pet = io.Volume("PET", img, geo, "PET", "SUV", info={"injected_MBq": 250.0, "dose_at_scan_MBq": 200.0, "weight_kg": 80.0})
    return Session(None, None, [pet, ct])


def test_roi_plane_details_all_images_save_and_load(tmp_path):
    from rsw import roi as R
    s = _two_image_session()
    k = int(np.argwhere(s.can["PET"] == 10.0).mean(axis=0).round()[0])            # axial slice through the ball centre
    reg2d = R.ellipse_region(R.plane_shape(s.shape, "axial"), 16, 16, 24, 24)       # inside the ball
    name = s.add_plane_roi("axial", k, reg2d, "ellipse")
    assert name == "ROI 1" and s.roi_selected == name
    info, rows = s.roi_details(name)
    assert info["plane"] == "axial" and info["slices_z"] == 1 and info["area_mm2"] == pytest.approx(reg2d.sum() * 4.0)
    by = {r["image"]: r for r in rows}
    assert set(by) == {"PET", "CT"} and by["PET"]["mean"] == pytest.approx(10.0) and by["CT"]["mean"] == pytest.approx(50.0)
    assert by["CT"]["peak"] is None and by["PET"]["unit"] == "SUV"
    df = measure.roi_table(rows)
    assert df.loc[df.image == "PET", "concentration_kBq_per_ml"].iloc[0] == pytest.approx(10.0 * 200 / 80)
    centre = tuple(int(v) for v in np.argwhere(s.rois[name].array).mean(axis=0).round())
    assert s.roi_hit(centre) == name and s.roi_hit((0, 0, 0)) is None
    out = s.save_roi(name, tmp_path / "liver.nii.gz")
    assert out.exists() and (tmp_path / "liver.roi.json").exists() and s.rois[name].saved
    s.remove_roi(name)
    assert not s.rois
    back = s.load_roi(tmp_path / "liver.nii.gz")
    assert back == "ROI 1" and s.rois[back].shape == "ellipse" and int(s.rois[back].array.sum()) == int(reg2d.sum())


def test_roi_from_mask_rename_and_reference_change():
    s = _two_image_session()
    s.add_empty_layer("m")
    s.layers["m"].apply(s.can["PET"] == 10.0, tuple(slice(0, n) for n in s.shape), 1)
    n = s.roi_from_layer("m")
    assert n == "m (mask)" and s.roi_details(n)[0]["voxels"] == int((s.can["PET"] == 10.0).sum())
    s.rename_roi(n, "lesion A")
    assert "lesion A" in s.rois and s.roi_selected == "lesion A"
    with pytest.raises(ValueError):
        s.add_roi(np.zeros(s.shape, bool), "freehand")
    v0 = int(s.rois["lesion A"].array.sum())
    s.set_reference("CT")                                         # same grid here: the ROI must survive unchanged
    assert int(s.rois["lesion A"].array.sum()) == v0


def test_pet_info_has_dose_decayed_to_scan(tmp_path):
    from rsw import io
    make_series(tmp_path / "pet")
    v = io.load_volume(tmp_path / "pet")
    i = v.info
    assert i["uptake_min"] > 0 and i["dose_at_scan_MBq"] < i["injected_MBq"]
    assert i["dose_at_scan_MBq"] == pytest.approx(i["injected_MBq"] * 2 ** (-i["uptake_min"] * 60 / i["half_life_s"]))
