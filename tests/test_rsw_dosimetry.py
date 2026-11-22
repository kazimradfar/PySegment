import numpy as np
import pytest

from rsw import dosimetry as ds


def test_monoexp_fit_is_exact_for_an_exponential():
    t = np.array([24.0, 48.0, 96.0, 168.0])
    a = 100.0 * np.exp(-0.01 * t)
    r = ds.time_integrated_activity(t, a, "monoexp")
    assert r["tia_MBq_h"] == pytest.approx(100.0 / 0.01, rel=1e-6) and r["t_eff_h"] == pytest.approx(np.log(2) / 0.01)
    assert r["r2"] == pytest.approx(1.0)


def test_trapezoid_with_effective_tail_is_close():
    t = np.array([4.0, 24.0, 48.0, 96.0, 168.0])
    a = 100.0 * np.exp(-0.02 * t)
    r = ds.time_integrated_activity(t, a, "trapezoid_effective", from_zero=False)
    exact = 100.0 / 0.02 * (np.exp(-0.02 * 4.0))                           # integral from 4 h to infinity
    assert r["tia_MBq_h"] == pytest.approx(exact, rel=0.08)                # trapezoids overestimate a convex curve slightly
    phys = ds.time_integrated_activity(t, a, "trapezoid_physical", from_zero=False)
    assert phys["tia_MBq_h"] != r["tia_MBq_h"]


def test_notes_and_errors():
    r = ds.time_integrated_activity([24, 48], [10, 9], "trapezoid_effective")
    assert any("extrapolated" in n for n in r["notes"]) and any("zero at time 0" in n for n in r["notes"])
    with pytest.raises(ValueError):
        ds.time_integrated_activity([24], [10])
    with pytest.raises(ValueError):
        ds.time_integrated_activity([24, 24], [10, 5])
    with pytest.raises(ValueError):
        ds.time_integrated_activity([24, 48], [5, 10], "monoexp")           # rising: no fit


def test_local_deposition_dose_reference_value():
    # 1 MBq*h = 3.6e9 decays; 0.148 MeV each = 8.54e-5 J; in 1 g -> 0.0854 Gy
    assert ds.dose_local_deposition_gy(1.0, 1.0, 0.148) == pytest.approx(0.0854, rel=1e-3)
    assert ds.dose_svalue_gy(100.0, 0.5) == 50.0


def test_dosimetry_table_two_targets():
    pts = [{"target": "kidney", "time_h": t, "activity_MBq": 100 * np.exp(-0.01 * t), "volume_ml": 300.0} for t in (24, 48, 96)]
    pts += [{"target": "lesion", "time_h": t, "activity_MBq": 20 * np.exp(-0.005 * t), "volume_ml": 10.0} for t in (24, 48, 96)]
    pts += [{"target": "bad", "time_h": 24, "activity_MBq": 1.0, "volume_ml": 1.0}]
    df = ds.dosimetry_table(pts, "monoexp", 159.53, 0.148, admin_mbq=7400.0)
    k = df[df.target == "kidney"].iloc[0]
    assert k["tia_MBq_h"] == pytest.approx(10000.0, rel=1e-6)
    assert k["dose_Gy"] == pytest.approx(ds.dose_local_deposition_gy(10000.0, 300.0, 0.148))
    assert k["residence_time_h"] == pytest.approx(10000.0 / 7400.0)
    assert k["dose_Gy_per_GBq"] == pytest.approx(k["dose_Gy"] / 7.4)
    assert "at least two" in df[df.target == "bad"].iloc[0]["note"]
    s = ds.dosimetry_table(pts[:3], "monoexp", 159.53, 0.148, s_value=0.01)
    assert s["dose_Gy"][0] == pytest.approx(100.0)


def test_dosimetry_dialog(monkeypatch, tmp_path):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("pyqtgraph")
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from rsw.gui.main_window import MainWindow
    w = MainWindow()
    d = w.open_dosimetry()
    for t in (24, 48, 96):
        d.add_row("kidney", t, 100 * np.exp(-0.01 * t), 300)
    d.method.setCurrentIndex(2)                                # one exponential
    d.admin.setValue(7400.0)
    d.iso.setCurrentText("Lu-177")
    d.calculate()
    r = d.df.iloc[0]
    assert r["tia_MBq_h"] == pytest.approx(10000.0, rel=1e-6) and r["dose_Gy"] > 0 and "Calculate" not in d.df.columns
    assert "HOW IT IS CALCULATED" in d.findChild(QtWidgets.QPlainTextEdit).toPlainText()
    w.close()


def test_energy_per_decay_alpha_chain_and_photons():
    """ICRP 107 energies per decay (validation/dosimetry/): Ac-225 with its short-lived daughters."""
    ac = ds.ISOTOPES["Ac-225"]
    full, weighted = ds.energy_per_decay(ac, 1.0, 0.0, 5.0)
    assert full == pytest.approx(5.8919 + 22.1026 + 0.0248 + 0.6648)
    assert weighted == pytest.approx(5.0 * (5.8919 + 22.1026) + 0.0248 + 0.6648)
    none = ds.energy_per_decay(ac, 0.0, 0.0, 1.0)[0]
    assert none == pytest.approx(5.8919 + 0.0248) and none < full                # daughters leave: far less energy
    assert ds.energy_per_decay(ds.ISOTOPES["Tc-99m"], 1.0, 0.0)[0] == pytest.approx(0.0162)
    assert ds.energy_per_decay(ds.ISOTOPES["Tc-99m"], 1.0, 0.1)[0] == pytest.approx(0.0162 + 0.01266)


def test_lu177_photon_fraction_counts():
    """Regression: the Lu-177 photon energy was 0, so the photon absorbed fraction had no effect at all."""
    lu = ds.ISOTOPES["Lu-177"]
    assert ds.energy_per_decay(lu, 1.0, 0.0)[0] == pytest.approx(0.1479)
    assert ds.energy_per_decay(lu, 1.0, 0.5)[0] == pytest.approx(0.1479 + 0.5 * 0.0351)


def test_dose_table_with_rbe_column():
    pts = [{"target": "t", "time_h": t, "activity_MBq": 1.0 * np.exp(-0.003 * t), "volume_ml": 5.0} for t in (24, 72, 168)]
    e, w = ds.energy_per_decay(ds.ISOTOPES["Ac-225"], 1.0, 0.0, 5.0)
    df = ds.dosimetry_table(pts, "monoexp", 238.1, e, 1.0, 8.0, None, True, weighted_energy_mev=w)
    assert df["dose_Gy_RBE"][0] == pytest.approx(df["dose_Gy"][0] * w / e) and "dose_Gy_RBE_per_GBq" in df


def test_calibration_profiles_roundtrip(tmp_path):
    from rsw import calibration
    p = tmp_path / "c.json"
    calibration.save_profile("Lu-PSMA SPECT", 11.5, "Lu-177", 159.53, p)
    assert calibration.load(p)["Lu-PSMA SPECT"]["sensitivity_cps_per_MBq"] == 11.5
    with pytest.raises(ValueError):
        calibration.save_profile("", 1.0, "Lu-177", 1.0, p)
    calibration.delete_profile("Lu-PSMA SPECT", p)
    assert calibration.load(p) == {} and calibration.load(tmp_path / "missing.json") == {}


def test_mird_organ_doses_matrix():
    import pandas as pd
    s = pd.DataFrame({"kidney": [2.0, 0.1], "liver": [0.05, 1.0]}, index=["kidney", "liver"])       # mGy/(MBq*h)
    out = ds.mird_organ_doses({"kidney": 100.0, "liver": 50.0, "spleen": 5.0}, s, "mGy/(MBq*h)", {"kidney": 0.12, "liver": 0.04})
    d = dict(zip(out["target"], out["dose_Gy"]))
    assert d["kidney"] == pytest.approx((2.0 * 100 + 0.05 * 50) * 1e-3)          # 0.2025 Gy
    assert d["liver"] == pytest.approx((0.1 * 100 + 1.0 * 50) * 1e-3)
    assert out.attrs["sources_missing_in_matrix"] == ["spleen"]
    assert out.attrs["effective_dose_Sv"] == pytest.approx(0.12 * d["kidney"] + 0.04 * d["liver"])
    assert ds.S_UNITS["mGy/(MBq*s)"] == pytest.approx(3.6) and ds.S_UNITS["Gy/(Bq*s)"] == pytest.approx(3.6e9)
    with pytest.raises(ValueError):
        ds.mird_organ_doses({"spleen": 1.0}, s)


def test_biexp_exact_and_errors():
    t = np.array([2.0, 12.0, 24.0, 48.0, 96.0, 168.0])
    a = 50 * np.exp(-0.2 * t) + 30 * np.exp(-0.01 * t)
    r = ds.time_integrated_activity(t, a, "biexp")
    assert r["tia_MBq_h"] == pytest.approx(50 / 0.2 + 30 / 0.01, rel=0.02) and r["r2"] > 0.999
    with pytest.raises(ValueError):
        ds.time_integrated_activity(t[:3], a[:3], "biexp")


def test_effective_dose_remainder_and_mass_scaling():
    import pandas as pd
    d = {"Red Marrow": 0.01, "Liver": 0.02, "Lungs": 0.01, "Testes": 0.001, "Spleen": 0.03, "Pancreas": 0.01}
    e, notes = ds.effective_dose_sv(d, "ICRP103")
    assert e == pytest.approx(0.12 * 0.01 + 0.04 * 0.02 + 0.12 * 0.01 + 0.08 * 0.001 + 0.12 * 0.02)   # remainder = mean(spleen, pancreas)
    assert any("colon" in n for n in notes)
    assert ds.remainder_tia(1000.0, 10.0, {"a": 4000.0, "b": 7000.0}) == 0.0
    assert ds.remainder_tia(1000.0, 10.0, {"a": 4000.0}) == 6000.0
    s = pd.DataFrame({"kidney": [2.0, 0.1], "liver": [0.05, 1.0]}, index=["kidney", "liver"])
    out = ds.mird_organ_doses({"kidney": 100.0}, s, mass_scale={"kidney": 0.5})           # patient kidney twice the model mass
    assert dict(zip(out["target"], out["dose_Gy"]))["kidney"] == pytest.approx(0.2 * 0.5 * 1.0)
    assert dict(zip(out["target"], out["dose_Gy"]))["liver"] == pytest.approx(0.1 * 100 * 1e-3)       # cross-dose not scaled


def test_dialog_mird_with_remainder_and_effective_dose(monkeypatch, tmp_path):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("pyqtgraph")
    import pandas as pd
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from rsw.gui import dosimetry_dialog as dd
    from rsw.gui.main_window import MainWindow
    m = pd.DataFrame({"Kidney": [1.0, 0.1, 0.0], "Liver": [0.1, 1.0, 0.0], "Remainder of Body": [0.01, 0.01, 0.05]},
                     index=["Kidney", "Liver", "Red Marrow"])
    csv = tmp_path / "s.csv"
    m.to_csv(csv)
    w = MainWindow()
    d = w.open_dosimetry()
    for t in (24, 48, 96):
        d.add_row("Kidney", t, 20 * np.exp(-0.01 * t), 300)
    d.method.setCurrentIndex(2)
    d.admin.setValue(1000.0)
    d.calculate()
    monkeypatch.setattr(dd.QtWidgets.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(csv), "")))
    monkeypatch.setattr(dd.QtWidgets.QInputDialog, "getItem", staticmethod(lambda *a, **k: ("mGy/(MBq*h)", True)))
    monkeypatch.setattr(dd.QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: dd.QtWidgets.QMessageBox.No))
    d.mird()
    out = dict(zip(d.df["target"], d.df["dose_Gy"]))
    tia_k = 20 / 0.01
    rem = max(1000.0 * (159.53 / 0.693147) - tia_k, 0.0)
    assert out["Red Marrow"] == pytest.approx(0.05 * rem * 1e-3, rel=1e-3)
    assert any(k.startswith("EFFECTIVE DOSE") for k in out)
    w.close()


def test_counts_jacobian_suv_decay_and_ct_density():
    import SimpleITK as sitk
    from rsw import io, measure
    from rsw.suv import suv_bw_factor  # noqa: F401  (the info keys are checked in the SUV tests)
    # counts: an affine scaling of 1.2 per axis must keep the total counts (Jacobian 1.728)
    z, y, x = np.mgrid[:40, :40, :40]
    arr = (100.0 * np.exp(-((z - 19.5) ** 2 + (y - 19.5) ** 2 + (x - 19.5) ** 2) / (2 * 4.0 ** 2))).astype(np.float32)
    geo = io.Geometry((40, 40, 40), (2.0, 2.0, 2.0), (-40.0, -40.0, -40.0), (1, 0, 0, 0, 1, 0, 0, 0, 1))
    vol = io.Volume("S", arr, geo, "SPECT", "CNTS")
    a = sitk.AffineTransform(3)
    a.SetMatrix([1.2, 0, 0, 0, 1.2, 0, 0, 0, 1.2])
    out = io.resample_to(vol, geo, transform=a)
    assert out.array.sum() == pytest.approx(arr.sum(), rel=0.03)
    conc = io.resample_to(io.Volume("P", arr, geo, "PET", "SUV"), geo, transform=a)
    assert conc.array.max() == pytest.approx(arr.max(), rel=0.02)                     # concentrations are not scaled
    # SUV -> kBq/mL uses the dose decayed to the scan start
    rows = [{"kind": "roi", "image": "PET", "unit": "SUV", "mean": 4.0, "volume_ml": 10.0, "voxels": 10, "total": 40.0,
             "dose_MBq": 100.0, "weight_kg": 80.0}]
    assert measure.roi_table(rows)["concentration_kBq_per_ml"][0] == pytest.approx(5.0)
    assert measure.ct_density([-1000, 0, 1000]).tolist() == pytest.approx([0.0, 1.0, 1.6])
    pts = [{"target": "k", "time_h": t, "activity_MBq": 10 * np.exp(-0.01 * t), "volume_ml": 100.0, "density_g_ml": 1.05}
           for t in (24, 48, 96)]
    d = ds.dosimetry_table(pts, "monoexp", 159.53, 0.148)
    assert d["mass_g"][0] == pytest.approx(105.0)


def test_biexponential_fit_stays_physical_with_noisy_data():
    """Regression: without bounds, noisy points gave a component that hardly decays and an integral up to 1e6 times
    too large. Every component must now clear at least as fast as physical decay."""
    import numpy as np
    from rsw import dosimetry as ds
    rng = np.random.default_rng(7)
    t = np.array([4.0, 24, 48, 96, 168])
    lam = lambda th: np.log(2) / th
    exact = 100 / lam(40)
    vals = [ds.time_integrated_activity(t, 100 * np.exp(-lam(40) * t) * (1 + rng.normal(0, 0.10, 5)), "biexp", 159.53)["tia_MBq_h"]
            for _ in range(300)]
    assert max(vals) < 2 * exact and min(vals) > 0.5 * exact


def test_local_deposition_warns_for_red_marrow():
    from rsw import dosimetry as ds
    pts = [{"target": "Red marrow", "time_h": t, "activity_MBq": a, "volume_ml": 1000} for t, a in [(24, 50), (96, 20)]]
    df = ds.dosimetry_table(pts, "monoexp", 159.53, 0.148)
    assert "not valid for red marrow" in df.loc[0, "note"]


def test_tail_never_decays_slower_than_physical_decay():
    from rsw import dosimetry as ds
    r = ds.time_integrated_activity([24, 96, 168], [50.0, 30.0, 29.9], "trapezoid_effective", 159.53)
    lam = 0.6931471805599453 / 159.53
    area = 24 * 50 / 2 + 72 * (50 + 30) / 2 + 72 * (30 + 29.9) / 2
    assert abs(r["tia_MBq_h"] - (area + 29.9 / lam)) < 1e-6
    assert any("slower than physical decay" in n for n in r["notes"])
