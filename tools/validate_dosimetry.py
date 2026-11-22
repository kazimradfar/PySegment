"""Validation of the PySegment dosimetry module (rsw/dosimetry.py) against analytic results and published values.

A. Nuclear data: half-lives against ICRP 107 (through the `radioactivedecay` package, which ships ICRP 107);
   mean electron energy per decay of Lu-177 against the EANM recommendation.
B. Local energy deposition against published self-dose S-values: IDAC-Dose 2.1 adult male S-values for Lu-177
   (kidneys, liver, spleen, red marrow) as tabulated in Table 2 of Sjögreen Gleisner K et al., EANM dosimetry
   committee recommendations for dosimetry of 177Lu-labelled somatostatin-receptor- and PSMA-targeting ligands,
   Eur J Nucl Med Mol Imaging 2022;49:1778-1809.
C. Time-integrated activity: kinetics with an exact integral (mono-exponential, bi-exponential, uptake and
   washout) sampled at clinical time points; every integration method of the program; noise-free and with
   multiplicative noise (Monte Carlo).
D. Formulas and units: absorbed dose by local deposition computed independently, S-value units, MIRD organ
   doses with a small source/target matrix computed by hand, remainder of body, ICRP 103 / 60 weights.

A2. Energies per decay (alpha, electron, photon, and the short-lived chain daughters) against ICRP 107 (data
   files of the R package RadData, https://github.com/markhogue/RadData, read with pyreadr).

Reproduce (a few seconds):  python tools/validate_dosimetry.py --icrp107 <RadData>/data
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsw import dosimetry as ds  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MEV_J = 1.602176634e-13
DECAYS_PER_MBQ_H = 1e6 * 3600.0

# EANM 2022, Table 2 (IDAC-Dose 2.1, adult male): organ -> (mass g, S self mGy/(MBq h)); local-deposition value
# Delta(177Lu, electrons) = 0.0853 mJ/(MBq h). Transcribed values: check against the PDF before quoting.
EANM_TABLE2 = {"kidneys": (422.0, 0.204), "liver": (2360.0, 0.0376), "spleen": (228.4, 0.377), "red marrow": (1394.0, 0.0349)}
EANM_DELTA_E_MJ = 0.0853

SCHEDULES = {"5 points (4, 24, 48, 96, 168 h)": [4, 24, 48, 96, 168], "3 points (24, 96, 168 h)": [24, 96, 168]}
NOISE = [0.0, 0.05, 0.10]
N_MC = 2000


def lam(t_half_h: float) -> float:
    return np.log(2.0) / t_half_h


def kinetics():
    """name -> (A(t) in MBq, exact integral from 0 to infinity in MBq h). Illustrative curves spanning typical
    Lu-177 behaviour (fast and slow clearance, two phases, uptake then washout); not patient data."""
    k = {}
    for th in (40.0, 100.0):
        k[f"mono-exponential, T_eff {th:.0f} h"] = (lambda t, th=th: 100 * np.exp(-lam(th) * t), 100 / lam(th))
    k["bi-exponential, 30 % T 3 h + 70 % T 45 h"] = (
        lambda t: 30 * np.exp(-lam(3) * t) + 70 * np.exp(-lam(45) * t), 30 / lam(3) + 70 / lam(45))
    k["uptake T 3 h, washout T 60 h"] = (
        lambda t: 100 * (np.exp(-lam(60) * t) - np.exp(-lam(3) * t)), 100 * (1 / lam(60) - 1 / lam(3)))
    return k


def part_a() -> dict:
    rows = []
    try:
        import radioactivedecay as rd
        for name, iso in ds.ISOTOPES.items():
            ref = rd.Nuclide(name).half_life("h")
            rows.append({"nuclide": name, "pysegment_h": iso["half_life_h"], "icrp107_h": round(ref, 4),
                         "difference_pct": round(100 * (iso["half_life_h"] - ref) / ref, 3)})
    except ImportError:
        rows.append({"note": "pip install radioactivedecay to compare half-lives with ICRP 107"})
    e_lu = ds.ISOTOPES["Lu-177"]["electron"]
    delta = e_lu * MEV_J * DECAYS_PER_MBQ_H * 1e3                        # mJ per MBq h
    return {"half_lives": rows, "lu177_delta_e_mJ_per_MBq_h": round(delta, 5),
            "eanm_delta_e_mJ_per_MBq_h": EANM_DELTA_E_MJ, "lu177_delta_difference_pct": round(100 * (delta / EANM_DELTA_E_MJ - 1), 2)}


def part_a2(radd_data: str | None) -> list[dict]:
    """PySegment energies per decay against ICRP 107. Daughters count when they are shorter-lived than the parent
    (equilibrium), weighted by their branching ratios."""
    if not radd_data:
        return []
    import pyreadr
    ndx = list(pyreadr.read_r(str(Path(radd_data) / "ICRP_07.NDX.rda")).values())[0].set_index("RN")
    unit = {"us": 1e-6 / 3600, "ms": 1e-3 / 3600, "s": 1 / 3600, "m": 1 / 60, "h": 1, "d": 24, "y": 24 * 365.25}

    def t_half(n):
        return float(ndx.loc[n, "half_life"]) * unit[str(ndx.loc[n, "units"]).strip()]

    def energies(n):
        r = ndx.loc[n]
        return np.array([r.E_alpha, r.E_electron, r.E_photon], float)

    def chain(n, t_max, w=1.0, acc=None):
        acc = np.zeros(3) if acc is None else acc
        for i in range(1, 5):
            p, b = ndx.loc[n, f"progeny_{i}"], ndx.loc[n, f"branch_{i}"]
            if isinstance(p, str) and p in ndx.index and b > 0 and t_half(p) < t_max:
                acc += w * b * energies(p)
                chain(p, t_max, w * b, acc)
        return acc

    rows = []
    for name, iso in ds.ISOTOPES.items():
        ref = np.r_[energies(name), chain(name, t_half(name))]
        for k, key in enumerate(("alpha", "electron", "photon", "alpha_d", "electron_d", "photon_d")):
            if iso[key] == 0 and ref[k] < 5e-5:
                continue
            rows.append({"nuclide": name, "component": key, "pysegment_MeV": iso[key], "icrp107_MeV": round(float(ref[k]), 5),
                         "difference_pct": round(100 * (iso[key] / ref[k] - 1), 2) if ref[k] else None})
    return rows


def part_b() -> pd.DataFrame:
    e = ds.energy_per_decay(ds.ISOTOPES["Lu-177"])[0]
    rows = []
    for organ, (mass, s_ref) in EANM_TABLE2.items():
        s_led = ds.dose_local_deposition_gy(1.0, mass, e) * 1e3          # mGy per MBq h
        rows.append({"organ": organ, "mass_g": mass, "S_IDAC_mGy_per_MBq_h": s_ref, "S_PySegment_LED_mGy_per_MBq_h": round(s_led, 5),
                     "difference_pct": round(100 * (s_led / s_ref - 1), 1)})
    return pd.DataFrame(rows)


def part_c(seed: int = 20261121) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for kname, (f, exact) in kinetics().items():
        for sname, times in SCHEDULES.items():
            t = np.asarray(times, float)
            for method in ds.METHODS:
                if method == "biexp" and len(t) < 4:
                    continue
                for sd in NOISE:
                    n = 1 if sd == 0 else N_MC
                    vals, fails = [], 0
                    for _ in range(n):
                        a = f(t) * (1 + rng.normal(0, sd, len(t))) if sd else f(t)
                        try:
                            vals.append(ds.time_integrated_activity(t, np.clip(a, 1e-9, None), method, 159.53)["tia_MBq_h"])
                        except ValueError:
                            fails += 1
                    err = 100 * (np.asarray(vals) / exact - 1) if vals else np.array([np.nan])
                    rows.append({"kinetics": kinetics_label(kname), "schedule": sname, "method": method, "noise_sd_pct": int(100 * sd),
                                 "exact_MBq_h": round(exact, 2), "bias_pct": round(float(np.mean(err)), 2),
                                 "sd_pct": round(float(np.std(err)), 2) if sd else 0.0,
                                 "rmse_pct": round(float(np.sqrt(np.mean(err ** 2))), 2), "failed_fits": fails, "runs": n})
    return pd.DataFrame(rows)


def kinetics_label(k: str) -> str:
    return k


def part_d() -> dict:
    checks = {}
    # local deposition: 10 MBq h of Lu-177 in 20 g, computed by hand
    hand = 10 * DECAYS_PER_MBQ_H * 0.148 * MEV_J / 0.020
    checks["local_deposition_Gy"] = {"pysegment": ds.dose_local_deposition_gy(10, 20, 0.148), "hand": hand}
    # S-value units: 1 mGy/(MBq s) = 3.6 Gy/(MBq h); 1 Gy/(Bq s) = 3.6e9 Gy/(MBq h)
    checks["unit_factors"] = {"mGy/(MBq*s)": ds.S_UNITS["mGy/(MBq*s)"], "expected_1": 3.6,
                              "Gy/(Bq*s)": ds.S_UNITS["Gy/(Bq*s)"], "expected_2": 3.6e9}
    # MIRD organ doses: 2 sources x 3 targets, by hand
    s = pd.DataFrame({"kidneys": [0.2, 0.01, 0.001], "liver": [0.01, 0.04, 0.002]}, index=["kidneys", "liver", "spleen"])
    tia = {"kidneys": 500.0, "liver": 1000.0, "tumour": 50.0}
    df = ds.mird_organ_doses(tia, s, unit="mGy/(MBq*h)")
    hand_d = {"kidneys": (0.2 * 500 + 0.01 * 1000) * 1e-3, "liver": (0.01 * 500 + 0.04 * 1000) * 1e-3,
              "spleen": (0.001 * 500 + 0.002 * 1000) * 1e-3}
    checks["mird_organ_doses_Gy"] = {"pysegment": dict(zip(df.target, df.dose_Gy.round(6))), "hand": hand_d,
                                     "missing_source_reported": df.attrs["sources_missing_in_matrix"]}
    checks["remainder_MBq_h"] = {"pysegment": ds.remainder_tia(7400, 30.0, {"kidneys": 9000.0, "liver": 20000.0}),
                                 "hand": 7400 * 30.0 - 29000.0}
    w103 = sum(ds.ICRP103_W.values()) + ds.REMAINDER_W["ICRP103"]
    w60 = sum(ds.ICRP60_W.values()) + ds.REMAINDER_W["ICRP60"]
    all_one = {k: 1.0 for k in list(ds.ICRP103_W) + ["kidneys"]}
    checks["tissue_weights"] = {"sum_ICRP103": round(w103, 6), "sum_ICRP60": round(w60, 6),
                                "E_all_tissues_1Gy_ICRP103_Sv": round(ds.effective_dose_sv(all_one, "ICRP103")[0], 6)}
    ok = (abs(checks["local_deposition_Gy"]["pysegment"] / hand - 1) < 1e-12
          and abs(ds.S_UNITS["mGy/(MBq*s)"] - 3.6) < 1e-12 and abs(ds.S_UNITS["Gy/(Bq*s)"] - 3.6e9) < 1e-3
          and all(abs(checks["mird_organ_doses_Gy"]["pysegment"][k] - v) < 1e-9 for k, v in hand_d.items())
          and checks["mird_organ_doses_Gy"]["missing_source_reported"] == ["tumour"]
          and checks["remainder_MBq_h"]["pysegment"] == checks["remainder_MBq_h"]["hand"]
          and abs(w103 - 1) < 1e-9 and abs(w60 - 1) < 1e-9 and abs(checks["tissue_weights"]["E_all_tissues_1Gy_ICRP103_Sv"] - 1) < 1e-9)
    checks["all_pass"] = bool(ok)
    return checks


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--icrp107", default="", help="data folder of the RadData package (ICRP_07.NDX.rda)")
    args = ap.parse_args(argv)
    out = ROOT / "validation" / "dosimetry"
    out.mkdir(parents=True, exist_ok=True)
    a, b, c, d = part_a(), part_b(), part_c(), part_d()
    e = part_a2(args.icrp107)
    if e:
        pd.DataFrame(e).to_csv(out / "energies_icrp107.csv", index=False)
    b.to_csv(out / "organ_svalues_lu177.csv", index=False)
    c.to_csv(out / "tia_accuracy.csv", index=False)
    pd.DataFrame(a["half_lives"]).to_csv(out / "half_lives.csv", index=False)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = ""
    noise0 = c[c.noise_sd_pct == 0]
    summary = {"pysegment_commit": commit, "python": platform.python_version(), "numpy": np.__version__,
               "nuclear_data": {k: v for k, v in a.items() if k != "half_lives"},
               "half_life_max_abs_difference_pct": float(pd.DataFrame(a["half_lives"]).difference_pct.abs().max())
               if "difference_pct" in pd.DataFrame(a["half_lives"]) else None,
               "energies_vs_icrp107_max_abs_difference_pct": max((abs(r["difference_pct"]) for r in e
                                                                    if r["difference_pct"] is not None), default=None),
               "organ_svalues": b.to_dict(orient="records"),
               "tia_noise_free_abs_bias_pct_by_method": noise0.groupby("method").bias_pct.apply(lambda x: f"{x.abs().min():.1f}-{x.abs().max():.1f}").to_dict(),
               "formulas_and_units": d}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))
    return summary


if __name__ == "__main__":
    main()
