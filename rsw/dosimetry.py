"""Simple dosimetry from activities measured at several times (no Qt).

Workflow in words (the same text is shown in the program):
  1. Measure the activity (MBq) of a lesion or organ on each scan (ROI window) and note the time of each scan.
     The activity must be the activity PRESENT at the scan (not decay-corrected back to the injection): the physical
     decay is part of the time-activity curve. PET in Bq/mL or SUV is decay-corrected to the scan start, which is
     what is needed; SPECT counts are converted with the calibration and must NOT be decay-corrected for dosimetry.
     Times are hours after the reference moment (for example the end of the infusion).
  2. Time-integrated activity  A~ = integral of A(t) dt  (MBq*h) = total number of decays / 3.6e9. It is found by
     trapezoids between the measurements plus an extrapolation after the last one, or by fitting one exponential.
  3. Absorbed dose  D = A~ x S.  Local deposition (beta only): all electron energy of a decay is absorbed in the
     target itself, so S = E / m with E the mean electron energy per decay and m the mass of the target.
     Photons and neighbouring organs (cross-fire) are NOT included: for Lu-177 lesions and kidneys this is usually a
     small error, for large organs or other isotopes it is not. Alternatively use an S-value you trust
     (OLINDA/MIRD, sphere model) in Gy per MBq*h.
"""
from __future__ import annotations

import numpy as np

MEV_J = 1.602176634e-13
# Mean energy emitted per decay of the nuclide, in MeV, split by radiation type, from ICRP Publication 107
# (Eckerman & Endo 2008; values checked in validation/dosimetry/). alpha_d / electron_d / photon_d: energy of the
# daughters of the chain that are shorter-lived than the parent, weighted by their branching (they decay where they
# are; see "daughters retained"). Half-lives: NuDat (ICRP 107 differs by < 0.2 %, Ac-225 by 0.8 %: 10.0 d).
ISOTOPES = {
    "Lu-177":  {"half_life_h": 159.53, "alpha": 0.0, "electron": 0.1479, "photon": 0.0351, "alpha_d": 0.0, "electron_d": 0.0, "photon_d": 0.0,
                "note": "beta + conversion/Auger electrons; 113/208 keV photons count only with a photon absorbed fraction"},
    "Y-90":    {"half_life_h": 64.05, "alpha": 0.0, "electron": 0.9331, "photon": 0.0, "alpha_d": 0.0, "electron_d": 0.0, "photon_d": 0.0,
                "note": "pure beta"},
    "I-131":   {"half_life_h": 192.5, "alpha": 0.0, "electron": 0.1918, "photon": 0.3827, "alpha_d": 0.0, "electron_d": 0.0, "photon_d": 0.0,
                "note": "beta + electrons; 364 keV photons: absorbed fraction depends on target size"},
    "Ac-225":  {"half_life_h": 238.1, "alpha": 5.8919, "electron": 0.0248, "photon": 0.0171, "alpha_d": 22.1026, "electron_d": 0.6648, "photon_d": 0.2022,
                "note": "alpha chain Fr-221, At-217, Bi-213, Po-213 (and betas); daughters may leave the target after the recoil"},
    "Ra-223":  {"half_life_h": 274.3, "alpha": 5.7701, "electron": 0.0781, "photon": 0.1413, "alpha_d": 21.1027, "electron_d": 0.9649, "photon_d": 0.1728,
                "note": "alpha chain Rn-219, Po-215, Pb-211, Bi-211 (and betas)"},
    "Tc-99m":  {"half_life_h": 6.0067, "alpha": 0.0, "electron": 0.0162, "photon": 0.1266, "alpha_d": 0.0, "electron_d": 0.0, "photon_d": 0.0,
                "note": "140 keV photons: absorbed fraction depends on target size (usually small)"},
    "F-18":    {"half_life_h": 1.8295, "alpha": 0.0, "electron": 0.2416, "photon": 0.9886, "alpha_d": 0.0, "electron_d": 0.0, "photon_d": 0.0,
                "note": "positron (96.7 % of decays) + 2 x 511 keV annihilation photons"},
    "Ga-68":   {"half_life_h": 1.1285, "alpha": 0.0, "electron": 0.7379, "photon": 0.9487, "alpha_d": 0.0, "electron_d": 0.0, "photon_d": 0.0,
                "note": "positron (88.9 % of decays) + 511 keV annihilation photons and 1077 keV gamma"},
}


def energy_per_decay(iso: dict, daughters_retained: float = 1.0, photon_fraction: float = 0.0, rbe_alpha: float = 1.0):
    """(absorbed energy, RBE-weighted energy) per decay in MeV for the target. Chain daughters count with the fraction
    that stays in the target; photons with the absorbed fraction; alphas are multiplied by the RBE in the second value."""
    f = float(daughters_retained)
    alpha = iso["alpha"] + f * iso.get("alpha_d", 0.0)
    electron = iso["electron"] + f * iso.get("electron_d", 0.0)
    photon = (iso["photon"] + f * iso.get("photon_d", 0.0)) * float(photon_fraction)
    return alpha + electron + photon, rbe_alpha * alpha + electron + photon


METHODS = {
    "trapezoid_effective": "Trapezoids, then decay after the last point with the effective rate from the last two points",
    "trapezoid_physical": "Trapezoids, then physical decay only after the last point (no biological clearance)",
    "monoexp": "One exponential fitted to all points (log-linear least squares), integrated from time 0",
    "biexp": "Sum of two exponentials fitted to all points (needs 4 or more points), integrated from time 0",
}


def decay_const(half_life_h: float) -> float:
    return float(np.log(2.0) / half_life_h)


def time_integrated_activity(times_h, acts_mbq, method: str = "trapezoid_effective", half_life_h: float = 159.53,
                             from_zero: bool = True) -> dict:
    """Returns tia_MBq_h, effective half-life (h, when defined), the method and a list of notes."""
    t, a = np.asarray(times_h, float), np.asarray(acts_mbq, float)
    order = np.argsort(t)
    t, a = t[order], a[order]
    if len(t) < 2 or np.any(a < 0):
        raise ValueError("at least two time points with non-negative activity are needed")
    if len(set(t.tolist())) != len(t):
        raise ValueError("two measurements have the same time")
    notes, lam_phys = [], decay_const(half_life_h)
    if method == "monoexp":
        if np.any(a <= 0):
            raise ValueError("the exponential fit needs activities above zero")
        slope, intercept = np.polyfit(t, np.log(a), 1)
        lam = -slope
        if lam <= 0:
            raise ValueError("the activity does not decrease, so no exponential fit is possible")
        a0 = float(np.exp(intercept))
        pred = intercept + slope * t
        ss_res, ss_tot = float(np.sum((np.log(a) - pred) ** 2)), float(np.sum((np.log(a) - np.log(a).mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        if lam < lam_phys:
            notes.append("fitted clearance is slower than physical decay; check the data")
        if len(t) == 2:
            notes.append("two points fit exactly; the uncertainty is large")
        return {"tia_MBq_h": a0 / lam, "t_eff_h": float(np.log(2.0) / lam), "r2": r2, "method": method, "notes": notes}
    if method == "biexp":
        if len(t) < 4 or np.any(a <= 0):
            raise ValueError("the two-exponential fit needs at least 4 points with activity above zero")
        from scipy.optimize import curve_fit

        def f(x, lo1, k1, lo2, k2):
            return np.exp(lo1) * np.exp(-np.exp(k1) * x) + np.exp(lo2) * np.exp(-np.exp(k2) * x)
        slope, icpt = np.polyfit(t, np.log(a), 1)
        # Each component clears at least as fast as physical decay (biological clearance >= 0) and not faster than a
        # 3-minute half-life. Without this lower bound noisy data can give a component that hardly decays and an
        # integral many orders of magnitude too large.
        k_lo, k_hi = np.log(lam_phys), np.log(np.log(2.0) / 0.05)
        p0 = [icpt, np.clip(np.log(max(-slope * 3, 1e-4)), k_lo, k_hi), icpt - 1.0,
              np.clip(np.log(max(-slope / 3, 1e-5)), k_lo, k_hi)]
        bounds = ([-np.inf, k_lo, -np.inf, k_lo], [np.inf, k_hi, np.inf, k_hi])
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(f, t, a, p0=p0, bounds=bounds, maxfev=20000)
        except (RuntimeError, ValueError) as e:
            raise ValueError(f"the two-exponential fit did not converge ({e})")
        a1, l1, a2, l2 = np.exp(popt[0]), np.exp(popt[1]), np.exp(popt[2]), np.exp(popt[3])
        if min(l1, l2) <= lam_phys * (1 + 1e-6):
            notes.append("one component decays with the physical half-life only (no biological clearance found)")
        pred = f(t, *popt)
        ss_res, ss_tot = float(np.sum((a - pred) ** 2)), float(np.sum((a - a.mean()) ** 2))
        if len(t) == 4:
            notes.append("four points for four parameters: the fit passes through them; the uncertainty is large")
        return {"tia_MBq_h": float(a1 / l1 + a2 / l2), "t_eff_h": float(np.log(2.0) / min(l1, l2)),
                "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0, "method": method, "notes": notes}
    if method not in ("trapezoid_effective", "trapezoid_physical"):
        raise ValueError(f"unknown method {method}")
    tt, aa = (np.r_[0.0, t], np.r_[0.0, a]) if (from_zero and t[0] > 0) else (t, a)
    if from_zero and t[0] > 0:
        notes.append("activity assumed to rise linearly from zero at time 0 to the first measurement")
    area = float(np.sum((aa[1:] + aa[:-1]) / 2.0 * np.diff(tt)))      # trapezoid rule
    lam_tail, t_eff = lam_phys, None
    if method == "trapezoid_effective" and len(t) >= 2 and a[-2] > 0 and a[-1] > 0 and a[-1] < a[-2]:
        lam_tail = float(np.log(a[-2] / a[-1]) / (t[-1] - t[-2]))
        if lam_tail < lam_phys:               # slower than physical decay is impossible (noise in the last points)
            notes.append("the last two points fall slower than physical decay: physical decay used after the last point")
            lam_tail = lam_phys
        t_eff = float(np.log(2.0) / lam_tail)
    elif method == "trapezoid_effective":
        notes.append("the last two points do not decrease: physical decay used after the last point")
    tail = a[-1] / lam_tail
    share = tail / (area + tail) if area + tail > 0 else 0.0
    if share > 0.3:
        notes.append(f"{100 * share:.0f} % of the integral is extrapolated after the last scan; add a later time point if you can")
    return {"tia_MBq_h": area + tail, "t_eff_h": t_eff, "r2": None, "method": method, "notes": notes}


def dose_local_deposition_gy(tia_mbq_h: float, mass_g: float, energy_mev: float) -> float:
    """Gy = decays x energy per decay / mass; decays = A~ (MBq*h) x 1e6 x 3600."""
    return float(tia_mbq_h) * 1e6 * 3600.0 * energy_mev * MEV_J / (mass_g / 1000.0)


def dose_svalue_gy(tia_mbq_h: float, s_gy_per_mbq_h: float) -> float:
    return float(tia_mbq_h) * float(s_gy_per_mbq_h)


def dosimetry_table(points, method: str, half_life_h: float, energy_mev: float, density_g_ml: float = 1.0,
                    admin_mbq: float | None = None, s_value: float | None = None, from_zero: bool = True,
                    weighted_energy_mev: float | None = None):
    """points: list of dicts {target, time_h, activity_MBq, volume_ml}. One result row per target."""
    import pandas as pd
    out = []
    for name in dict.fromkeys(p["target"] for p in points):
        pts = [p for p in points if p["target"] == name]
        row = {"target": name, "n_points": len(pts)}
        try:
            r = time_integrated_activity([p["time_h"] for p in pts], [p["activity_MBq"] for p in pts], method, half_life_h, from_zero)
        except ValueError as e:
            row["note"] = str(e)
            out.append(row)
            continue
        vols = [p.get("volume_ml") for p in pts if p.get("volume_ml")]
        dens = [p.get("density_g_ml") for p in pts if p.get("density_g_ml")]
        rho = float(np.mean(dens)) if dens else density_g_ml              # from CT when given, else the global value
        mass = float(np.mean(vols)) * rho if vols else None
        row.update({"tia_MBq_h": r["tia_MBq_h"], "t_eff_h": r["t_eff_h"], "fit_r2": r["r2"], "mass_g": mass})
        if admin_mbq:
            row["residence_time_h"] = r["tia_MBq_h"] / admin_mbq
        if s_value:
            row["dose_Gy"] = dose_svalue_gy(r["tia_MBq_h"], s_value)
        elif mass:
            row["dose_Gy"] = dose_local_deposition_gy(r["tia_MBq_h"], mass, energy_mev)
            if "marrow" in str(name).lower():
                r["notes"].append("local deposition is not valid for red marrow (trabecular bone): it overestimates the "
                                  "self-dose (about 1.75 x for Lu-177, EANM 2022); use an S-value")
            if weighted_energy_mev and abs(weighted_energy_mev - energy_mev) > 1e-9:
                row["dose_Gy_RBE"] = dose_local_deposition_gy(r["tia_MBq_h"], mass, weighted_energy_mev)
        if admin_mbq and row.get("dose_Gy") is not None:
            row["dose_Gy_per_GBq"] = row["dose_Gy"] / (admin_mbq / 1000.0)
            if "dose_Gy_RBE" in row:
                row["dose_Gy_RBE_per_GBq"] = row["dose_Gy_RBE"] / (admin_mbq / 1000.0)
        row["note"] = "; ".join(r["notes"])
        out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------- MIRD organ-level dose
S_UNITS = {"mGy/(MBq*h)": 1e-3, "mGy/(MBq*s)": 1e-3 * 3600.0, "Gy/(Bq*s)": 1e6 * 3600.0}      # factor to Gy per (MBq*h)


ICRP103_W = {"red marrow": 0.12, "colon": 0.12, "lungs": 0.12, "stomach": 0.12, "breast": 0.12, "gonads": 0.08,
             "urinary bladder": 0.04, "esophagus": 0.04, "liver": 0.04, "thyroid": 0.04, "bone surface": 0.01,
             "brain": 0.01, "salivary glands": 0.01, "skin": 0.01}
ICRP60_W = {"gonads": 0.20, "red marrow": 0.12, "colon": 0.12, "lungs": 0.12, "stomach": 0.12, "urinary bladder": 0.05,
            "breast": 0.05, "liver": 0.05, "esophagus": 0.05, "thyroid": 0.05, "bone surface": 0.01, "skin": 0.01}
REMAINDER_W = {"ICRP103": 0.12, "ICRP60": 0.05}
_ALIAS = {"red bone marrow": "red marrow", "marrow": "red marrow", "lung": "lungs", "bladder": "urinary bladder",
          "urinary bladder wall": "urinary bladder", "bladder wall": "urinary bladder", "salivary": "salivary glands",
          "bone surfaces": "bone surface", "osteogenic cells": "bone surface", "oesophagus": "esophagus",
          "stomach wall": "stomach", "colon wall": "colon", "breasts": "breast", "thyroid gland": "thyroid"}


def _norm(name: str) -> str:
    n = str(name).strip().lower()
    return _ALIAS.get(n, n)


def effective_dose_sv(doses_gy: dict, scheme: str = "ICRP103"):
    """Simplified effective dose: sum of w_T x D_T over the tissues present; gonads = mean of testes/ovaries/gonads;
    remainder tissues = arithmetic mean of all targets that have no weight of their own. Returns (Sv, notes). Photon
    and electron weighting factor 1, so this is only meaningful for beta/gamma emitters (for alpha emitters it is not)."""
    w = ICRP103_W if scheme == "ICRP103" else ICRP60_W
    d = {_norm(k): float(v) for k, v in doses_gy.items()}
    gon = [d[k] for k in ("testes", "ovaries", "gonads") if k in d]
    if gon:
        d["gonads"] = float(np.mean(gon))
    notes, total = [], 0.0
    for tissue, wt in w.items():
        if tissue in d:
            total += wt * d[tissue]
        else:
            notes.append(f"no dose for '{tissue}' (weight {wt} counted as zero)")
    rest = [v for k, v in d.items() if k not in w and k not in ("testes", "ovaries")]
    if rest:
        total += REMAINDER_W[scheme] * float(np.mean(rest))
    else:
        notes.append("no remainder tissues in the matrix")
    return total, notes


def remainder_tia(admin_mbq: float, whole_body_tau_h: float, tia_mbq_h: dict) -> float:
    """Time-integrated activity in the rest of the body = whole body minus the organs entered (never below zero)."""
    return max(admin_mbq * whole_body_tau_h - float(sum(tia_mbq_h.values())), 0.0)


def mird_organ_doses(tia_mbq_h: dict, s_matrix, unit: str = "mGy/(MBq*h)", weights: dict | None = None,
                     mass_scale: dict | None = None):
    """MIRD schema D(target) = sum over sources of A~(source) x S(target <- source).
    tia_mbq_h: time-integrated activity per SOURCE organ (MBq*h). s_matrix: DataFrame with targets as rows and sources
    as columns, taken from a reference you are licensed to use (OLINDA/EXM, IDAC, MIRDcalc). Sources without a column
    are reported, never silently ignored. weights: optional tissue weighting factors per target for an effective dose
    (sum of w_T x D_T; the quantity is only as meaningful as the S-values and weights you supply)."""
    import pandas as pd
    f = S_UNITS[unit]
    missing = [s for s in tia_mbq_h if s not in s_matrix.columns]
    use = [s for s in tia_mbq_h if s in s_matrix.columns]
    if not use:
        raise ValueError("none of the source organs matches a column of the S-value matrix")
    def col(src):
        c = s_matrix[src].astype(float) * f * tia_mbq_h[src]
        if mass_scale and src in mass_scale and src in c.index:       # self-dose of an organ scales with 1/mass (non-penetrating)
            c = c.copy()
            c[src] = c[src] * float(mass_scale[src])
        return c
    dose = sum(col(s) for s in use)
    df = pd.DataFrame({"target": s_matrix.index, "dose_Gy": dose.values})
    for s in use:
        df[f"from_{s}_Gy"] = col(s).values
    df.attrs["sources_missing_in_matrix"] = missing
    if weights:
        w = df["target"].map(weights).fillna(0.0)
        df.attrs["effective_dose_Sv"] = float((w * df["dose_Gy"]).sum())
    return df
