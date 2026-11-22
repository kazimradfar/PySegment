# PySegment – validation of the dosimetry module

Reproduce: `python tools/validate_dosimetry.py --icrp107 <RadData>/data` (a few seconds; `pip install radioactivedecay
pyreadr`; RadData from https://github.com/markhogue/RadData). Code version and environment are recorded in `summary.json`.

## A. Nuclear data

Half-lives against ICRP 107 (data of the `radioactivedecay` package):

| Nuclide | PySegment (h) | ICRP 107 (h) | Difference |
|---|---|---|---|
| Lu-177 | 159.53 | 159.528 | +0.00 % |
| Y-90 | 64.05 | 64.1 | -0.08 % |
| I-131 | 192.5 | 192.497 | +0.00 % |
| Ac-225 | 238.1 | 240 | -0.79 % |
| Ra-223 | 274.3 | 274.32 | -0.01 % |
| Tc-99m | 6.0067 | 6.015 | -0.14 % |
| F-18 | 1.8295 | 1.8295 | +0.00 % |
| Ga-68 | 1.1285 | 1.1285 | +0.00 % |

The largest difference (Ac-225, −0.8 %) is the difference between the ICRP 107 evaluation (10.0 d) and the newer
NuDat value used by PySegment (9.92 d).

Mean energy per decay (`_d`: daughters shorter-lived than the parent, weighted by branching) against ICRP 107
(Eckerman & Endo 2008; data files of the RadData package, `energies_icrp107.csv`):

| Nuclide | Component | PySegment (MeV) | ICRP 107 (MeV) | Difference |
|---|---|---|---|---|
| Lu-177 | electron | 0.1479 | 0.14792 | -0.01 % |
| Lu-177 | photon | 0.0351 | 0.03509 | +0.03 % |
| Y-90 | electron | 0.9331 | 0.9331 | +0.00 % |
| I-131 | electron | 0.1918 | 0.19184 | -0.02 % |
| I-131 | photon | 0.3827 | 0.38275 | -0.01 % |
| Ac-225 | alpha | 5.8919 | 5.8919 | +0.00 % |
| Ac-225 | electron | 0.0248 | 0.02476 | +0.16 % |
| Ac-225 | photon | 0.0171 | 0.01708 | +0.12 % |
| Ac-225 | alpha_d | 22.1026 | 22.1026 | +0.00 % |
| Ac-225 | electron_d | 0.6648 | 0.66475 | +0.01 % |
| Ac-225 | photon_d | 0.2022 | 0.20217 | +0.02 % |
| Ra-223 | alpha | 5.7701 | 5.7701 | +0.00 % |
| Ra-223 | electron | 0.0781 | 0.07809 | +0.01 % |
| Ra-223 | photon | 0.1413 | 0.14126 | +0.03 % |
| Ra-223 | alpha_d | 21.1027 | 21.1027 | -0.00 % |
| Ra-223 | electron_d | 0.9649 | 0.96491 | -0.00 % |
| Ra-223 | photon_d | 0.1728 | 0.1728 | +0.00 % |
| Tc-99m | electron | 0.0162 | 0.01618 | +0.12 % |
| Tc-99m | photon | 0.1266 | 0.12659 | +0.01 % |
| F-18 | electron | 0.2416 | 0.2416 | +0.00 % |
| F-18 | photon | 0.9886 | 0.98858 | +0.00 % |
| Ga-68 | electron | 0.7379 | 0.73794 | -0.01 % |
| Ga-68 | photon | 0.9487 | 0.94869 | +0.00 % |

All within 0.16 % (rounding to four decimals). **Error found and fixed:** before this check the table held indicative
values – Lu-177 had no photon energy (so the photon absorbed fraction of the program had no effect at all), and for
Ac-225 and Ra-223 the parent photons were missing and the alpha energy of the chain was 1.3 % low (28.3 instead of
28.7 MeV per decay for Ac-225 with all daughters retained). The table now holds the ICRP 107 values (regression test
`test_lu177_photon_fraction_counts`). Lu-177 electrons: 0.1479 MeV = 0.08531 mJ/(MBq h); EANM
0.0853 mJ/(MBq h).

## B. Local energy deposition against published S-values (Lu-177)

Self-dose S-values in mGy/(MBq h), IDAC-Dose 2.1 adult male, from Table 2 of Sjögreen Gleisner et al.,
Eur J Nucl Med Mol Imaging 2022;49:1778–1809 (EANM recommendations; values transcribed, to be checked against the
PDF before submission):

| Organ (mass) | IDAC-Dose 2.1 | PySegment, local deposition | Difference |
|---|---|---|---|
| kidneys (422 g) | 0.204 | 0.2023 | -0.8 % |
| liver (2360 g) | 0.0376 | 0.0362 | -3.8 % |
| spleen (228.4 g) | 0.377 | 0.3738 | -0.9 % |
| red marrow (1394 g) | 0.0349 | 0.0612 | +75.5 % |

For compact organs of 200–500 g the local-deposition approximation of PySegment agrees within 1 %, as the EANM
recommendation states (1–2 % for convex regions of about 2–300 g). The liver (−3.8 %) receives some photon
self-dose that local deposition leaves out. For red marrow local deposition is not valid (trabecular geometry; the
EANM ratio is 0.57): PySegment now writes a warning for targets named "marrow" and recommends an S-value.

## C. Time-integrated activity

Illustrative Lu-177 kinetics with an exact integral (not patient data), sampled at two schedules; every method of
the program; noise-free and with multiplicative Gaussian noise (5 % and 10 %, 2000 Monte Carlo runs each; all
numbers in `tia_accuracy.csv`).

**5 points (4, 24, 48, 96, 168 h)** – bias (%) without noise / RMSE (%) with 10 % noise

| Kinetics | trapezoid_effective | trapezoid_physical | monoexp | biexp |
|---|---|---|---|---|
| Mono-exp., T_eff 40 h | +0.2 / 4.7 | +16.5 / 17.5 | -0.0 / 4.6 | +0.0 / 8.0 |
| Mono-exp., T_eff 100 h | -0.7 / 11.4 | +17.9 / 19.4 | +0.0 / 7.7 | +0.0 / 12.4 |
| Bi-exp. (30 % 3 h + 70 % 45 h) | +0.3 / 4.6 | +19.0 / 19.7 | +0.8 / 4.5 | +0.0 / 10.2 |
| Uptake 3 h, washout 60 h | -1.1 / 5.3 | +24.0 / 25.0 | -0.3 / 5.3 | +20.2 / 22.7 |

**3 points (24, 96, 168 h)** – bias (%) without noise / RMSE (%) with 10 % noise

| Kinetics | trapezoid_effective | trapezoid_physical | monoexp |
|---|---|---|---|
| Mono-exp., T_eff 40 h | -12.7 / 14.0 | +3.6 / 7.3 | -0.0 / 6.9 |
| Mono-exp., T_eff 100 h | -7.2 / 12.0 | +11.4 / 13.4 | +0.0 / 8.1 |
| Bi-exp. (30 % 3 h + 70 % 45 h) | -14.3 / 15.0 | +4.3 / 7.6 | -2.6 / 7.0 |
| Uptake 3 h, washout 60 h | -5.7 / 8.1 | +19.4 / 20.6 | +5.0 / 7.9 |

Findings:

1. With the five-point schedule the default method (trapezoids + effective tail) and the mono-exponential fit are
   within ±1 % without noise; the bi-exponential fit reproduces mono- and bi-exponential curves exactly.
2. **Late schedules (first scan at 24 h) and the trapezoid method:** the activity is assumed to rise linearly from
   zero at injection, which underestimates the integral by 6–14 % when the uptake is fast. For such schedules the
   mono-exponential fit is better (bias ≤ 5 %).
3. **trapezoid_physical** (no biological clearance after the last scan) is conservative by design: +4 to +24 %.
4. **Uptake then washout** cannot be represented by a sum of two positive exponentials (biexp: +20 %); a dedicated
   uptake–washout model is not implemented (limitation).
5. **Two fitting errors found and fixed by this validation** (regression tests in `tests/test_rsw_dosimetry.py`):
   - the bi-exponential fit allowed a component that decays slower than physical decay; with noisy data it
     returned integrals up to 10⁶ times too large without any warning. Every component is now bounded by the
     physical decay constant (and a 3-minute half-life at the fast end);
   - the tail of the trapezoid method used the slope of the last two points even when it was slower than physical
     decay (noise), which gave a 32 % RMSE for slow kinetics; it now falls back to physical decay with a note
     (RMSE 11 %).

## D. Formulas and units

Absorbed dose by local deposition, S-value unit factors (mGy/(MBq s), Gy/(Bq s)), MIRD organ doses for a 2-source ×
3-target matrix (with an unmatched source reported), remainder of body and the ICRP 103 / ICRP 60 tissue weights
(sum 1.00; effective dose 1 Sv when every tissue receives 1 Gy) were compared with independent hand calculations:
all pass (`summary.json`, `formulas_and_units`).

## Still to do (see `validation/VALIDATION_PLAN.md`)

- Sphere S-values (OLINDA/EXM sphere model) for lesion doses, and an end-to-end comparison with published patient
  dosimetry (same time–activity data, OLINDA/EXM or MIRDcalc), ideally the SNMMI Lu-177 Dosimetry Challenge.
