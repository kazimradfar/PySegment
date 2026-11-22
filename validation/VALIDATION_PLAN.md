# PySegment – validation plan for the software paper

Goal: validate every module that produces a number a user could report, in the order in which the numbers are
produced (image → registration → measurement → dosimetry), with public reference data and the comparisons a
reviewer of a software paper (e.g. EJNMMI Physics, Physica Medica, Medical Physics) would expect. Every step has
a script in `tools/`, results in `validation/<step>/` (README with methods, results and explained deviations,
CSV tables, `summary.json` with the code version) and, where useful, a test in `tests/`.

Rules: public data only (images are not stored in the repository, only computed values); default settings of
the program unless the step tests an option; deviations are explained, never hidden; the installer is built only
after the final evaluation.

| # | Module | Reference | Status |
|---|---|---|---|
| 1a | Radiomics features | IBSI 1 (digital phantom, chest CT configs C–E; Zwanenburg 2020) | done – `validation/ibsi1/` (IBSI method 102, 102, 99, 102 of 102) |
| 1b | Radiomics filters | IBSI 2 phases 1–2 (LoG, wavelets; Whybra 2024) | done – `validation/ibsi2/` |
| 1c | Remaining IBSI 1 deviations | config C grey-level numbering (22 features), PCA axes N vs N−1 (2 features) | done – resolved in the IBSI method (phantom and config C 102/102); default method unchanged |
| 2a | Registration, known transforms | 51 real PET/CT/MR studies (IBSI STS set): CT→CT, PET→PET (rigid, affine, B-spline), hybrid PET/CT, PET→CT and MR→CT consistency, SUV after resampling | done – after the first run (150 iterations: CT rigid 1.60, affine 4.03 mm) rigid-first affine and 500 iterations; re-run: CT rigid 0.39, affine 0.30, PET rigid 0.79, affine 0.17, B-spline 0.55–0.56 mm; PET→CT consistency 2.25 mm, MR→CT 9.6 mm (not reliable on the crops) – `registration/README.md` |
| 2b | Landmark registration | FRE/TRE against the Fitzpatrick prediction | in 2a |
| 2c | Registration, expert landmarks | POPI 4D CT (Léon Bérard / CREATIS, 6 patients, ~100 landmarks inhale/exhale; DIR-Lab not reachable from Iran) | done – elastix lung profile, settings fixed on tuning patients (bh, mm2: 6.88 → 1.01 mm), test patients (bl, dx, gt, ng): 8.74 → 1.25 mm (0.87–1.81), ~100 s per patient; rigid/affine ≤ 0.8 mm gain; lung mask fixed after dx, all patients re-run – `registration_landmarks/README.md` |
| 3a | SUV and ROI statistics | QIBA FDG-PET/CT Digital Reference Object (DRO, Pierce 2015): SUVbw from DICOM, SUVmax/mean/min/SD in the 6 reference ROIs | done – `validation/qiba_dro/` (48/48 within bounds) |
| 3b | SUVpeak, PET/CT fusion, ROI fidelity | QIBA DROe (`DROe_20140714.zip`) | done – `validation/qiba_dro/` (SUVpeak = reference, fusion offset 0.0 mm) |
| 3c | Metabolic tumour volume (thresholds, PERCIST) | QIBA DROmtv (`DROmtv_20160428.zip`) | done – `validation/qiba_dro/` (half-way threshold ≤ 5 % noise-free; 1 leak error found and fixed) |
| 3d | SPECT counts → activity, decay | analytic digital phantoms (known counts, sensitivity, decay) | done – `validation/spect_counts/` (exact; counts kept within 0.1 %) |
| 4a | Dosimetry, analytic and published S-values | time-integrated activity of known kinetics, ICRP 107 half-lives, Lu-177 local deposition vs EANM 2022 / IDAC-Dose 2.1 organ S-values, formulas, units, ICRP 103/60 weights | done – `validation/dosimetry/` (2 fit errors found and fixed) |
| 4b | Dosimetry, reference values | energies per decay vs ICRP 107 (done, ≤ 0.16 %, Lu-177 photon error fixed); sphere S-values (OLINDA/EXM sphere model); published organ doses with the same inputs (OLINDA/EXM, MIRDcalc) | partly done |
| 4c | Dosimetry, patient data set | SNMMI Lu-177 Dosimetry Challenge (public serial SPECT/CT; also real SPECT/CT registration) | to do – check access |
| 5 | Segmentation agreement | Dice, Jaccard, HD95 on shapes with analytic answers; ICC(2,1) against a reference implementation | done – `validation/agreement/` (identical to SimpleITK, MedPy, pingouin and Shrout & Fleiss) |
| 6 | Performance | run times of the main steps on a standard laptop | to do |

After step 6: manuscript draft (methods = this plan, results = the `validation/` READMEs), then the installer. Draft started: `manuscript/pysegment/` (EJNMMI Physics; registration values of 2a to be filled after the re-run).
