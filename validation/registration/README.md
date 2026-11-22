# 2a/2b. Registration with known transforms and between modalities (51 real studies)

**Data.** IBSI validation set: 51 soft-tissue sarcoma patients (TCIA, Vallières et al. 2015) with CT, FDG PET of
the same PET/CT session and T1-weighted MR, each cropped to about 5 cm around the tumour, with the tumour mask.
**Code.** `tools/validate_registration.py`, PySegment commit 276590e (rigid-first affine, 500 iterations per level),
settings as in the registration dialog (`summary.json`). Laptop (4 cores), 4 patients in parallel.

## Known transforms (exact ground truth)

A random transform D (rotation ≤ 10° per axis, translation ≤ 15 mm; affine also scaling 0.95–1.05 and shear ±0.03;
B-spline: smaller rigid motion plus a smooth deformation) was applied to a real image (cubic interpolation, added
noise); PySegment had to recover it. TRE at the tumour voxels = |D(R(q)) − q|, mean per patient.

| Pair | Mode | n | Start (median) | TRE median [IQR] | < 2 mm | > 5 mm | Worst patient | Time (median) |
|---|---|---|---|---|---|---|---|---|
| CT→CT | rigid | 51 | 15.2 mm | 0.39 [0.26–0.70] | 94 % | 0 | 3.89 | 325 s |
| CT→CT | affine | 51 | 16.1 | 0.30 [0.20–0.44] | 100 % | 0 | 0.99 | 467 s |
| CT→CT | B-spline | 6 | 9.3 | 0.56 [0.37–1.27] | 83 % | 0 | 2.01 | 1120 s |
| PET→PET | rigid | 51 | 15.9 | 0.79 [0.20–1.86] | 78 % | 1 | 9.21 | 9 s |
| PET→PET | affine | 51 | 15.4 | 0.17 [0.11–0.24] | 100 % | 0 | 0.40 | 11 s |
| PET→PET | B-spline | 51 | 8.1 | 0.55 [0.40–0.71] | 100 % | 0 | 1.27 | 116 s |

No folding; the start was never kept. Times are per patient with four patients running at the same time.

**Hybrid PET/CT** (CT→CT rigid, applied to the PET): tumour SUVmean differed from resampling with the true transform
by 0.14 % [0.06–0.29], SUVmax by 0.81 % [0.32–1.67]. The resampling itself (true transform vs original image)
changes SUVmax by 3.81 % [1.60–7.29] and SUVmean by 0.16 %.

## Different modalities (consistency, not accuracy)

The reference alignment (scanner for PET/CT, curators for MR) is not an exact ground truth. Consistency = distance
between the results from the original pair and from the same pair misaligned by a random rigid transform.

| Pair | Consistency median [IQR] | Distance from reference (median) | Start kept | Time |
|---|---|---|---|---|
| PET→CT | 2.25 [1.49–4.46] mm | 5.96 mm | 3/51 | 240 s |
| MR→CT | 9.63 [3.50–15.74] mm | 4.75 mm | 12/51 | 235 s |

PET→CT is moderately consistent; MR→CT on these small limb crops is not reliable (start kept in 12 patients because
every result was worse than the start). For serial PET/CT or SPECT/CT the hybrid option (CT→CT applied to the
functional image) is recommended; for MR→CT a landmark start.

## Landmark registration (Fitzpatrick prediction)

Six landmark pairs, localisation error 1 mm SD per axis, 50 trials per patient: observed RMS TRE at the tumour centre
1.07 [1.03–1.11] mm, predicted 1.07 [1.04–1.11] mm (observed/predicted 0.988).

## First run (150 iterations, affine from identity) and why it was repeated

The first run (commit 7113611, values kept in `raw_run1_150it/`) gave CT→CT rigid 1.60 [0.91–4.04] mm (10 patients
> 5 mm), affine 4.03 [1.91–5.28] mm (15 > 5 mm), PET→PET rigid 2.24 mm, affine 2.35 mm. On a POPI CT slab of the same
thickness the rigid optimiser stopped at 150 iterations with 1.4–2.0 mm and reached 0.17–0.23 mm when allowed to
converge (300–400 iterations); the full thoracic CT converged in 25 iterations. The crops of about 10 cm weakly
constrain rotations, so convergence is slow. Changes (commit 9c6c45e): affine starts from a rigid registration,
500 iterations per level. Effect: CT rigid 1.60 → 0.39 mm, CT affine 4.03 → 0.30 mm, PET rigid 2.24 → 0.79 mm,
PET affine 2.35 → 0.17 mm, PET B-spline 0.62 → 0.55 mm; CT B-spline (6 patients) 0.50 → 0.56 mm, unchanged within the
spread. Time per patient rose (CT rigid 127 → 325 s, affine 140 → 467 s, with four patients in parallel).

In the second run the limit of 500 iterations was still reached in most runs (CT rigid 33/51, PET rigid 45/51)
although the TRE was sub-millimetre, so the *Not converged* warning carried no information. Since commit 9d1e5c8 the
warning is given only if the image corners still move more than 0.05 mm over the last 20 iterations (POPI slab: 150
iterations → 0.83 mm, warning; 500 iterations → converged, no warning). This change only reads the optimiser; the
transforms and the values above are unchanged.

## Reproduce

```
python tools/validate_registration.py --data <ibsi_validation/nifti> --jobs 4 --bspline-ct 6
```
