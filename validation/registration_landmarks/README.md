# 2c. Registration against expert landmarks (POPI 4D CT)

**Data.** POPI model (Léon Bérard Cancer Centre / CREATIS, Lyon): six patients, 4D CT of the thorax,
about 100 expert landmarks per phase (bh 113, mm2 107, the others 100). The fixed image is the end-inhale
phase (00) and the moving image the end-exhale phase (50; 60 for bl). The landmarks were never used by the
registration; they only measure the result. Error per landmark (TRE): |T(p_inhale) − p_exhale|.

**Split.** The settings of the elastix lung profile (2 mm isotropic grid, HU window −1000..200, lung mask,
normalised correlation, B-spline grid 10 mm, 500 iterations, 3000 random samples, seed 1234) were chosen on the
two *tuning* patients (bh, mm2) only. The four *test* patients (bl, dx, gt, ng) were run once with these
fixed settings.

**One change after the first test run (reported for transparency).** In test patient dx the airways reach
the edge of the image, so the first lung mask (air not connected to the image border) contained no lung and
the registration stopped with a sampler error. The mask was changed to "air inside the filled body outline"
(commit c0eab30, regression test `test_lung_mask_when_the_airways_reach_the_edge_of_the_image`); no other
setting was changed, and the elastix profile was re-run on **all six** patients with the new mask. The new
mask contains 100 % of the landmarks in both phases of every patient. Effect on the patients that had already
run: bh 0.97 → 0.97 mm, mm2 1.12 → 1.05 mm, bl 0.86 → 0.87 mm.

## Results (mean TRE per patient, mm)

| Patient | Set | No registration | SimpleITK rigid | SimpleITK affine | elastix lung B-spline | p95 | max | < 2 mm | time |
|---|---|---|---|---|---|---|---|---|---|
| bh  | tuning | 6.68 | 6.11 | 6.11 | **0.97** | 2.22 | 6.21 | 94 % | 169 s |
| mm2 | tuning | 7.09 | 7.00 | 6.63 | **1.05** | 2.22 | 6.79 | 93 % | 101 s |
| bl  | test   | 5.90 | 5.81 | 5.46 | **0.87** | 1.70 | 2.25 | 98 % |  99 s |
| dx  | test   | 7.67 | 7.42 | 6.78 | **1.04** | 2.35 | 7.99 | 93 % | 104 s |
| gt  | test   | 7.33 | 7.42 | 7.19 | **1.27** | 3.60 | 12.97 | 90 % |  79 s |
| ng  | test   | 14.04 | 14.14 | 13.97 | **1.81** | 4.19 | 15.15 | 77 % | 103 s |

p95, max, "< 2 mm" and time refer to the elastix lung profile (commit c0eab30; the elastix engine was not changed
later). Cloud workspace, 2 threads.

- **Test set (4 patients): mean TRE 8.74 → 1.25 mm** (range 0.87–1.81 mm). Tuning set: 6.88 → 1.01 mm.
  The test result is slightly worse than the tuning result, as expected for settings fixed beforehand.
- Rigid and affine registration (SimpleITK, default settings since commit 9c6c45e: rigid-first affine, 500
  iterations; re-run for all six patients) remove almost nothing (≤ 0.6 and ≤ 0.9 mm; in bh the affine stage ended
  worse than its rigid start, which was kept): the motion between inhale and exhale is
  a deformation, which only the B-spline can follow.
- ng has the largest breathing motion (14 mm mean, 32 mm max) and the largest residual error; a few landmarks
  per patient keep errors of 6–15 mm (gt, ng), most likely near the diaphragm and the lung base, where the
  motion is largest. Not checked landmark by landmark.
- Folding (Jacobian determinant ≤ 0): 0 voxels in bh, bl, gt; 2·10⁻⁶ to 4·10⁻⁵ of the voxels in mm2, dx, ng
  (fraction of the fixed grid). PySegment warns about it; the warning first printed this as "0.00 %", now as
  a voxel count with two significant digits.
- For comparison: the default SimpleITK B-spline on tuning patient bh gave 3.25 mm in 47 min, which is why the
  elastix engine was added.

## Reproduce

```
pip install itk-elastix==0.25.4
python tools/validate_registration_landmarks.py --popi <folder with bh, mm2, bl, dx, gt, ng>
python tools/validate_registration_landmarks.py --popi <folder> --rerun "elastix lung bspline"   # one method again
```

Files: `cases.csv` (one row per patient and method), `summary.json` (by tuning/test set, with commit and
versions). `raw/` (per-patient cache) is not committed.
