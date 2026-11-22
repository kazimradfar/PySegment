# PySegment – validation with the QIBA FDG-PET/CT Digital Reference Objects

Reproduce: `python tools/validate_qiba_dro.py --dro <DRO_20130529> --droe <DROe_20140714> --dromtv <DROmtv_20160428>` (packages from
https://depts.washington.edu/petctdro/; the images are not stored in the repository). Code version in
`summary.json`.

Reference: Pierce LA, Elston BF, Clunie DA, Nelson D, Kinahan PE. A digital reference object to analyze calculation
accuracy of PET standardized uptake value. Radiology 2015;277:538–545. DRO user guide (version 05/29/2013,
updated June 2019), tables 2 and 4.

## 1. SUVbw from DICOM and ROI statistics (DRO, female 20130529 and male 20130528)

The PET DICOM series is read by PySegment, which converts it to SUVbw from the DICOM header (weight, injected
activity, injection and acquisition times, F-18 half-life). Result: background exactly 1.00, hot spheres 4.00 and
the two test voxels 4.11 and −0.11 for both DROs (59.94 kg and 70 kg), as the DRO defines.

The six ROIs of the DRO report are placed automatically on slice 40 (centre of each object rounded to the nearest
voxel, as a user would click) with PySegment's own ROI shapes: circles of 25 mm diameter (ROIs 1–5) and a sphere of
25 mm diameter (ROI 6); statistics from `rsw.measure.roi_stats` (SD with n − 1). Acceptance bounds in brackets (for
software reporting two or more decimals). Female DRO; the male DRO gives identical values.

| ROI | Definition | Voxels (size) | Max | Min | Mean | SD |
|---|---|---|---|---|---|---|
| 1 | circle, smallest hot sphere | 129 (492 mm2) | 4.000 [3.995-4.005] | 0.637 [0.635-0.645] | 1.379 [1.295-1.505] | 0.970 [0.875-1.095] |
| 2 | circle, largest hot sphere | 129 (492 mm2) | 4.000 [3.995-4.005] | 4.000 [3.995-4.005] | 4.000 [3.995-4.005] | 0.000 [-0.005-0.005] |
| 3 | circle, hot test voxel | 129 (492 mm2) | 4.110 [4.105-4.115] | 1.000 [0.995-1.005] | 1.024 [1.015-1.035] | 0.274 [0.235-0.325] |
| 4 | circle, cold test voxel | 129 (492 mm2) | 1.000 [0.995-1.005] | -0.110 [-0.115--0.105] | 0.991 [0.985-0.995] | 0.098 [0.085-0.115] |
| 5 | circle, 2D test pattern | 129 (492 mm2) | 0.900 [0.895-0.905] | 0.100 [0.095-0.105] | 0.528 [0.465-0.535] | 0.401 [0.395-0.405] |
| 6 | sphere, 3D test pattern | 1037 (7912 mm3) | 0.900 [0.895-0.905] | 0.100 [0.095-0.105] | 0.522 [0.475-0.525] | 0.400 [0.395-0.405] |

**48/48 values within the bounds** (both DROs). The circle covers 129 voxels = 492 mm²
(DRO: 490 mm²); the voxelised sphere 7912 mm³ (DRO: 8181 mm³). A voxel belongs to an ROI when its centre lies
inside the circle or sphere. `dro_slice40_rois.png` shows the placed ROIs.

## 2. SUVpeak (DROe, PET slice 49)

PySegment's SUVpeak is the highest mean in a 1 mL sphere whose centre lies anywhere in the lesion (PERCIST); the
voxelised sphere has 21 voxels = 1.008 mL on the 4 × 4 × 3 mm DROe grid.
It was compared with an independent brute-force implementation (every sphere position evaluated directly).

| Blob | Max | SUVpeak PySegment | Reference | Sphere at the hottest voxel |
|---|---|---|---|---|
| Left (uniform) | 4.0013 | 3.4488 | 3.4488 | 3.4488 |
| Right (off-centre hot voxel) | 4.3011 | 3.4488 | 3.4488 | 2.9862 |

PySegment agrees with the reference to 4 decimals. The right blob is the purpose of this DROe test: software that
centres the sphere on the hottest voxel reports 2.9862 instead of
3.4488 (−13 %); PySegment does not.

## 3. PET/CT fusion alignment (DROe, CT slice 64)

The PET volume (4 × 4 × 3 mm) is brought onto the CT grid (2 × 2 × 1.5 mm) by PySegment's fusion path
(`rsw.io.resample_to`) and the alignment stripes of both are located. Vertical stripes: 5
in CT, 5 in PET, offsets [0.0, 0.0, 0.0, 0.0, 0.0] mm; horizontal stripes:
5 / 5, offsets [0.0, 0.0, 0.0, 0.0, 0.0] mm.
The PET and CT geometry from DICOM (position, orientation, spacing, slice order) is therefore handled without any
shift.

## 4. Metabolic tumour volume (DROmtv 20160428)

The DROmtv holds six spheres at 4:1 and six at 2:1 contrast against a background of SUV 1.0, a cube, an irregular
object, a smooth blob and a thin lobed object, noise-free and with noise (6 mm FWHM). No reference values come with
the package. Because the noise-free voxels hold background + (level − background) × occupied fraction, the **true
volume** of every object with a uniform level is the sum of these fractions times the voxel volume. For the spheres this
gives exactly the NEMA volumes (0.524, 1.150, 2.569, 5.575, 11.494 mL); the largest sphere holds 23.7 mL (the nominal
37 mm sphere would be 26.5 mL), so the image content is used as the truth. The blob and the lobed object have no single
level, so only their measured volumes are listed (`dromtv_results.csv`).

PySegment's region growing was started at the centre of each object with four settings: 41 % and 50 % of the local
maximum, an absolute SUV of 2.5, and an absolute threshold half-way between the background (1.0) and the object level.

**noise-free** – MTV in mL (error vs true volume)

| Object | True | 41 % of max | 50 % of max | SUV 2.5 | half-way |
|---|---|---|---|---|---|
| sphere 1, 4:1 | 0.524 | 0.67 (+28 %) | 0.59 (+13 %) | 0.51 (-2 %) | 0.51 (-2 %) |
| sphere 2, 4:1 | 1.150 | 1.38 (+20 %) | 1.30 (+13 %) | 1.10 (-4 %) | 1.10 (-4 %) |
| sphere 3, 4:1 | 2.569 | 3.01 (+17 %) | 2.78 (+8 %) | 2.53 (-2 %) | 2.53 (-2 %) |
| sphere 4, 4:1 | 5.575 | 6.30 (+13 %) | 5.98 (+7 %) | 5.47 (-2 %) | 5.47 (-2 %) |
| sphere 5, 4:1 | 11.494 | 12.69 (+10 %) | 12.14 (+6 %) | 11.52 (+0 %) | 11.52 (+0 %) |
| sphere 6, 4:1 | 23.733 | 26.11 (+10 %) | 25.25 (+6 %) | 23.78 (+0 %) | 23.78 (+0 %) |
| sphere 1, 2:1 | 0.524 | refused | refused | 0 (none) | 0.51 (-2 %) |
| sphere 2, 2:1 | 1.150 | refused | refused | 0 (none) | 1.12 (-3 %) |
| sphere 3, 2:1 | 2.569 | refused | refused | 0 (none) | 2.53 (-2 %) |
| sphere 4, 2:1 | 5.575 | refused | refused | 0 (none) | 5.47 (-2 %) |
| sphere 5, 2:1 | 11.494 | refused | refused | 0 (none) | 11.52 (+0 %) |
| sphere 6, 2:1 | 23.528 | refused | refused | 0 (none) | 23.55 (+0 %) |
| cube | 46.656 | 46.66 (+0 %) | 46.66 (+0 %) | 46.66 (+0 %) | 46.66 (+0 %) |
| irregular | 3.931 | 4.90 (+24 %) | 4.38 (+12 %) | 3.74 (-5 %) | 3.74 (-5 %) |

**noisy, 6 mm FWHM** – MTV in mL (error vs true volume)

| Object | True | 41 % of max | 50 % of max | SUV 2.5 | half-way |
|---|---|---|---|---|---|
| sphere 1, 4:1 | 0.524 | refused | refused | 0.13 (-76 %) | 0.13 (-76 %) |
| sphere 2, 4:1 | 1.150 | 1.74 (+51 %) | 1.20 (+4 %) | 0.64 (-44 %) | 0.64 (-44 %) |
| sphere 3, 4:1 | 2.569 | 3.66 (+43 %) | 2.62 (+2 %) | 2.00 (-22 %) | 2.00 (-22 %) |
| sphere 4, 4:1 | 5.575 | 7.25 (+30 %) | 5.58 (+0 %) | 4.73 (-15 %) | 4.73 (-15 %) |
| sphere 5, 4:1 | 11.494 | 14.45 (+26 %) | 11.87 (+3 %) | 10.30 (-10 %) | 10.30 (-10 %) |
| sphere 6, 4:1 | 23.733 | 29.99 (+26 %) | 24.96 (+5 %) | 22.13 (-7 %) | 22.13 (-7 %) |
| sphere 1, 2:1 | 0.524 | refused | refused | 0 (none) | 0.38 (-27 %) |
| sphere 2, 2:1 | 1.150 | refused | refused | 0 (none) | 0.89 (-23 %) |
| sphere 3, 2:1 | 2.569 | refused | refused | 0.05 (-98 %) | 2.58 (+0 %) |
| sphere 4, 2:1 | 5.575 | refused | refused | 0 (none) | 4.86 (-13 %) |
| sphere 5, 2:1 | 11.494 | refused | refused | 0 (none) | 9.78 (-15 %) |
| sphere 6, 2:1 | 23.528 | refused | refused | 0 (none) | 21.30 (-10 %) |
| cube | 46.656 | 55.34 (+19 %) | 45.50 (-2 %) | 44.79 (-4 %) | 44.79 (-4 %) |
| irregular | 3.931 | refused | 5.54 (+41 %) | 1.75 (-55 %) | 1.75 (-55 %) |

Median absolute error (objects with a true volume that were measured): noise-free 41 % 15.1 %,
50 % 7.8 %, half-way 1.7 %;
noisy 41 % 28.2 %, 50 % 3.3 %,
half-way 15.1 %.

Findings:

1. A threshold half-way between background and object recovers the true volume within 5 % in the noise-free DRO
   (all 14 objects), as expected for partial-volume voxels. With noise and 6 mm smoothing the small spheres are
   underestimated (partial-volume effect of the blurring, not of the program).
2. 41 % and 50 % of the maximum overestimate the volume of 4:1 objects by 5–28 % (the threshold lies below the
   half-way level) – a property of the method, known from the literature.
3. **Error found and fixed:** for 2:1 objects a percentage of the maximum lies at background level. Without noise the
   region flooded the background and the existing guard (> 300 mL) stopped it, but with noise "50 % of max" returned,
   without any warning, 254 mL for an 11.5 mL sphere and 20 mL for a 5.6 mL sphere. Every region that leaked had a
   threshold below 1.5 × the local background (median of a 4–12 mm shell around the region), every region that did not
   leak had at least 1.64 × the background. PySegment now refuses a percentage threshold below 1.5 × the local
   background and tells the user to set an absolute threshold ("refused" in the tables; regression test in
   `tests/test_rsw_project.py`). Absolute thresholds are not restricted.
4. SUV 2.5 finds nothing in the 2:1 objects (their maximum is 2.0) and fits the 4:1 objects only because 2.5 happens to
   be half-way there – a fixed SUV threshold is not transferable between contrasts.

## Not tested here

The DROe ROI voxel-inclusion test (descriptive in the DROe report): PySegment includes a voxel when its centre is
inside the ROI.
