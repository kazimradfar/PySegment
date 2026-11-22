# PySegment – IBSI 1 validation of the radiomics module

Features were computed through PySegment's own code path (`rsw.radiomics.extract`) and compared with the
IBSI 1 consensus reference values and tolerances (Zwanenburg et al., Radiology 2020;295:328-338).
A value passes when |PySegment − reference| ≤ tolerance (plus half a unit of the 3rd significant digit,
because the reference values are published to 3 significant digits).

Reproduce: `python tools/validate_ibsi1.py --data <theibsi/data_sets> --ref <mirp/test/data/ibsi_1_reference_values>`
(PyRadiomics 3.1.1.dev111+g8ed5793, SimpleITK 2.5.8, NumPy 2.5.5, SciPy 1.18.1; Python 3.13, Linux).

Compared: 102 features per data set that PySegment computes and IBSI 1 defines – morphology 11,
statistics 15, intensity histogram 2, GLCM 23 (3D averaged), GLRLM 16 (3D averaged), GLSZM 16, NGTDM 5,
NGLDM 14 (PyRadiomics GLDM, α = 0). Not offered by PySegment: GLDZM, IVH, local intensity, 2D and 2.5D
aggregation, some morphology (Moran's I, Geary's C, bounding-volume densities).

## Results

| Data set | PySegment, resampling = PyRadiomics (default) | PySegment, resampling = IBSI |
|---|---|---|
| Digital phantom (no processing) | 100/102 | **102/102** |
| Chest CT, config C (2 mm, trilinear, [−1000, 400] HU, bin width 25 HU) | 21/102 | **102/102** |
| Chest CT, config D (2 mm, trilinear, 3σ outlier filter, 32 bins) | 19/102 | **99/102** |
| Chest CT, config E (2 mm, tricubic spline, range + 3σ, 32 bins) | 11/102 | **102/102** |

IBSI method (`Settings(resampling="ibsi")`, Radiomics tab → Method): the image and mask are
interpolated as IBSI specifies (grid centres aligned, mask interpolated trilinearly and thresholded at 0.5,
intensities rounded to whole HU for CT), range re-segmentation and outlier filtering are combined into one
interval, and only the grid points around each mask are evaluated (same values as resampling the whole
image; `tests/test_rsw_ibsi.py`). In PyRadiomics mode a value range and outlier filtering cannot be
combined, so config E uses only the range there. In the IBSI method the fixed bin size is also numbered from the
re-segmentation lower bound (or the ROI minimum when there is none) and the PCA axes use the covariance with N − 1,
as IBSI defines (deviations 1 and 3 below; resolved in the IBSI method, unchanged in the default method). The whole
validation runs in about 5 s.

## Explained deviations

1. **PCA minor and least axis (digital phantom, default method: 2 features).** PyRadiomics normalises the
   covariance by N, IBSI by N − 1: × √(74/73) gives 11.40 / 9.31 / 8.54 = the reference values. Elongation and
   flatness (ratios) are not affected. *Resolved in the IBSI method* (axis lengths × √(N/(N − 1))).
2. **Resampling (configs C–E, left column).** PyRadiomics aligns the new grid with the bounding box and
   resamples the mask with nearest neighbour; IBSI aligns grid centres and interpolates the mask. This
   changes the ROI (volume −1.8 %) and therefore every feature class. With the IBSI method, config E
   matches in all 102 features, which shows that the feature computation itself is IBSI-compliant.
3. **Fixed bin size numbering (config C, 22 features).** IBSI starts the first bin at the re-segmentation
   lower bound (−1000 HU); PyRadiomics starts it at floor(ROI minimum / width) × width (−950 HU here).
   The bin edges are the same, but grey levels are numbered 2 lower: joint average 36.98 + 2 = 38.98
   (reference 39.0), sum average 73.96 + 4 = 77.96 (78.0). Only features that depend on the grey-level
   value itself are affected (joint/sum average, autocorrelation, low/high grey-level emphases, ...);
   features that depend only on grey-level differences or counts are not. *Resolved in the IBSI method*
   (config C: 80/102 → 102/102); the default method keeps the PyRadiomics numbering, so results computed earlier
   with it do not change.
4. **Config D (3 features)** – szm_zsnu (+1.2 %), szm_zs_entr (−0.3 %), ngl_ldlge (+2 %): slightly outside
   the tolerance; zone and dependence features are sensitive to single boundary voxels after outlier
   filtering.

The data sets are public benchmarks (IBSI digital phantom, CC BY 4.0; IBSI chest CT phantom, CC BY-NC 3.0)
and are not stored in this repository; only the computed feature values are.
