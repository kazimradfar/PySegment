# PySegment – IBSI 2 validation of the image filters

Reference: Whybra P, et al. The Image Biomarker Standardization Initiative: standardized convolutional
filters for reproducible radiomics and enhanced clinical insights. Radiology 2024;310:e231319.
Reference response maps (phase 1) and reference feature values (phase 2) as distributed in the MIRP test
suite (https://github.com/oncoray/mirp, `test/`); chest CT phantom from https://github.com/theibsi/data_sets.

Reproduce: `python tools/validate_ibsi2.py --mirp-test <mirp/test> --data <theibsi/data_sets>`
(PyRadiomics 3.1.1.dev111+g8ed5793, SciPy 1.18.1, PyWavelets; runs in a few seconds).

PySegment offers the Laplacian of Gaussian (LoG) and the separable, non-rotation-invariant wavelets of
PyRadiomics. Mean, Laws, Gabor, rotation-invariant (pooled) wavelets and Simoncelli filters are not offered
and were not tested.

## Phase 1 – response maps of the digital phantoms (criterion: every voxel within 0.001)

| Config | Filter | PySegment, PyRadiomics method | PySegment, IBSI method |
|---|---|---|---|
| 2A | LoG σ 3 mm, impulse, constant boundary | max diff 8.2; response × 2.27 | **max diff 0.0002; all voxels pass** |
| 2B | LoG σ 5 mm, checkerboard, reflect | max diff 137; response × 6.31 | max diff 0.0018 (0.007 % of the response range); 77.8 % of voxels ≤ 0.001 |
| 5A1 | Daubechies 2, LHL, level 1, impulse | **max diff 4 × 10⁻⁶; all pass** | same (PyRadiomics wavelet) |
| 6A1 | Coiflet 1, HHL, level 1, sphere | **max diff 8 × 10⁻⁶; all pass** | same (PyRadiomics wavelet) |

## Phase 2 – intensity statistics of the filtered chest CT phantom (3D, 1 mm tricubic spline, [−1000, 400] HU)

| Config | PyRadiomics method | IBSI method |
|---|---|---|
| 1.B (no filter) | 14/15 | **15/15** |
| 3.B (LoG σ 1.5 mm, reflect) | 2/15 | **15/15** |

Compared: the 15 IBSI intensity statistics that PySegment computes (not offered: coefficient of variation,
quartile coefficient of dispersion, median absolute deviation).

## Explained deviations

1. **LoG scale (PyRadiomics method).** PyRadiomics normalises the LoG across scales (multiplies by σ² in
   voxel units: 1.5² = 2.25 and 2.5² = 6.25) and uses a recursive (IIR) Gaussian, which adds about 1 %
   (factors 2.27 and 6.31). The response is proportional to the IBSI response, so ratio-type statistics are
   nearly unchanged, but absolute values (mean, percentiles, energy, ...) are not IBSI-compliant.
2. **IBSI method.** PySegment computes the LoG itself (exact Gaussian kernel truncated at 4σ, σ in mm,
   response in voxel units, as in the IBSI 2 reference maps) after IBSI resampling, re-segments the ROI on
   the original intensities and computes the features on the response map. Because the response is in
   voxel units, LoG features are comparable only between images resampled to the same voxel size, which
   the IBSI method guarantees.
3. **2B residual (0.0018).** Interior voxels at the checkerboard corners; it comes from the discretisation
   of the truncated kernel and is far below the phase 2 tolerances (3.B passes 15/15).
4. **1.B, PyRadiomics method: maximum** outside tolerance because of PyRadiomics resampling (see
   `validation/ibsi1/`).
