# PySegment – validation of SPECT quantification (counts → activity, count conservation)

Reproduce: `python tools/validate_spect_counts.py` (a few seconds). Code version in `summary.json`.

Digital phantom: 64³ grid of 4.8 mm voxels, a body cylinder (10 counts per voxel) with a 30 mm sphere (200 counts per
voxel), smoothed to 10 mm FWHM like a reconstructed SPECT image; about 1.03 million counts in total.

## Counts → activity

ROI over the whole field of view through `rsw.measure.roi_stats` and `roi_table`, calibration 9.4 cps/MBq, 1800 s,
measurement 24 h after the reference time (Lu-177): total counts and activity agree with the hand calculation
A = counts / (sensitivity × time) × 2^(24 h / 159.53 h) to machine precision (relative difference 2 × 10⁻¹⁶); decay
factor 1.1099 in both.

## Conservation of counts

| Operation (PySegment `resample_volume`) | Change of the total counts |
|---|---|
| Coarser grid, 6.4 mm voxels | +0.08 % |
| Finer grid, 2.4 mm voxels | 0.00 % |
| Affine, scale 0.9 (Jacobian applied) | −0.04 % |
| Affine, scale 1.1 (Jacobian applied) | +0.10 % |
| Rigid, 8° rotation and 7 mm shift | 0.00 % |

Counts images are multiplied by the voxel-volume ratio and by the Jacobian of the transform, so the total stays
within 0.1 % (linear interpolation) as long as the object stays in the field of view.

## Limitation: decay during the acquisition

PySegment corrects the decay between the reference time and the scan, but not the decay during the acquisition,
which lowers the counts by λt / (1 − e^(−λt)): Lu-177, 30 min: 0.11 %; I-131, 30 min: 0.09 %; Tc-99m, 20 min: 1.9 %.
Negligible for Lu-177 dosimetry; for Tc-99m it should be added to the calibration (or the sensitivity measured with
the same acquisition time).
