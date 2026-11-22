# PySegment – validation of the agreement metrics (Dice, Jaccard, HD95, ICC)

Reproduce: `pip install pingouin medpy`, then `python tools/validate_agreement.py` (about 15 s). Versions and code
commit in `summary.json`.

## Dice and Jaccard

Exact answers (a cube of 20 voxels and the same cube shifted along one axis):

| Case | Dice (exact) | Jaccard (exact) |
|---|---|---|
| cubes shifted 0 voxels | 1.0000 (1.0000) | 1.0000 (1.0000) |
| cubes shifted 4 voxels | 0.8000 (0.8000) | 0.6667 (0.6667) |
| cubes shifted 8 voxels | 0.6000 (0.6000) | 0.4286 (0.4286) |
| cubes shifted 12 voxels | 0.4000 (0.4000) | 0.2500 (0.2500) |
| cubes shifted 16 voxels | 0.2000 (0.2000) | 0.1111 (0.1111) |
| cubes shifted 20 voxels | 0.0000 (0.0000) | 0.0000 (0.0000) |

200 random pairs of ellipsoids on anisotropic grids (voxels 0.8–5 mm) against SimpleITK
`LabelOverlapMeasuresImageFilter`: largest difference 1.1e-16 (Dice) and
0.0e+00 (Jaccard), i.e. identical.

## HD95 (95th percentile of the pooled surface distances, mm)

Concentric spheres on a 0.5 mm grid, where every surface distance equals the difference of the radii:

| Case | HD95 PySegment | Expected |
|---|---|---|
| concentric spheres r 10 and 12 mm (0.5 mm voxels) | 2.062 mm | 2 mm |
| concentric spheres r 10 and 14 mm (0.5 mm voxels) | 4.062 mm | 4 mm |
| concentric spheres r 15 and 20 mm (0.5 mm voxels) | 5.025 mm | 5 mm |

The excess (0.03–0.06 mm, about a tenth of a voxel) comes from the voxelised surfaces.

Against MedPy (`medpy.metric.binary.hd95`, same definition) on the 200 random pairs: identical (largest difference
0.0 mm) when both define the surface with 26-neighbour connectivity, as
PySegment does. With MedPy's default 6-neighbour surface the values differ by a median of
0.51 mm and at most 3.3 mm
on coarse anisotropic grids. HD95 values from different programs are therefore only comparable when the surface
definition is the same; a publication must state it (PySegment: voxels of the mask removed by a 3 × 3 × 3 erosion).

## ICC(A,1) = ICC(2,1) with the McGraw & Wong confidence interval

Published example (Shrout & Fleiss, Psychol Bull 1979;86:420–428; 6 targets, 4 judges): PySegment
ICC = 0.2898 (95 % CI 0.0188–0.7611); published ICC(2,1) = 0.29; pingouin
0.2898 (0.02–0.76).

500 random data sets (5–200 subjects, 2–4 raters, varying error and rater bias) against pingouin 0.7 ICC(A,1):
largest difference of the ICC 1e-15; of the confidence limits
0.005, which is the rounding of the
pingouin output to two decimals.

Koo & Li (2016) categories at the limits: 0.49 poor, 0.50 moderate, 0.749 moderate, 0.75 good, 0.90 good,
0.901 excellent ("greater than 0.90" excellent).

No error was found in this module.
