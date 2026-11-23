# PySegment – Segmentation & Radiomics Workstation

Desktop program (Windows; Python/Qt) for reader-corrected segmentation and reproducible radiomics of PET,
CT and MRI. It works offline; images, masks and results never leave the computer.
Research use only – not a medical device.

## What it does

| Area | Functions |
|---|---|
| Images | NIfTI, NRRD, MHA and DICOM series (PET in Bq/mL → body-weight SUV). Several images per case on the grid of the reference image (PET if present). **File → Export images as NIfTI**: DICOM → NIfTI (PET in SUV, CT in HU), each on its own grid or all on the PET grid. |
| Viewer | Axial, coronal, sagittal and PET MIP (click on the MIP jumps to the hottest voxel); PET over CT fusion, colour maps, window, linked cursor, value under the mouse. Radiological orientation from the direction cosines, without resampling. |
| Segmentation | Mask layers; brush paint / erase (2D disk or 3D sphere, optionally only where the value ≥ a threshold); PET region growing from one click (% of the local maximum, e.g. 41 %, or an absolute value; a guard stops floods, and a percentage that falls below 1.5 x the local background is refused because the region would leak into normal tissue; for low-contrast lesions use an absolute threshold half-way between background and lesion, see `validation/qiba_dro/`); remove small islands; threshold (whole image >= value; keep >= value inside the mask; per-lesion % of own maximum, e.g. 41 %); smooth outline (Gaussian, sigma in mm); expand / shrink by real millimetres (also for non-cubic voxels); lower and upper value limit for the brush; Alt + wheel = brush size; true-size brush outline; undo / redo per stroke; lesion list with jump. |
| Readers | Projects with several readers. A reader continues his own mask or starts from the automatic mask, never from another reader's mask; optional blinding hides the other readers' masks. Masks are saved as uint8 with exactly the reference geometry; the previous file goes to `backup/`; every save is logged in `mask_history.jsonl`. |
| Radiomics | PyRadiomics (IBSI) for the whole mask and each connected lesion; fixed bin width / count, isotropic resampling, re-segmentation, texture only for regions ≥ N voxels, LoG and wavelet optional. Resampling method: PyRadiomics (default, as before) or IBSI (grid centres aligned, mask interpolated), with an upper re-segmentation bound and IBSI outlier filtering (mean ± 3 SD); with the IBSI method the LoG filter is computed as IBSI 2 defines it, the fixed bin size is numbered from the re-segmentation lower bound and the PCA axes use N − 1; validated against the IBSI 1 and IBSI 2 reference values (`validation/ibsi1/`, `validation/ibsi2/`). Settings are saved next to every CSV. |
| Agreement | Dice, Jaccard, HD95 and volume difference between two masks or two readers over a project; ICC(2,1) with 95 % CI for every feature (Koo & Li categories) and the share kept at a threshold. |
| Registration | **Registration → Register images…**: inter-modality (CT↔PET, CT↔SPECT, PET↔SPECT; Mattes mutual information) and intra-modality (CT↔CT, PET↔PET, SPECT↔SPECT; mean squares or correlation). Rigid, affine or B-spline, multi-resolution, landmark or geometry start (SimpleITK). Fixed image = the case reference. Hybrid option: estimate on the CTs of two studies, apply to the PET/SPECT of the moving study. QA: checkerboard, metric, Jacobian folding check, total-activity change; Accept / Reject; transforms saved as `.tfm` and re-applied with *Load transform file*. Native images are never modified; masks are resampled with nearest neighbour. |
| SPECT | Multi-frame DICOM `NM` series; values stay in counts (or the unit of the file); **no SUV** is computed without a validated quantitative calibration. Displayed like PET. |
| Licence | Open source, MIT licence (`rsw/LICENSE`). Research use; not a medical device. |

## Projects (folder layouts)

* **Generic**: `<folder>/<case>/` with image files or DICOM folders; masks are files whose name contains
  `mask`, `seg`, `label`, `roi` or `lesion`, or files in `<case>/masks/`. A mask whose name contains `auto`
  is the starting point for readers. Reader masks: `<case>/masks/<structure>_<reader>.nii.gz`.
* **PROSPeR / PSMA pipeline**: `data/nifti/<case>/PET_SUV.nii.gz, CT.nii.gz` and `data/masks/<case>/`;
  reader masks `tumor_mask_<reader>.nii.gz`.
* A single folder with images is one case.

Project settings (readers, structure name) are kept in `rsw_project.json` in the folder.

## Mouse and keys

Wheel = slice · Ctrl+wheel or right drag = zoom · middle drag = pan · Shift+left drag = window ·
N / P / E / G = navigate / paint / erase / grow · [ ] = brush size · Ctrl+Z / Ctrl+Y = undo / redo ·
Ctrl+S = save the active mask · Ctrl+O = open a project folder.

## Run, test, build

```
pip install -r requirements-desktop.txt
pip install --no-deps "pyradiomics @ git+https://github.com/AIM-Harvard/pyradiomics.git"
python -m rsw [project folder]
python -m pytest tests/test_rsw_core.py tests/test_rsw_project.py tests/test_rsw_gui.py -q
pyinstaller rsw.spec                                   # dist/PySegment/PySegment.exe; PySegment.exe --self-test result.json
```

`.github/workflows/build-rsw.yml` builds and tests the program on Windows (tests, self-test of the built
and of the installed program, uninstall) on every push to `main` that changes the program, and attaches the setup wizard to every release published on GitHub (for example `v0.1.2`).

## Registration: status

Automatic tests use synthetic phantoms with a known transform only (rigid/affine/B-spline recovery for every modality pair, mask propagation, activity preservation, Jacobian folding, dialog behaviour). It has **not** been validated on clinical images. Validation on real images (51 PET/CT/MR studies, known transforms, hybrid PET/CT, landmark TRE): `tools/validate_registration.py`, results in `validation/registration/`; full plan in `validation/VALIDATION_PLAN.md`. Start: *Automatic* (scanner coordinates when the images overlap, else image centres). The result is never worse than the start (otherwise the start is kept, with a warning); corrections above 20 mm give a warning. Affine (and the affine stage of B-spline) starts from a rigid registration, so it is never worse than rigid. Default 500 iterations per level (the optimiser stops earlier when it converges; small or cropped fields of view need 300–400); reaching the limit while the transform still moves (image corners > 0.05 mm over the last 20 iterations) gives a *Not converged* warning.

## Limits of version 0.1

* Display orientation uses the main axis of each direction cosine (very oblique scans are shown in array order).
* No DICOM-RT import/export yet; nnU-Net / TotalSegmentator are run outside the program (import their masks).
* Tested automatically with synthetic images only.
- One transform per image; masks drawn before a registration is accepted are not moved. Transforms are not stored in the project file.
- SPECT reading: one multi-frame NM file with complete geometry; no SUV.

## Manual segmentation toolkit
Paint / erase (true-size outline, lower and upper value limit, 3D sphere), **polygon** (click corners, double-click or Enter closes; Esc cancels; option to remove), region grow, threshold (whole image / inside mask / per-lesion % of maximum), smooth, expand/shrink in mm, remove islands; **Mask operations**: fill holes, keep largest part, fill between slices (shape-based interpolation), clear, duplicate, union / intersect / subtract two layers; undo/redo for each. Display: choose background and overlay image (e.g. the registered SPECT), overlay opacity, mask fill (0 = outline only). Spin boxes, combo boxes and sliders react to the wheel only after a click into them, so scrolling the panel never changes a value. Registration: fixed (usually CT) and moving (usually PET/SPECT) image are free choices; after a run the moved image is shown in colour on the fixed one, the cursor jumps to its content and a Before / after button toggles the view.

## Measurements and DICOM tools
- **Profile tool (O):** two clicks give a line profile with FWHM; CSV export.
- **File ▸ DICOM header and SUV check** (technical tags only, tells whether body-weight SUV can be computed) and **File ▸ Anonymise DICOM folder** (names, IDs, institution, physicians removed; UIDs remapped consistently; weight, dose and times kept so SUV still works; optional common date shift). Burned-in pixel text is not removed.
- Tested on synthetic data only; no clinical validation yet.

## Landmark registration and TRE
Registration dialog, section *Landmarks*: place the same anatomical point on the fixed and the moving image (View fixed / Fixed point, View moving / Moving point), at least 3 well-spread pairs. *Landmarks only* gives a rigid fit with the fiducial registration error; *Start: Landmark pairs* initialises the image optimisation. Pairs ticked *validation* are excluded from the fit and used only to report the target registration error (mean, max) after any registration. *Export pairs* writes a CSV. Changing the fixed image clears the pairs (their coordinates belong to its frame).

## ROIs (ROI tab)
ROIs are measurement regions, separate from segmentation masks. Draw them with the ROI tool (R): freehand (drag), polygon (clicks, double-click closes), ellipse (drag), circle 2D or sphere 3D (click, radius in mm), or copy the active mask (*Mask → ROI*). Each ROI is outlined in its own colour and listed in the ROI tab. *Save…* / *Save all…* write `roi_<name>.nii.gz` on the reference grid plus `roi_<name>.roi.json` (name, shape, plane, slice, colour); *Load…* reads them back. With the Navigate tool, a click inside an ROI (or a double-click in the list) opens **ROI details**: geometry (volume, voxels, extent, centre, area for 2D) and one row per open image (CT HU, PET SUV, SPECT counts) with mean, SD, min, max, median, peak 1 mL, total counts, activity and concentration with a counts calibration or SUV with dose and weight, plus a histogram. *Table* lists every ROI on the functional image shown, with the per-radiopharmaceutical calibration (sensitivity cps/MBq, acquisition time, decay correction). PERCIST: select a liver ROI, *Selected ROI = reference*, *→ Threshold*. Gaussian filter: Tools menu.

## Simple dosimetry (Tools ▸ Dosimetry)
Activities (MBq) of a target at several times give the time-integrated activity (trapezoids with an effective or physical tail, or one fitted exponential; linear rise from zero optional), then the absorbed dose by local deposition (decays x mean electron energy / mass; default Lu-177 0.148 MeV, verify against your reference) or a user S-value. Gives residence time and Gy/GBq when the administered activity is entered. Photons and cross-fire are not included. The method is explained in the window itself. Research use only, not clinically validated.

Dosimetry is general: alpha, electron (beta/conversion/Auger) and photon energy per decay for Lu-177, Y-90, I-131, Ac-225, Ra-223, Tc-99m, F-18, Ga-68 or custom (indicative table values, verify against your reference). Alpha chains use a "daughters retained" fraction, photons an absorbed fraction you set, and an optional alpha RBE gives a second Gy(RBE) column. Calibration profiles (sensitivity per radiopharmaceutical and isotope) are entered by hand in the Measurements window and kept in `~/.pysegment/calibrations.json`.

MIRD organ mode: D(target) = sum of A~(source) x S(target <- source) from an S-value CSV you export from OLINDA/EXM, IDAC or MIRDcalc (their data are not bundled). Unmatched sources are reported, not ignored. Research use only: not cleared for clinical use and not equivalent to OLINDA/EXM until validated against it with identical inputs.

OLINDA-like functions: mono/bi-exponential and trapezoid kinetics, remainder of body (whole-body minus organs), MIRD organ doses from your S-value matrix, effective dose with ICRP 103/60 weights (simplified), organ-mass scaling of self-dose. Not implemented: phantoms and S-value data, bladder and GI-tract models, tumour sphere model, uptake-washout kinetic model. Validated against analytic kinetics, ICRP 107 half-lives and the EANM 2022 / IDAC-Dose 2.1 Lu-177 S-values (`validation/dosimetry/`); not yet against OLINDA/EXM with identical patient inputs. Fit safeguards: no kinetic component or extrapolated tail decays slower than physical decay; red marrow with local deposition gives a warning.


Paint and Erase: a single click paints one brush spot, dragging paints a stroke. The Segment tab holds the mask tools only; ROI and Profile tools are in the ROI tab.

**Labels on the image.** Tab *Labels*, tool *Label* (T): click on any view and type a text; it is drawn with an arrow pointing at the point. Drag the text or the arrow tip to move them; click a label to select it, double-click to edit; the tab lists all labels (click one to go to its slice) and sets arrow, text size and colour. *Show mask and ROI names on the image* writes the names next to the regions. Both are exported by File → *Save view as image*. Labels are kept while the case is open.

The Windows installer includes the elastix engine (itk-elastix).

**DICOM.** File → *Open DICOM folder…*: every image series under the folder is found, also when PET and CT were exported into the same folder; with several series a list asks which to open (PET/SPECT and CT ticked). PET in Bq/mL is converted to SUV. A project folder with one sub-folder per patient (PET/ and CT/ inside) opens with *Open project folder*.

**Figures.** File → *Save window as image…* saves the whole program window (panels, menus, views), and the F12 key saves whichever window is in front, also a dialog (registration, dosimetry, ROI details), File → *Save view as image…* only the image views; both at three times the screen resolution, marked 300 dpi, with labels and names.
