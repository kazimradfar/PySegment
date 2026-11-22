"""Validate PySegment's radiomics module against the IBSI 1 reference values.

Features are computed through PySegment's own code path (rsw.radiomics.extract with rsw Settings), then
compared with the IBSI consensus reference values and tolerances (Zwanenburg et al., Radiology 2020).

Data (public, not patient data of this study):
  * IBSI 1 digital phantom and chest CT phantom: https://github.com/theibsi/data_sets
  * reference value tables: test/data/ibsi_1_reference_values in https://github.com/oncoray/mirp

Usage:
  python tools/validate_ibsi1.py --data <data_sets folder> --ref <ibsi_1_reference_values folder> --out validation/ibsi1
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from rsw import radiomics as rr
from rsw.io import Geometry

# PyRadiomics feature name -> IBSI tag stem. GLCM and GLRLM in PyRadiomics are computed per direction and
# averaged (IBSI "3D, averaged"); GLSZM, GLDM (= NGLDM with alpha 0) and NGTDM are 3D.
MAP = {
    "shape": {"MeshVolume": "morph_volume", "VoxelVolume": "morph_vol_approx", "SurfaceArea": "morph_area_mesh",
              "SurfaceVolumeRatio": "morph_av", "Sphericity": "morph_sphericity", "Maximum3DDiameter": "morph_diam",
              "MajorAxisLength": "morph_pca_maj_axis", "MinorAxisLength": "morph_pca_min_axis",
              "LeastAxisLength": "morph_pca_least_axis", "Elongation": "morph_pca_elongation",
              "Flatness": "morph_pca_flatness"},
    "firstorder": {"Mean": "stat_mean", "Variance": "stat_var", "Skewness": "stat_skew", "Kurtosis": "stat_kurt",
                   "Median": "stat_median", "Minimum": "stat_min", "10Percentile": "stat_p10",
                   "90Percentile": "stat_p90", "Maximum": "stat_max", "InterquartileRange": "stat_iqr",
                   "Range": "stat_range", "MeanAbsoluteDeviation": "stat_mad",
                   "RobustMeanAbsoluteDeviation": "stat_rmad", "Energy": "stat_energy",
                   "RootMeanSquared": "stat_rms", "Entropy": "ih_entropy", "Uniformity": "ih_uniformity"},
    "glcm": {"MaximumProbability": "cm_joint_max", "JointAverage": "cm_joint_avg", "SumSquares": "cm_joint_var",
             "JointEntropy": "cm_joint_entr", "DifferenceAverage": "cm_diff_avg", "DifferenceVariance": "cm_diff_var",
             "DifferenceEntropy": "cm_diff_entr", "SumAverage": "cm_sum_avg", "SumEntropy": "cm_sum_entr",
             "JointEnergy": "cm_energy", "Contrast": "cm_contrast", "Id": "cm_inv_diff", "Idn": "cm_inv_diff_norm",
             "Idm": "cm_inv_diff_mom", "Idmn": "cm_inv_diff_mom_norm", "InverseVariance": "cm_inv_var",
             "Correlation": "cm_corr", "Autocorrelation": "cm_auto_corr", "ClusterTendency": "cm_clust_tend",
             "ClusterShade": "cm_clust_shade", "ClusterProminence": "cm_clust_prom", "Imc1": "cm_info_corr1",
             "Imc2": "cm_info_corr2"},
    "glrlm": {"ShortRunEmphasis": "rlm_sre", "LongRunEmphasis": "rlm_lre", "LowGrayLevelRunEmphasis": "rlm_lgre",
              "HighGrayLevelRunEmphasis": "rlm_hgre", "ShortRunLowGrayLevelEmphasis": "rlm_srlge",
              "ShortRunHighGrayLevelEmphasis": "rlm_srhge", "LongRunLowGrayLevelEmphasis": "rlm_lrlge",
              "LongRunHighGrayLevelEmphasis": "rlm_lrhge", "GrayLevelNonUniformity": "rlm_glnu",
              "GrayLevelNonUniformityNormalized": "rlm_glnu_norm", "RunLengthNonUniformity": "rlm_rlnu",
              "RunLengthNonUniformityNormalized": "rlm_rlnu_norm", "RunPercentage": "rlm_r_perc",
              "GrayLevelVariance": "rlm_gl_var", "RunVariance": "rlm_rl_var", "RunEntropy": "rlm_rl_entr"},
    "glszm": {"SmallAreaEmphasis": "szm_sze", "LargeAreaEmphasis": "szm_lze", "LowGrayLevelZoneEmphasis": "szm_lgze",
              "HighGrayLevelZoneEmphasis": "szm_hgze", "SmallAreaLowGrayLevelEmphasis": "szm_szlge",
              "SmallAreaHighGrayLevelEmphasis": "szm_szhge", "LargeAreaLowGrayLevelEmphasis": "szm_lzlge",
              "LargeAreaHighGrayLevelEmphasis": "szm_lzhge", "GrayLevelNonUniformity": "szm_glnu",
              "GrayLevelNonUniformityNormalized": "szm_glnu_norm", "SizeZoneNonUniformity": "szm_zsnu",
              "SizeZoneNonUniformityNormalized": "szm_zsnu_norm", "ZonePercentage": "szm_z_perc",
              "GrayLevelVariance": "szm_gl_var", "ZoneVariance": "szm_zs_var", "ZoneEntropy": "szm_zs_entr"},
    "gldm": {"SmallDependenceEmphasis": "ngl_lde", "LargeDependenceEmphasis": "ngl_hde",
             "LowGrayLevelEmphasis": "ngl_lgce", "HighGrayLevelEmphasis": "ngl_hgce",
             "SmallDependenceLowGrayLevelEmphasis": "ngl_ldlge", "SmallDependenceHighGrayLevelEmphasis": "ngl_ldhge",
             "LargeDependenceLowGrayLevelEmphasis": "ngl_hdlge", "LargeDependenceHighGrayLevelEmphasis": "ngl_hdhge",
             "GrayLevelNonUniformity": "ngl_glnu", "DependenceNonUniformity": "ngl_dcnu",
             "DependenceNonUniformityNormalized": "ngl_dcnu_norm", "GrayLevelVariance": "ngl_gl_var",
             "DependenceVariance": "ngl_dc_var", "DependenceEntropy": "ngl_dc_entr"},
    "ngtdm": {"Coarseness": "ngt_coarseness", "Contrast": "ngt_contrast", "Busyness": "ngt_busyness",
              "Complexity": "ngt_complexity", "Strength": "ngt_strength"},
}
SUFFIX = {"glcm": "_3D_avg", "glrlm": "_3D_avg", "glszm": "_3D", "gldm": "_3D", "ngtdm": "_3D"}
# Convention difference: PyRadiomics reports kurtosis, IBSI reports excess kurtosis (kurtosis - 3).
CONVERT = {"stat_kurt": lambda v: v - 3.0}

# IBSI 1 chest CT configurations C-E (3D). A and B are 2D and are not offered by PySegment.
CONFIGS = {
    "C": dict(interpolator="sitkLinear", discretisation="width", bin_value=25.0, range=True, outlier=False),
    "D": dict(interpolator="sitkLinear", discretisation="count", bin_value=32, range=False, outlier=True),
    "E": dict(interpolator="sitkBSpline", discretisation="count", bin_value=32, range=True, outlier=True),
}


def to_ibsi(row: pd.Series) -> dict:
    out = {}
    for k, v in row.items():
        parts = str(k).split("_")
        if len(parts) != 3 or parts[0] != "original":
            continue
        cls, name = parts[1], parts[2]
        tag = MAP.get(cls, {}).get(name)
        if tag:
            tag += SUFFIX.get(cls, "")
            out[tag] = CONVERT.get(tag, lambda x: x)(float(v))
    return out


def compare(values: dict, ref_csv: Path) -> pd.DataFrame:
    ref = pd.read_csv(ref_csv, sep=";")
    ref.columns = ["reference", "tolerance", "tag"]
    ref = ref[ref.tag.isin(values)].copy()
    ref["pysegment"] = ref.tag.map(values)
    ref["abs_diff"] = (ref.pysegment - ref.reference).abs()
    # Reference values are published to 3 significant digits: allow half a unit of the last digit.
    mag = np.floor(np.log10(ref.reference.abs().where(ref.reference != 0, 1.0)))
    ref["within_tolerance"] = ref.abs_diff <= ref.tolerance + 0.5 * 10.0 ** (mag - 2) + 1e-12
    ref["family"] = ref.tag.str.split("_").str[0]
    return ref[["family", "tag", "reference", "tolerance", "pysegment", "abs_diff", "within_tolerance"]]


def digital_phantom(data: Path, resampling: str = "pyradiomics") -> dict:
    import pydicom
    base = data / "ibsi_1_digital_phantom" / "dicom_deprecated"
    img = pydicom.dcmread(base / "image" / "phantom.deprecated", force=True).pixel_array.astype(np.float32)
    msk = pydicom.dcmread(base / "mask" / "mask.deprecated", force=True).pixel_array.astype(np.uint8)
    geo = Geometry(size=img.shape[::-1], spacing=(2.0, 2.0, 2.0), origin=(0, 0, 0),
                   direction=(1, 0, 0, 0, 1, 0, 0, 0, 1))
    s = rr.Settings(discretisation="width", bin_value=1.0, resample_mm=None, resegment_low=None,
                    resampling=resampling, min_texture_voxels=0, per_lesion=False)
    df = rr.extract(img, geo, msk, "ibsi_digital_phantom", "roi", s)
    return to_ibsi(df.iloc[0])


def chest_ct(data: Path, cfg: dict, resampling: str) -> dict:
    """Chest CT phantom through PySegment, with its default (PyRadiomics) or its IBSI resampling option."""
    base = data / "ibsi_1_ct_radiomics_phantom" / "nifti"
    img = sitk.ReadImage(str(base / "image" / "phantom.nii.gz"))
    msk = sitk.ReadImage(str(base / "mask" / "mask.nii.gz"))
    ibsi = resampling == "ibsi"
    s = rr.Settings(discretisation=cfg["discretisation"], bin_value=cfg["bin_value"], resample_mm=2.0,
                    interpolator=cfg["interpolator"], resampling=resampling, round_intensities=ibsi,
                    resegment_low=-1000.0 if cfg["range"] else None,
                    resegment_high=400.0 if cfg["range"] else None,
                    # PyRadiomics cannot combine a value range with outlier filtering: only the range is used there
                    outlier_sigma=3.0 if cfg["outlier"] and (ibsi or not cfg["range"]) else None,
                    min_texture_voxels=0, per_lesion=False)
    df = rr.extract(sitk.GetArrayFromImage(img).astype(np.float32), Geometry.of(img),
                    sitk.GetArrayFromImage(msk), "ibsi_chest_ct", "gtv", s)
    return to_ibsi(df.iloc[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("validation/ibsi1"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    import radiomics
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True).stdout.strip()
    summary = {"pyradiomics_version": radiomics.__version__, "pysegment_commit": commit}
    runs = {f"digital_phantom_{mode}_method": (lambda mode=mode: digital_phantom(a.data, mode), "ibsi_1_dig_phantom.csv")
            for mode in ("pyradiomics", "ibsi")}
    for c, cfg in CONFIGS.items():
        for mode in ("pyradiomics", "ibsi"):
            runs[f"chest_ct_config_{c}_{mode}_resampling"] = (lambda cfg=cfg, mode=mode: chest_ct(a.data, cfg, mode),
                                                              f"ibsi_1_chest_config_{c.lower()}.csv")
    for name, (fn, ref) in runs.items():
        res = compare(fn(), a.ref / ref)
        res.to_csv(a.out / f"{name}.csv", index=False)
        fam = res.groupby("family").within_tolerance.agg(["sum", "count"]).astype(int)
        summary[name] = {"passed": int(res.within_tolerance.sum()), "compared": len(res),
                         "by_family": {f: f"{r['sum']}/{r['count']}" for f, r in fam.iterrows()},
                         "failed": res.loc[~res.within_tolerance, "tag"].tolist()}
        print(f"{name}: {summary[name]['passed']}/{summary[name]['compared']}  {summary[name]['by_family']}")
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
