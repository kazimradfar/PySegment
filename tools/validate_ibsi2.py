"""Validate PySegment's image filters (LoG, wavelet) against IBSI 2 (Whybra et al., Radiology 2024).

Phase 1: response maps of the IBSI 2 digital phantoms compared voxel by voxel with the reference response
maps (criterion |difference| <= 0.001 for every voxel). Phase 2: intensity statistics of the filtered chest
CT phantom (3D configurations 1.B = no filter and 3.B = LoG, sigma 1.5 mm) compared with the reference values.
PySegment offers LoG and the non-rotation-invariant separable wavelets of PyRadiomics; mean, Laws, Gabor,
rotation-invariant wavelets and Simoncelli filters are not offered and are not tested.

Data (public): IBSI 2 digital phantoms, reference response maps and phase 2 reference values as distributed
in the MIRP test suite (https://github.com/oncoray/mirp, test/), and the IBSI chest CT phantom
(https://github.com/theibsi/data_sets).

Usage:
  python tools/validate_ibsi2.py --mirp-test <mirp/test> --data <theibsi/data_sets> --out validation/ibsi2
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from rsw import radiomics as rr
from rsw.io import Geometry

STAT = {"Mean": "stat_mean", "Variance": "stat_var", "Skewness": "stat_skew", "Kurtosis": "stat_kurt",
        "Median": "stat_median", "Minimum": "stat_min", "10Percentile": "stat_p10", "90Percentile": "stat_p90",
        "Maximum": "stat_max", "InterquartileRange": "stat_iqr", "Range": "stat_range",
        "MeanAbsoluteDeviation": "stat_mad", "RobustMeanAbsoluteDeviation": "stat_rmad", "Energy": "stat_energy",
        "RootMeanSquared": "stat_rms"}

PHASE1 = [  # id, phantom, filter, parameters, reference file
    ("2A", "impulse/image/response", "log", dict(sigma=3.0, boundary="constant"), "id_2_impulse_CT_2A_log_s3.0"),
    ("2B", "checkerboard/image/checkerboard", "log", dict(sigma=5.0, boundary="reflect"),
     "id_2_checkerboard_CT_2B_log_s5.0"),
    ("5A1", "impulse/image/response", "wavelet", dict(wavelet="db2", band="LHL"),
     "id_5_impulse_CT_5A1_wavelet_db2_LHL_level_1"),
    ("6A1", "sphere/image/sphere", "wavelet", dict(wavelet="coif1", band="HHL"),
     "id_6_sphere_CT_6A1_wavelet_coif1_HHL_level_1"),
]


def phase1(mirp_test: Path) -> pd.DataFrame:
    import radiomics.imageoperations as io
    rows = []
    for cid, phantom, kind, par, ref in PHASE1:
        img = sitk.ReadImage(str(mirp_test / "data" / "ibsi_2_digital_phantom" / f"{phantom}.nii.gz"))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        reference = sitk.GetArrayFromImage(sitk.ReadImage(
            str(mirp_test / "data" / "ibsi_2_reference_response_maps" / f"{ref}.nii.gz"))).astype(np.float64)
        full = sitk.Cast(img * 0 + 1, sitk.sitkUInt8)
        maps = {}
        if kind == "log":
            maps["pyradiomics"] = sitk.GetArrayFromImage(next(io.getLoGImage(img, full, sigma=[par["sigma"]]))[0])
            maps["ibsi"] = rr._ibsi_log(arr, img.GetSpacing(), par["sigma"], par["boundary"])
        else:   # PySegment uses the PyRadiomics wavelet in both methods
            bands = {k: v for v, k, _ in io.getWaveletImage(img, full, wavelet=par["wavelet"], start_level=0, level=1)}
            maps["pyradiomics"] = maps["ibsi"] = sitk.GetArrayFromImage(bands["wavelet-" + par["band"]])
        for method, m in maps.items():
            d = np.abs(m.astype(np.float64) - reference)
            rows.append({"configuration": cid, "filter": kind, "parameters": json.dumps(par), "method": method,
                         "max_abs_diff": float(d.max()), "reference_max_abs": float(np.abs(reference).max()),
                         "voxels_within_0.001_pct": float(np.mean(d <= 1e-3) * 100),
                         "all_within_0.001": bool(d.max() <= 1e-3),
                         "scale_vs_reference": float(np.linalg.lstsq(reference.reshape(-1, 1), m.reshape(-1).astype(np.float64),
                                                                     rcond=None)[0][0])})
    return pd.DataFrame(rows)


def phase2_reference(test_file: Path, config: str) -> pd.DataFrame:
    """Reference values and tolerances of one configuration from MIRP's IBSI 2 phase 2 test."""
    text = test_file.read_text(encoding="utf-8")
    block = re.split(r"# Configuration ", text)
    part = next(b for b in block if b.startswith(config + " "))
    vals = re.findall(r'within_tolerance\(([-\d.e+]+), ([-\d.e+]+), data\["(\w+)"\]\)', part)
    return pd.DataFrame([(float(r), float(t), tag) for r, t, tag in vals], columns=["reference", "tolerance", "tag"])


def phase2(data: Path, mirp_test: Path, config: str, method: str) -> pd.DataFrame:
    base = data / "ibsi_1_ct_radiomics_phantom" / "nifti"
    img = sitk.ReadImage(str(base / "image" / "phantom.nii.gz"))
    msk = sitk.ReadImage(str(base / "mask" / "mask.nii.gz"))
    use_log = config == "3.B"
    s = rr.Settings(discretisation="count", bin_value=32, resample_mm=1.0, interpolator="sitkBSpline",
                    resampling=method, resegment_low=-1000.0, resegment_high=400.0, classes=["firstorder"],
                    image_types=["LoG"] if use_log else ["Original"], log_sigma_mm=[1.5], log_boundary="reflect",
                    min_texture_voxels=0, per_lesion=False)
    if use_log and method == "pyradiomics":
        s.image_types = ["Original", "LoG"]
    row = rr.extract(sitk.GetArrayFromImage(img).astype(np.float32), Geometry.of(img), sitk.GetArrayFromImage(msk),
                     "ibsi_chest_ct", "gtv", s).iloc[0]
    prefix = "log-sigma-1-5-mm-3D_firstorder_" if use_log else "original_firstorder_"
    values = {STAT[k[len(prefix):]]: float(v) for k, v in row.items()
              if str(k).startswith(prefix) and k[len(prefix):] in STAT}
    values["stat_kurt"] = values["stat_kurt"] - 3.0          # PyRadiomics kurtosis -> IBSI excess kurtosis
    ref = phase2_reference(mirp_test / "ibsi_2_feature_test.py", config)
    ref = ref[ref.tag.isin(values)].copy()
    ref["pysegment"] = ref.tag.map(values)
    ref["abs_diff"] = (ref.pysegment - ref.reference).abs()
    mag = np.floor(np.log10(ref.reference.abs().where(ref.reference != 0, 1.0)))
    ref["within_tolerance"] = ref.abs_diff <= ref.tolerance + 0.5 * 10.0 ** (mag - 2) + 1e-12
    ref.insert(0, "configuration", config)
    ref.insert(1, "method", method)
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirp-test", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("validation/ibsi2"))
    a = ap.parse_args()
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    a.out.mkdir(parents=True, exist_ok=True)
    p1 = phase1(a.mirp_test)
    p1.to_csv(a.out / "phase1_response_maps.csv", index=False)
    print(p1[["configuration", "method", "max_abs_diff", "voxels_within_0.001_pct", "scale_vs_reference"]].to_string(index=False))
    p2 = pd.concat([phase2(a.data, a.mirp_test, c, m) for c in ("1.B", "3.B") for m in ("pyradiomics", "ibsi")])
    p2.to_csv(a.out / "phase2_features.csv", index=False)
    summary = {f"{c} {m}": f"{int(g.within_tolerance.sum())}/{len(g)}" for (c, m), g in p2.groupby(["configuration", "method"])}
    import radiomics
    (a.out / "summary.json").write_text(json.dumps({"pyradiomics_version": radiomics.__version__,
                                                    "phase1_all_within_0.001": {f"{r.configuration} {r.method}": r.all_within_0_001
                                                                                for r in p1.rename(columns={"all_within_0.001": "all_within_0_001"}).itertuples()},
                                                    "phase2": summary,
                                                    "phase2_failed": {f"{c} {m}": g.loc[~g.within_tolerance, "tag"].tolist()
                                                                      for (c, m), g in p2.groupby(["configuration", "method"])}},
                                                   indent=1), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
