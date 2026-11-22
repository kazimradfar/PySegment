"""Project-wide jobs: radiomics for every case, mask agreement between two readers, ICC per feature.
Each job takes a `progress(i, n, text)` callback so the interface can show progress and stay responsive."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rsw import io, metrics
from rsw.project import Project
from rsw.radiomics import Settings, extract


def _noop(*_):
    pass


def reference_volume(project: Project, case) -> io.Volume:
    return io.load_volume(project.path(case.images[case.reference]), name=case.reference)


def mask_for(project: Project, case, which: str, ref: io.Geometry):
    """which = a reader name (that reader's saved mask) or the name of any mask of the case."""
    rel = project.reader_mask_rel(case, which) if which in project.readers else case.masks.get(which)
    if not rel or not project.path(rel).exists():
        return None
    return io.load_mask(project.path(rel), ref)[0]


def radiomics_project(project: Project, which: str, settings: Settings, progress=_noop) -> pd.DataFrame:
    frames = []
    for i, case in enumerate(project.cases):
        progress(i, len(project.cases), case.id)
        vol = reference_volume(project, case)
        m = mask_for(project, case, which, vol.geometry)
        if m is not None and m.any():
            frames.append(extract(vol.array, vol.geometry, m, case.id, which, settings))
    progress(len(project.cases), len(project.cases), "done")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def agreement_project(project: Project, a: str, b: str, progress=_noop) -> pd.DataFrame:
    rows = []
    for i, case in enumerate(project.cases):
        progress(i, len(project.cases), case.id)
        vol = reference_volume(project, case)
        ma, mb = mask_for(project, case, a, vol.geometry), mask_for(project, case, b, vol.geometry)
        if ma is None or mb is None:
            continue
        rows.append({"case": case.id, "a": a, "b": b, **metrics.agreement(ma, mb, vol.geometry.spacing)})
    progress(len(project.cases), len(project.cases), "done")
    return pd.DataFrame(rows)


def icc_project(project: Project, a: str, b: str, settings: Settings, threshold: float = 0.75,
                progress=_noop) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Features of the whole mask for readers a and b on the same cases, and the ICC of every feature.
    Returns (icc table, features a, features b)."""
    s = Settings(**{**settings.__dict__, "per_lesion": False})
    fa = radiomics_project(project, a, s, lambda i, n, t: progress(i, 2 * n, f"{a}: {t}"))
    fb = radiomics_project(project, b, s, lambda i, n, t: progress(n + i, 2 * n, f"{b}: {t}"))
    if fa.empty or fb.empty:
        return pd.DataFrame(), fa, fb
    drop = ["mask", "lesion", "voxels"]
    tab = metrics.icc_table(fa.drop(columns=drop), fb.drop(columns=drop))
    tab["kept"] = tab["icc"] >= threshold
    return tab.sort_values("icc", ascending=False, na_position="last").reset_index(drop=True), fa, fb


def summary_icc(tab: pd.DataFrame) -> str:
    if tab.empty:
        return "No cases with both readers' masks."
    kept = int(tab["kept"].sum())
    return (f"{len(tab)} features, {kept} with ICC ≥ threshold ({100 * kept / len(tab):.0f}%), "
            f"median ICC {np.nanmedian(tab['icc']):.2f}, n = {int(tab['n'].max())} cases")
