"""Projects: a folder of cases, each with images and masks, and the readers who correct the masks.

Two folder layouts are recognised when a folder is opened:

* generic   <folder>/<case>/<image files or DICOM folders>, masks are files whose name contains
            mask / seg / label / roi (or that sit in a masks/ subfolder). Reader masks are saved as
            <case>/masks/<structure>_<reader>.nii.gz.
* PROSPeR   <folder>/data/nifti/<case>/PET_SUV.nii.gz, CT.nii.gz and <folder>/data/masks/<case>/*.nii.gz;
            reader masks are tumor_mask_<reader>.nii.gz (the layout of the PSMA PET pipeline).

The project is remembered in <folder>/rsw_project.json (relative paths, so the folder can be moved).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rsw.io import guess_kind, is_image_file, stem

PROJECT_FILE = "rsw_project.json"
MASK_WORDS = re.compile(r"(mask|seg|label|roi|contour|lesion)", re.I)
DICOM_HINT = re.compile(r"\.(dcm|ima)$", re.I)


@dataclass
class Case:
    id: str
    folder: str                                  # case folder (relative to the project root)
    images: dict = field(default_factory=dict)   # name -> relative path (file or DICOM folder)
    masks: dict = field(default_factory=dict)    # name -> relative path
    reference: str = ""                          # image whose grid the masks use (PET if present)


@dataclass
class Project:
    root: str
    layout: str = "generic"
    readers: list = field(default_factory=lambda: ["reader1", "reader2"])
    structure: str = "lesions"                   # name of the structure the readers correct
    cases: list = field(default_factory=list)

    # ------------------------------------------------------------------ paths
    def path(self, rel: str) -> Path:
        return Path(self.root) / rel

    def case(self, cid: str) -> Case:
        return next(c for c in self.cases if c.id == cid)

    def reader_mask_rel(self, case: Case, reader: str) -> str:
        if self.layout == "prosper":
            return f"data/masks/{case.id}/tumor_mask_{reader}.nii.gz"
        return f"{case.folder}/masks/{self.structure}_{reader}.nii.gz"

    def start_mask(self, case: Case, reader: str) -> tuple[str, str | None]:
        """Where a reader's correction starts: the reader's own saved mask, else the automatic mask
        (a mask whose name contains 'auto'), else empty. Never another reader's mask (blinding)."""
        own = self.reader_mask_rel(case, reader)
        if self.path(own).exists():
            return "continue", own
        auto = next((p for n, p in case.masks.items() if "auto" in n.lower()), None)
        if auto:
            return "auto", auto
        return "empty", None

    def visible_masks(self, case: Case, reader: str = "", blind: bool = False) -> dict:
        """Masks the current reader may see: with blinding, masks of the other readers are hidden."""
        if not (blind and reader):
            return dict(case.masks)
        others = [r for r in self.readers if r != reader]
        return {n: p for n, p in case.masks.items() if not any(n.lower().endswith(o.lower()) for o in others)}

    # ------------------------------------------------------------------ persistence
    def save(self) -> Path:
        f = Path(self.root) / PROJECT_FILE
        d = asdict(self)
        d.pop("root")
        f.write_text(json.dumps(d, indent=1), encoding="utf-8")
        return f

    @classmethod
    def load(cls, root: str | Path) -> "Project":
        root = Path(root)
        d = json.loads((root / PROJECT_FILE).read_text(encoding="utf-8"))
        d["cases"] = [Case(**c) for c in d.get("cases", [])]
        return cls(root=str(root), **d)

    def rescan(self) -> "Project":
        """Find new cases / files on disk, keep readers and structure."""
        fresh = scan(self.root)
        fresh.readers, fresh.structure = self.readers, self.structure
        return fresh


# ---------------------------------------------------------------------- discovery

def _is_dicom_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    files = [f for f in p.iterdir() if f.is_file()]
    return bool(files) and (any(DICOM_HINT.search(f.name) for f in files)
                            or all("." not in f.name for f in files[:20]))


def _ref_name(images: dict) -> str:
    pets = [n for n in images if guess_kind(n) == "PET"]
    return pets[0] if pets else (sorted(images)[0] if images else "")


def _scan_generic_case(root: Path, folder: Path) -> Case | None:
    images, masks = {}, {}
    for p in sorted(folder.iterdir()):
        rel = p.relative_to(root).as_posix()
        if p.name in ("backup",) or p.name.startswith("."):
            continue
        if p.is_dir() and p.name.lower() in ("masks", "mask", "segmentations", "labels"):
            for m in sorted(p.iterdir()):
                if is_image_file(m):
                    masks[stem(m)] = m.relative_to(root).as_posix()
        elif is_image_file(p):
            (masks if MASK_WORDS.search(stem(p)) else images)[stem(p)] = rel
        elif _is_dicom_dir(p):
            images[p.name] = rel
    if not images:
        return None
    return Case(folder.name, folder.relative_to(root).as_posix(), images, masks, _ref_name(images))


def scan(root: str | Path) -> Project:
    root = Path(root).resolve()
    nifti = root / "data" / "nifti"
    if nifti.is_dir():                                     # PROSPeR / PSMA pipeline layout
        cases = []
        for d in sorted(p for p in nifti.iterdir() if p.is_dir()):
            images = {stem(f): f.relative_to(root).as_posix() for f in sorted(d.iterdir()) if is_image_file(f)}
            if not images:
                continue
            mdir = root / "data" / "masks" / d.name
            masks = {stem(f): f.relative_to(root).as_posix() for f in sorted(mdir.glob("*.nii*"))} if mdir.is_dir() else {}
            ref = "PET_SUV" if "PET_SUV" in images else _ref_name(images)
            cases.append(Case(d.name, f"data/nifti/{d.name}", images, masks, ref))
        return Project(str(root), "prosper", cases=cases)
    cases = [c for c in (_scan_generic_case(root, d) for d in sorted(root.iterdir()) if d.is_dir()
                         and d.name not in ("backup", "results") and not d.name.startswith(".")) if c]
    if not cases:                                          # images directly in the folder: one case
        single = _scan_generic_case(root, root)
        cases = [single] if single else []
    return Project(str(root), "generic", cases=cases)


def open_project(root: str | Path) -> Project:
    """Open a folder: the saved project if there is one (with new files added), otherwise a new scan."""
    root = Path(root)
    if (root / PROJECT_FILE).exists():
        return Project.load(root).rescan()
    return scan(root)
