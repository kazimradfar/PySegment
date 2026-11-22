# PyInstaller spec for the desktop workstation (built on Windows by .github/workflows/build-rsw.yml;
# locally: pyinstaller rsw.spec). Output: dist/PySegment/PySegment.exe (window) with its libraries in the same folder
# (Qt / PySide6 stay separate, replaceable files, as the LGPL requires).
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata


def _not_tests(name: str) -> bool:
    return not any(p in ("tests", "testing", "test", "conftest") for p in name.split(".")[1:])


datas, binaries, hiddenimports = [], [], []
for pkg in ("radiomics", "SimpleITK", "pydicom", "pywt", "pykwalify", "pyqtgraph"):
    datas += collect_data_files(pkg)
    binaries += collect_dynamic_libs(pkg)
    hiddenimports += collect_submodules(pkg, filter=_not_tests, on_error="warn once")
# itk-elastix (registration engine, lung profile): ITK loads its modules lazily from itk/Configuration/*.py and the
# compiled libraries next to them, so the whole package is copied with its .py files
from PyInstaller.utils.hooks import collect_all
_d, _b, _h = collect_all("itk", include_py_files=True)
datas, binaries, hiddenimports = datas + _d, binaries + _b, hiddenimports + _h
# PyRadiomics loads radiomics/schemas/schemaFuncs.py as a file (pykwalify)
datas += collect_data_files("radiomics", include_py_files=True, subdir="schemas")
for pkg in ("pyradiomics", "numpy", "pandas", "scipy", "SimpleITK", "pydicom", "PySide6", "pyqtgraph"):
    try:
        datas += copy_metadata(pkg)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: metadata {pkg}: {exc}")

datas += [("rsw/resources", "rsw/resources"), ("rsw/edition.json", "rsw"), ("rsw/LICENSE", "."),
          ("rsw/THIRD_PARTY_NOTICES.txt", ".")]
hiddenimports += collect_submodules("rsw") + collect_submodules("scipy", filter=_not_tests, on_error="warn once") \
    + ["yaml", "pykwalify"]

excludes = ["IPython", "ipykernel", "jupyter", "notebook", "torch", "tensorflow", "tkinter", "pytest", "PyQt5",
            "PyQt6", "matplotlib", "streamlit", "sklearn", "sksurv", "xgboost", "shap", "numba", "prosper",
            "pipeline", "modeling", "registry", "webview"]

a = Analysis(["rsw/__main__.py"], pathex=["."], binaries=binaries, datas=datas,
             hiddenimports=hiddenimports, excludes=excludes, noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="PySegment", console=False, upx=False,
          icon="rsw/resources/rsw.ico")
coll = COLLECT(exe, a.binaries, a.datas, upx=False, name="PySegment")
