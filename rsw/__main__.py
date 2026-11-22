"""Start the workstation:  python -m rsw [project folder]
Built program: PySegment.exe [project folder];  PySegment.exe --self-test <result.json> checks the installation
with a synthetic patient (used by the automatic build)."""
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path


def self_test(out: str) -> int:
    """Synthetic case: open it in the window, start a reader, paint, save on the PET grid, radiomics."""
    checks, code = [], 0
    try:
        import numpy as np
        import SimpleITK as sitk
        from PySide6 import QtWidgets

        from rsw import APP_NAME, APP_VERSION
        from rsw.gui.main_window import MainWindow
        from rsw.radiomics import Settings, extract
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        tmp = Path(tempfile.mkdtemp(prefix="rsw_selftest_"))
        case = tmp / "study" / "SYN001"
        case.mkdir(parents=True)
        pet = np.ones((40, 48, 48), np.float32)
        pet[18:24, 20:28, 20:28] = 9.0
        for arr, name in ((pet, "PET_SUV.nii.gz"), ((pet > 4).astype(np.uint8), "lesions_auto.nii.gz")):
            im = sitk.GetImageFromArray(arr)
            im.SetSpacing((4.0, 4.0, 4.0))
            sitk.WriteImage(im, str(case / name))
        w = MainWindow()
        w.resize(1366, 768)
        w.show()
        w.open_folder(tmp / "study")
        w.open_case("SYN001", ask=False)
        w.start_reader()
        s = w.session
        a0 = s.shape[0] - 1 - 20
        w.viewer.views["axial"].index = a0
        w.on_paint("axial", 5, 5, True, True)
        checks.append({"check": "paint", "ok": bool(s.layer().voxels > int((pet > 4).sum()))})
        res = s.save_layer()
        saved = sitk.ReadImage(res["path"])
        checks.append({"check": "mask on PET grid", "ok": saved.GetSize() == (48, 48, 40)})
        df = extract(pet, s.geo, (pet > 4).astype(np.uint8), "SYN001", "auto", Settings(resample_mm=None))
        checks.append({"check": "radiomics", "ok": bool(len(df) == 2 and df.loc[0, "original_firstorder_Maximum"] == 9.0)})
        # elastix engine (itk-elastix, lazily loaded modules): recover a 4 mm shift of a smooth blob
        from rsw import registration as rg
        from rsw.io import Volume
        z, y, x = np.mgrid[:40, :48, :48]
        blob = np.exp(-(((z - 20) / 6.0) ** 2 + ((y - 24) / 8.0) ** 2 + ((x - 24) / 7.0) ** 2)).astype(np.float32) * 100
        f = Volume("F", blob, s.geo, "PET", "SUV")
        m = Volume("M", np.roll(blob, 1, axis=2), s.geo, "PET", "SUV")
        r = rg.register(f, m, mode="rigid", engine="elastix", init="identity")
        shift = r.transform.TransformPoint((0.0, 0.0, 0.0))
        checks.append({"check": "elastix registration", "ok": bool(abs(abs(shift[0]) - 4.0) < 1.0),
                       "detail": f"x shift {shift[0]:.2f} mm (expected 4)"})
        for _ in range(10):
            app.processEvents()
        shot = Path(out).with_suffix(".png")
        w.grab().save(str(shot))
        checks.append({"check": "window drawn", "ok": shot.exists()})
        w.session.layers[w.session.active].saved = True
        w.close()
        code = 0 if all(c["ok"] for c in checks) else 1
        info = {"app": APP_NAME, "version": APP_VERSION}
    except Exception:  # noqa: BLE001
        checks.append({"check": "exception", "ok": False, "detail": traceback.format_exc()})
        code, info = 1, {}
    Path(out).write_text(json.dumps({"exit": code, **info, "checks": checks}, indent=1, default=str))
    return code


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--self-test"]:
        return self_test(argv[1] if len(argv) > 1 else os.path.join(tempfile.gettempdir(), "rsw_selftest.json"))
    from PySide6 import QtGui, QtWidgets

    from rsw import APP_NAME
    from rsw.gui.main_window import MainWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    icon = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent)) / "rsw" / "resources" / "rsw.ico"
    if icon.exists():
        app.setWindowIcon(QtGui.QIcon(str(icon)))
    w = MainWindow()
    w.show()
    if argv:
        w.open_folder(argv[0])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
