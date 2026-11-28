# PyInstaller spec for ShapeForge (Linux, Windows, macOS).
# Build from the repo root:  pyinstaller --noconfirm packaging/shapeforge.spec
# The app bundles only the GUI. PyTorch and Shap-E are installed later by the
# in-app setup assistant into a private Python environment.
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent
VERSION = os.environ.get("SHAPEFORGE_VERSION", "dev")

datas = [(str(p), "engine") for p in sorted((ROOT / "engine").glob("*.py"))]
datas += collect_data_files("trimesh") + collect_data_files("pyqtgraph")
hiddenimports = (
    collect_submodules("OpenGL.platform")
    + collect_submodules("pyqtgraph.opengl")
    + ["OpenGL.arrays.numpymodule", "OpenGL.arrays.ctypesarrays",
       "OpenGL.arrays.lists", "OpenGL.arrays.numbers", "OpenGL.arrays.strings",
       "scipy.spatial.transform._rotation_groups"]
)
try:  # optional mesh decimation
    import fast_simplification  # noqa: F401
    hiddenimports.append("fast_simplification")
except ImportError:
    pass

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "torch", "torchvision", "matplotlib", "IPython",
              "PyQt5", "PyQt6", "PySide2", "sklearn", "skimage", "pandas",
              "dask", "numba", "sympy", "pytest"],
)

if sys.platform.startswith("linux"):
    # Use the host's graphics drivers, never copies from the build machine.
    skip = ("libGL.", "libGLX.", "libEGL.", "libOpenGL.", "libGLdispatch.")
    a.binaries = [b for b in a.binaries if not b[0].startswith(skip)]

pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="ShapeForge",
          console=False, icon=str(ROOT / "assets" / "icon.png"))
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="ShapeForge")

if sys.platform == "darwin":
    app = BUNDLE(
        coll, name="ShapeForge.app", icon=str(ROOT / "assets" / "icon.png"),
        bundle_identifier="com.somore100.shapeforge",
        info_plist={
            "CFBundleName": "ShapeForge",
            "CFBundleShortVersionString": VERSION.lstrip("v"),
            "NSHighResolutionCapable": True,
        },
    )
