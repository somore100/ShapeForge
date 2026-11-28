"""Where things live, and which Python runs the engine.

Source checkout: the engine runs in the same Python as the app (as before).
Packaged app (PyInstaller): there is no usable Python inside the bundle, so
the engine runs in its own virtual environment under the user's data folder,
created by the setup assistant. SHAPEFORGE_PYTHON overrides either case.
"""

import os
import sys
from pathlib import Path


def is_frozen():
    return bool(getattr(sys, "frozen", False))


def app_data_dir():
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "ShapeForge"


def venv_dir():
    return app_data_dir() / "engine-venv"


def venv_python():
    sub = ("Scripts", "python.exe") if sys.platform.startswith("win") else ("bin", "python")
    return venv_dir().joinpath(*sub)


def engine_python():
    override = os.environ.get("SHAPEFORGE_PYTHON")
    if override:
        return override
    return str(venv_python()) if is_frozen() else sys.executable


def uses_managed_python():
    """True when the setup assistant must create the engine's Python itself."""
    return is_frozen() and not os.environ.get("SHAPEFORGE_PYTHON")


def output_root():
    """Where generated models go. Next to the code in a checkout; in a visible
    folder in the home directory for a packaged app (the bundle is read-only)."""
    if is_frozen():
        return Path.home() / "ShapeForge"
    return Path(__file__).resolve().parent.parent
