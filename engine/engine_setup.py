"""Engine setup assistant: finds, downloads and installs the Shap-E engine,
and hosts the Quality presets. Works on machines without an NVIDIA GPU (the
engine then runs on the CPU, which is slow but works)."""

import importlib.util
import json
import os
import platform
import re
import shutil
import tarfile
import subprocess
import sys
import tempfile
import textwrap
import urllib.request
import zipfile
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QVBoxLayout,
)

from engine.enhance_ui import BUTTON_STYLE, COMBO_STYLE
from engine.runtime import (  # noqa: F401
    app_data_dir, engine_python, is_frozen, uses_managed_python, venv_dir,
)

REPO_ZIP_URL = "https://github.com/openai/shap-e/archive/refs/heads/main.zip"
CLIP_ZIP_URL = "https://github.com/openai/CLIP/archive/refs/heads/main.zip"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
UV_URL = "https://github.com/astral-sh/uv/releases/latest/download/"
ENGINE_PYTHON_VERSION = "3.12"

# import name -> pip name (None = special handling)
REQUIRED = {
    "torch": "torch", "torchvision": "torchvision", "clip": None,
    "blobfile": "blobfile", "filelock": "filelock", "PIL": "Pillow",
    "requests": "requests", "tqdm": "tqdm", "scipy": "scipy",
    "numpy": "numpy", "yaml": "pyyaml", "ipywidgets": "ipywidgets",
    "ftfy": "ftfy", "regex": "regex", "packaging": "packaging",
}
EXTRA_PIP = {"skimage": "scikit-image", "cv2": "opencv-python-headless",
             "IPython": "ipython", "fire": "fire", "humanize": "humanize",
             "matplotlib": "matplotlib"}
VERIFY = (
    "import torch\n"
    "import shap_e.models.download, shap_e.diffusion.sample, shap_e.util.notebooks\n"
    "print('SHAPEFORGE_OK', torch.cuda.is_available())\n"
)

PRESETS = {
    "Draft (fastest)": {"steps": 32, "grid": 64},
    "Standard": {"steps": 64, "grid": 128},
    "High detail": {"steps": 64, "grid": 192},
}
PRESET_HELP = (
    "Draft: quick and rough, good for laptops and testing prompts. "
    "Standard: the default. High detail: finer shape and sharper colors, "
    "slower and heavier. Results vary: judge on your own machine."
)


# ----------------------------------------------------------- locations --

def repo_ok(home):
    return (Path(home) / "shap_e" / "__init__.py").exists()


def resolve_home():
    """SHAPE_E_HOME if set, else an existing ~/shap-e, else the app-data copy."""
    env = os.environ.get("SHAPE_E_HOME")
    if env:
        return Path(env).expanduser().resolve()
    legacy = Path.home() / "shap-e"
    if repo_ok(legacy):
        return legacy.resolve()
    return (app_data_dir() / "shap-e").resolve()


def has_nvidia():
    return shutil.which("nvidia-smi") is not None


def missing_modules(python=None):
    python = python or engine_python()
    if os.path.abspath(python) == os.path.abspath(sys.executable):
        return [m for m in REQUIRED if importlib.util.find_spec(m) is None]
    if not os.path.exists(python):
        return list(REQUIRED)
    code = ("import importlib.util, json; print(json.dumps([m for m in "
            f"{list(REQUIRED)!r} if importlib.util.find_spec(m) is None]))")
    try:
        out = subprocess.run([python, "-c", code], capture_output=True,
                             text=True, timeout=60).stdout
        return json.loads(out.strip().splitlines()[-1])
    except Exception:
        return list(REQUIRED)


def quick_status(home):
    missing = missing_modules()
    repo = repo_ok(home)
    return {"repo": repo, "missing": missing, "ready": repo and not missing}


def default_preset():
    return "Standard" if has_nvidia() else "Draft (fastest)"


# ---------------------------------------------------------------- plan --

def pip_plan(missing, nvidia, platform, uv=False):
    """-> list of pip argument lists, in install order."""
    plan = []
    torch_pkgs = [m for m in ("torch", "torchvision") if m in missing]
    if torch_pkgs:
        args = list(torch_pkgs)
        if uv:
            args += ["--torch-backend", "auto"]  # CUDA, ROCm or CPU as fits
        elif platform.startswith("linux") and not nvidia:
            args += ["--index-url", CPU_INDEX]  # small CPU-only build
        plan.append(args)
    others = [REQUIRED[m] for m in missing if REQUIRED.get(m)]
    others = [p for p in others if p not in ("torch", "torchvision")]
    if others:
        plan.append(others)
    if "clip" in missing:
        plan.append(["--no-deps", CLIP_ZIP_URL])
    return plan


def missing_module(output):
    match = re.search(r"No module named '([\w.]+)'", output)
    return match.group(1).split(".")[0] if match else None


# ---------------------------------------------------------------- task --

class SetupTask(QThread):
    line = Signal(str)
    finished_setup = Signal(bool, str, str)  # ok, message, device

    def __init__(self, home, pip_runner=None, verify_runner=None, uv_mode=None):
        super().__init__()
        self.home = Path(home)
        self.uv_mode = uses_managed_python() if uv_mode is None else uv_mode
        self.uv = None
        self._cancel = False
        self._proc = None
        self._pip = pip_runner or self._run_pip
        self._verify = verify_runner or self._run_verify

    def cancel(self):
        self._cancel = True
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()

    def _check_cancel(self):
        if self._cancel:
            raise RuntimeError("Cancelled.")

    def run(self):
        try:
            self._download_repo()
            self._ensure_python()
            self._install_libraries()
            device = self._verify_import()
            note = ("NVIDIA GPU found: generation will use it." if device == "cuda"
                    else "No CUDA GPU in use: models will run on the CPU. Pick "
                         "the Draft quality preset. Expect minutes per model.")
            self.finished_setup.emit(True, "Engine ready. " + note, device)
        except Exception as error:
            self.finished_setup.emit(False, str(error), "")

    # -- steps
    def _download_repo(self):
        if repo_ok(self.home):
            self.line.emit("Shap-E code already present.")
            return
        self.line.emit(f"Downloading Shap-E code to {self.home} ...")
        self.home.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(self.home.parent)) as tmp:
            archive = Path(tmp) / "shap-e.zip"
            try:
                self._fetch(REPO_ZIP_URL, archive)
            except OSError as error:
                raise RuntimeError(f"Could not download Shap-E ({error}). "
                                   "Check your internet connection.") from error
            self.home.mkdir(parents=True, exist_ok=True)
            root = self.home.resolve()
            with zipfile.ZipFile(archive) as zf:
                for member in zf.infolist():
                    parts = Path(member.filename).parts[1:]  # drop shap-e-main/
                    if not parts:
                        continue
                    target = (root / Path(*parts)).resolve()
                    if root not in target.parents and target != root:
                        raise RuntimeError("Unsafe path in download; aborting.")
                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(member) as src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst)
        if not repo_ok(self.home):
            raise RuntimeError("Download finished but shap_e/ was not found.")
        self.line.emit("Shap-E code installed.")

    def _fetch(self, url, destination):
        with urllib.request.urlopen(url, timeout=60) as response, open(destination, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done, last = 0, -1
            while True:
                self._check_cancel()
                chunk = response.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                step = int(done * 10 / total) if total else done >> 22
                if step != last:
                    last = step
                    self.line.emit(f"  downloaded {done / 1e6:.1f} MB"
                                   + (f" ({done * 100 // total}%)" if total else ""))

    def _ensure_python(self):
        """Packaged app: create the engine's own Python (via uv) once."""
        if not self.uv_mode:
            return
        self.uv = self._ensure_uv()
        python = engine_python()
        if os.path.exists(python):
            self.line.emit("Engine Python already present.")
            return
        self.line.emit(f"Creating engine Python {ENGINE_PYTHON_VERSION} (one time) ...")
        code, output = self._stream([self.uv, "venv", "--python", ENGINE_PYTHON_VERSION,
                                     str(venv_dir())])
        if code != 0 or not os.path.exists(python):
            tail = "\n".join(output.strip().splitlines()[-5:])
            raise RuntimeError("Could not create the engine's Python:\n" + tail)

    def _ensure_uv(self):
        target = app_data_dir() / "tools" / ("uv.exe" if sys.platform.startswith("win") else "uv")
        if target.exists():
            return str(target)
        machine = platform.machine().lower()
        arch = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"
        if sys.platform.startswith("win"):
            asset = f"uv-{arch}-pc-windows-msvc.zip"
        elif sys.platform == "darwin":
            asset = f"uv-{arch}-apple-darwin.tar.gz"
        else:
            asset = f"uv-{arch}-unknown-linux-gnu.tar.gz"
        self.line.emit("Downloading the uv installer tool ...")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / asset
            try:
                self._fetch(UV_URL + asset, archive)
            except OSError as error:
                raise RuntimeError(f"Could not download uv ({error}). "
                                   "Check your internet connection.") from error
            wanted = target.name
            if asset.endswith(".zip"):
                with zipfile.ZipFile(archive) as zf:
                    name = next((n for n in zf.namelist() if Path(n).name == wanted), None)
                    if name is None:
                        raise RuntimeError("uv download did not contain uv.")
                    with zf.open(name) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
            else:
                with tarfile.open(archive) as tf:
                    member = next((m for m in tf.getmembers()
                                   if m.isfile() and Path(m.name).name == wanted), None)
                    if member is None:
                        raise RuntimeError("uv download did not contain uv.")
                    with tf.extractfile(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        target.chmod(0o755)
        return str(target)

    def _install_libraries(self):
        status = quick_status(self.home)
        plan = pip_plan(status["missing"], has_nvidia(), sys.platform, uv=self.uv_mode)
        if not plan:
            self.line.emit("Python libraries already installed.")
        for args in plan:
            self._check_cancel()
            self.line.emit(("uv pip install " if self.uv_mode else "pip install ") + " ".join(args))
            code, output = self._pip(args)
            if code != 0:
                raise RuntimeError(self._explain_pip_failure(args, output))

    def _verify_import(self):
        for _ in range(8):
            self._check_cancel()
            self.line.emit("Checking that the engine can start ...")
            code, output = self._verify()
            match = re.search(r"SHAPEFORGE_OK (True|False)", output)
            if code == 0 and match:
                return "cuda" if match.group(1) == "True" else "cpu"
            module = missing_module(output)
            if not module or module in ("torch", "shap_e"):
                tail = "\n".join(output.strip().splitlines()[-6:])
                raise RuntimeError("The engine did not start:\n" + tail)
            pip_name = EXTRA_PIP.get(module) or REQUIRED.get(module) or module
            self.line.emit(f"Also needs {pip_name}; installing ...")
            code, output = self._pip(["--no-deps", CLIP_ZIP_URL] if module == "clip" else [pip_name])
            if code != 0:
                raise RuntimeError(self._explain_pip_failure([pip_name], output))
        raise RuntimeError("Still missing libraries after several attempts.")

    # -- subprocess helpers
    def _stream(self, command, env=None, cwd=None, quiet=False):
        self._proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env, cwd=cwd)
        lines = []
        for raw in self._proc.stdout:
            raw = raw.rstrip()
            lines.append(raw)
            if raw and not quiet:
                self.line.emit(raw)
        self._proc.wait()
        code, self._proc = self._proc.returncode, None
        self._check_cancel()
        return code, "\n".join(lines)

    def _run_pip(self, args):
        if self.uv_mode:
            return self._stream([self.uv, "pip", "install", "--python", engine_python(), *args])
        return self._stream([engine_python(), "-m", "pip", "install",
                             "--disable-pip-version-check", "--progress-bar", "off", *args])

    def _run_verify(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.home) + os.pathsep + env.get("PYTHONPATH", "")
        return self._stream([engine_python(), "-c", VERIFY], env=env,
                            cwd=str(self.home), quiet=True)

    @staticmethod
    def _explain_pip_failure(args, output):
        low = output.lower()
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        if "externally-managed-environment" in low:
            return ("Your system Python does not allow installing packages. Run "
                    "ShapeForge inside a virtual environment: "
                    "python3 -m venv .venv && source .venv/bin/activate")
        if ("no matching distribution" in low or "could not find a version" in low
                or "no solution found" in low or "no version of" in low):
            return (f"No build of {args[0]} is available for Python {version} on "
                    "this system yet. Use Python 3.10-3.12 (for example "
                    "'uv venv --python 3.12') and run ShapeForge there.")
        tail = "\n".join(output.strip().splitlines()[-5:])
        return f"Installing {' '.join(args)} failed:\n{tail}"


# -------------------------------------------------------------- dialog --

class SetupDialog(QDialog):
    def __init__(self, parent, home):
        super().__init__(parent)
        self.home = Path(home)
        self.task = None
        self.device = ""
        self.setWindowTitle("Set up the Shap-E engine")
        self.resize(640, 480)
        layout = QVBoxLayout(self)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Install progress appears here.")
        layout.addWidget(self.log, 1)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1)
        self.bar.setTextVisible(False)
        layout.addWidget(self.bar)
        row = QHBoxLayout()
        self.go = QPushButton("Download && install engine")
        self.close_button = QPushButton("Close")
        for button in (self.go, self.close_button):
            button.setStyleSheet(BUTTON_STYLE)
        self.go.clicked.connect(self.start)
        self.close_button.clicked.connect(self.close)
        row.addStretch(1)
        row.addWidget(self.go)
        row.addWidget(self.close_button)
        layout.addLayout(row)
        self.refresh()

    def refresh(self):
        status = quick_status(self.home)
        mark = lambda ok: "\u2713" if ok else "\u2717"
        libs = ("all installed" if not status["missing"]
                else "missing: " + ", ".join(status["missing"]))
        hardware = ("NVIDIA GPU detected." if has_nvidia() else
                    "No NVIDIA GPU detected: generation will run on the CPU "
                    "(works, but takes minutes per model; use the Draft preset).")
        self.summary.setText(
            f"<b>ShapeForge needs the Shap-E engine to generate models.</b><br><br>"
            f"{mark(status['repo'])} Shap-E code: {self.home}<br>"
            f"{mark(not status['missing'])} PyTorch and libraries: {libs}<br><br>"
            f"{hardware}<br><br>"
            "Downloading installs the Shap-E code (small), the missing Python "
            "libraries (PyTorch is large: about 200 MB CPU-only, 2 GB+ with CUDA) "
            "and, the first time you generate, the AI models (about 3 GB)."
            + ("<br><br>This app keeps the engine in its own private Python "
               "environment, so nothing is installed system-wide."
               if uses_managed_python() else ""))
        self.go.setEnabled(not status["ready"] and self.task is None)
        if status["ready"]:
            self.go.setText("Engine is installed")

    def start(self):
        if self.task is not None:
            return
        self.log.clear()
        self.bar.setRange(0, 0)
        self.go.setText("Working...")
        self.go.setEnabled(False)
        self.close_button.setText("Cancel")
        self.task = SetupTask(self.home)
        self.task.line.connect(self.log.appendPlainText)
        self.task.finished_setup.connect(self._finished)
        self.task.start()

    def _finished(self, ok, message, device):
        self.bar.setRange(0, 1)
        self.close_button.setText("Close")
        task, self.task = self.task, None
        if task is not None:
            task.wait()
        self.device = device
        self.log.appendPlainText(("\nDONE: " if ok else "\nFAILED: ") + message)
        self.go.setText("Download && install engine" if not ok else "Engine is installed")
        self.refresh()
        if ok:
            self.go.setEnabled(False)

    def reject(self):
        if self.task is not None:
            self.task.cancel()
            self.task.wait(5000)
            self.task = None
        super().reject()

    def closeEvent(self, event):
        if self.task is not None:
            self.task.cancel()
            self.task.wait(5000)
            self.task = None
        super().closeEvent(event)


# -------------------------------------------------------- app wiring --

def ensure_engine(window):
    """Called before generating. True if the engine can run."""
    if not hasattr(window, "engine_home"):
        return True  # setup UI not installed (e.g. tests)
    home = Path(window.engine_home())
    if quick_status(home)["ready"]:
        return True
    SetupDialog(window, home).exec()
    return quick_status(home)["ready"]


def install_engine_ui(window, engine_proc):
    """Quality picker + Engine setup button; no engine code runs here."""
    window._engine_proc = engine_proc
    window.engine_home = lambda: window._engine_proc.home
    layout = window.generate_button.parentWidget().layout()
    index = layout.indexOf(window.generate_button)

    row = QHBoxLayout()
    label = QLabel("Quality")
    window.quality_combo = QComboBox()
    window.quality_combo.addItems(list(PRESETS))
    window.quality_combo.setStyleSheet(COMBO_STYLE)
    for widget in (label, window.quality_combo):
        widget.setToolTip(textwrap.fill(PRESET_HELP, 58))
    row.addWidget(label)
    row.addWidget(window.quality_combo, 1)
    layout.insertLayout(index + 1, row)

    def apply_preset(name):
        window._engine_proc.options = dict(PRESETS[name])
    window.quality_combo.currentTextChanged.connect(apply_preset)
    window.quality_combo.setCurrentText(default_preset())
    apply_preset(window.quality_combo.currentText())

    window.engine_button = QPushButton("Engine setup...")
    window.engine_button.setStyleSheet(BUTTON_STYLE)
    window.engine_button.setToolTip(textwrap.fill(
        "Check, download or repair the Shap-E engine (code, PyTorch and "
        "libraries) that generates the models.", 58))
    window.engine_button.clicked.connect(
        lambda: SetupDialog(window, window.engine_home()).exec())
    anchor = window.status
    anchor.parentWidget().layout().insertWidget(
        anchor.parentWidget().layout().indexOf(anchor), window.engine_button)
