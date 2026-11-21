"""UI side of mesh cleanup and export, mixed into the ShapeForge window."""

import shutil
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QSpinBox
)

from engine.enhance_ui import BUTTON_STYLE, COMBO_STYLE
from engine.meshtools import (  # noqa: F401  (best_mesh etc. re-exported)
    CLEAN_OBJ, DECIMATE_CHOICES, best_mesh, cleanup, decimation_available,
    export_mesh, purge_clean_files, summarize
)

SPIN_STYLE = (
    "QSpinBox { background: #151820; border: 1px solid #414858;"
    " border-radius: 7px; padding: 5px 8px; color: white; }"
)
TOGGLE_STYLE = (
    "QPushButton { background: #2b3140; padding: 6px 10px; }"
    "QPushButton:hover { background: #3a4256; }"
    "QPushButton:checked { background: #3478f6; }"
    "QPushButton:disabled { background: #414858; color: #8892a6; }"
)
ORIENTATIONS = (
    ("Orientation: auto", "auto"),
    ("Keep as generated", "keep"),
    ("Convert to Y-up", "yup"),
)


class CleanupTask(QThread):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, folder, options, parent=None):
        super().__init__(parent)
        self.folder = folder
        self.options = options

    def run(self):
        try:
            report = cleanup(self.folder, **self.options)
            self.done.emit({"folder": str(self.folder), "report": report})
        except Exception as error:
            self.failed.emit(str(error))


class CleanupMixin:
    # ------------------------------------------------------------ building

    def build_cleanup_ui(self, layout):
        self.clean_task = None
        self.previewed_path = None

        top = QHBoxLayout()
        self.clean_fragments = QCheckBox("Remove fragments under")
        self.clean_fragments.setChecked(True)
        self.clean_fragments.setToolTip(
            "Delete disconnected pieces smaller than this share of the "
            "largest piece"
        )
        top.addWidget(self.clean_fragments)
        self.clean_fragment_size = QSpinBox()
        self.clean_fragment_size.setRange(1, 50)
        self.clean_fragment_size.setValue(5)
        self.clean_fragment_size.setSuffix(" %")
        self.clean_fragment_size.setStyleSheet(SPIN_STYLE)
        top.addWidget(self.clean_fragment_size)
        self.clean_holes = QCheckBox("Fill holes")
        self.clean_holes.setChecked(True)
        top.addWidget(self.clean_holes)
        top.addWidget(QLabel("Smooth:"))
        self.clean_smooth = QSpinBox()
        self.clean_smooth.setRange(0, 50)
        self.clean_smooth.setValue(10)
        self.clean_smooth.setSuffix(" passes")
        self.clean_smooth.setStyleSheet(SPIN_STYLE)
        self.clean_smooth.setToolTip(
            "Volume-preserving smoothing. 0 turns it off."
        )
        top.addWidget(self.clean_smooth)
        top.addStretch(1)
        layout.addLayout(top)

        bottom = QHBoxLayout()
        bottom.addWidget(QLabel("Decimate:"))
        self.clean_decimate = QComboBox()
        self.clean_decimate.setStyleSheet(COMBO_STYLE)
        for label, value in DECIMATE_CHOICES:
            self.clean_decimate.addItem(label, value)
        if not decimation_available():
            self.clean_decimate.setEnabled(False)
            self.clean_decimate.setToolTip(
                "Needs a small extra package:\n"
                "python3 -m pip install --no-deps fast-simplification"
            )
        else:
            self.clean_decimate.setToolTip(
                "Reduce the face count. Very low targets can open holes."
            )
        bottom.addWidget(self.clean_decimate)

        self.clean_button = QPushButton("Clean Up Model")
        self.clean_button.setStyleSheet(BUTTON_STYLE)
        self.clean_button.setToolTip(
            "Cleans the model in the preview. The original is kept; the "
            "result is saved as generated_clean.obj/.ply beside it."
        )
        self.clean_button.setEnabled(False)
        self.clean_button.clicked.connect(self.clean_model)
        bottom.addWidget(self.clean_button)

        self.variant_group = QButtonGroup(self)
        self.variant_group.setExclusive(True)
        self.show_original_button = QPushButton("Original")
        self.show_cleaned_button = QPushButton("Cleaned")
        for button, variant in (
            (self.show_original_button, "original"),
            (self.show_cleaned_button, "clean"),
        ):
            button.setCheckable(True)
            button.setEnabled(False)
            button.setStyleSheet(TOGGLE_STYLE)
            button.clicked.connect(
                lambda checked, v=variant: self.show_variant(v)
            )
            self.variant_group.addButton(button)
            bottom.addWidget(button)

        self.auto_clean = QCheckBox("Auto-clean batch")
        self.auto_clean.setToolTip(
            "Apply these settings to every batch model as it finishes"
        )
        bottom.addWidget(self.auto_clean)
        bottom.addStretch(1)
        layout.addLayout(bottom)

    def add_extra_export_buttons(self, row):
        self.stl_button = QPushButton("Export STL")
        self.glb_button = QPushButton("Export GLB")
        self.stl_button.setToolTip("For 3D printing")
        self.glb_button.setToolTip("For game engines and Blender")
        for button, extension in (
            (self.stl_button, "stl"), (self.glb_button, "glb")
        ):
            button.setEnabled(False)
            button.clicked.connect(
                lambda checked, e=extension: self.export(e)
            )
            row.addWidget(button)

        self.export_orientation = QComboBox()
        self.export_orientation.setStyleSheet(COMBO_STYLE)
        for label, value in ORIENTATIONS:
            self.export_orientation.addItem(label, value)
        self.export_orientation.setToolTip(
            "Auto: GLB is exported Y-up as the glTF format expects; "
            "other formats keep the generated orientation."
        )
        row.addWidget(self.export_orientation)

    def shutdown_cleanup(self):
        task = self.clean_task
        if task is not None and task.isRunning():
            task.wait(5000)

    # ----------------------------------------------------------- preview hook

    def mesh_loaded(self, path):
        """Called by load_mesh after a model is shown."""
        self.previewed_path = Path(path)
        for button in (
            self.obj_button, self.ply_button,
            self.stl_button, self.glb_button, self.clean_button,
        ):
            button.setEnabled(True)

        folder = self.previewed_path.parent
        has_clean = (folder / CLEAN_OBJ).exists()
        is_clean = self.previewed_path.name == CLEAN_OBJ
        for button in (self.show_original_button, self.show_cleaned_button):
            button.setEnabled(has_clean)
        self.variant_group.setExclusive(False)
        self.show_original_button.setChecked(has_clean and not is_clean)
        self.show_cleaned_button.setChecked(has_clean and is_clean)
        self.variant_group.setExclusive(True)

    def show_variant(self, variant):
        if self.previewed_path is None:
            return
        folder = self.previewed_path.parent
        target = folder / (CLEAN_OBJ if variant == "clean" else "generated.obj")
        try:
            self.load_mesh(target)
        except Exception as error:
            self.log.append(f"Preview error: {error}")

    # ---------------------------------------------------------------- cleanup

    def cleanup_options(self):
        return {
            "remove_fragments": self.clean_fragments.isChecked(),
            "fragment_fraction": self.clean_fragment_size.value() / 100.0,
            "fill_holes": self.clean_holes.isChecked(),
            "smooth": self.clean_smooth.value(),
            "decimate": self.clean_decimate.currentData(),
        }

    def clean_model(self):
        if self.previewed_path is None:
            return
        if self.clean_task is not None and self.clean_task.isRunning():
            return
        folder = self.previewed_path.parent
        self.clean_button.setEnabled(False)
        self.status.setText("Cleaning up model...")
        self.clean_task = CleanupTask(folder, self.cleanup_options(), self)
        self.clean_task.done.connect(self.clean_done)
        self.clean_task.failed.connect(self.clean_failed)
        self.clean_task.start()

    def clean_done(self, result):
        self.clean_button.setEnabled(True)
        summary = summarize(result["report"])
        self.log.append(f"Cleanup: {summary}")
        self.status.setText(f"Cleaned: {summary}")
        try:
            self.load_mesh(Path(result["folder"]) / CLEAN_OBJ)
        except Exception as error:
            self.log.append(f"Preview error: {error}")

    def clean_failed(self, message):
        self.clean_button.setEnabled(True)
        self.status.setText("Cleanup failed.")
        self.log.append(f"Cleanup failed: {message}")

    def auto_clean_job(self, success, output_dir, index):
        if not success or not self.auto_clean.isChecked():
            return
        try:
            report = cleanup(output_dir, **self.cleanup_options())
            self.log.append(f"Auto-clean job {index}: {summarize(report)}")
        except Exception as error:
            self.log.append(f"Auto-clean failed for job {index}: {error}")

    # ----------------------------------------------------------------- export

    def export(self, extension):
        source = self.previewed_path
        if source is None or not source.exists():
            QMessageBox.warning(
                self, "Nothing to export", "Load or generate a model first."
            )
            return

        destination, _ = QFileDialog.getSaveFileName(
            self, f"Export {extension.upper()}", f"my_model.{extension}",
            f"{extension.upper()} files (*.{extension})",
        )
        if not destination:
            return
        if not destination.lower().endswith("." + extension):
            destination += "." + extension

        orientation = self.export_orientation.currentData()
        rotates = orientation == "yup" or (
            orientation == "auto" and extension == "glb"
        )
        try:
            sibling = source.with_suffix("." + extension)
            if extension in ("obj", "ply") and not rotates and sibling.exists():
                shutil.copy2(sibling, destination)
            else:
                export_mesh(source, destination, orientation)
        except Exception as error:
            QMessageBox.warning(self, "Export failed", str(error))
            return
        self.log.append(f"Exported: {destination}")
