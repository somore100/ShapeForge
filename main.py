import os
import sys
import re
import subprocess
import shutil
from datetime import datetime
from pathlib import Path

import trimesh
import numpy as np
import pyqtgraph.opengl as gl

from PySide6.QtCore import QThread, Signal, QUrl, QSize, Qt
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QFileDialog, QMessageBox, QSplitter,
    QListWidget, QListWidgetItem, QListView
)

APP_DIR = Path(__file__).resolve().parent
SHAPE_E_HOME = Path(
    os.environ.get("SHAPE_E_HOME", str(Path.home() / "shap-e"))
).expanduser().resolve()

ENGINE = APP_DIR / "engine" / "generate.py"
SERVER = APP_DIR / "engine" / "server.py"
OUTPUT = APP_DIR / "generated"
BATCH_OUTPUT = APP_DIR / "outputs"


def run_generator_oneshot(prompt, output_dir, log_signal):
    command = [
        sys.executable,
        str(ENGINE),
        "--prompt", prompt,
        "--output", str(output_dir)
    ]

    child_env = os.environ.copy()
    existing_path = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = (
        str(SHAPE_E_HOME)
        + (os.pathsep + existing_path if existing_path else "")
    )

    process = subprocess.Popen(
        command,
        cwd=str(SHAPE_E_HOME),
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        log_signal.emit(line.rstrip())

    process.wait()
    return process.returncode


import atexit
from engine.client import PersistentEngine
from engine.batchtools import (
    BATCH_HELP, parse_prompts, render_thumbnail, placeholder_image
)

ENGINE_PROC = PersistentEngine(SERVER, SHAPE_E_HOME)
atexit.register(ENGINE_PROC.shutdown)


def run_generator(prompt, output_dir, log_signal):
    if SERVER.exists():
        code = ENGINE_PROC.run(prompt, output_dir, log_signal.emit)
        if code != 2:
            return code
        log_signal.emit("Falling back to one-shot generator...")
    return run_generator_oneshot(prompt, output_dir, log_signal)


class Generator(QThread):
    log = Signal(str)
    finished_generation = Signal(bool, str, str)

    def __init__(self, prompt, output_dir):
        super().__init__()
        self.prompt = prompt
        self.output_dir = Path(output_dir)

    def run(self):
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            code = run_generator(
                self.prompt, self.output_dir, self.log
            )

            if code == 0:
                self.finished_generation.emit(
                    True, "Generation completed!",
                    str(self.output_dir)
                )
            else:
                self.finished_generation.emit(
                    False, f"Generator exited with code {code}",
                    str(self.output_dir)
                )

        except Exception as error:
            self.finished_generation.emit(
                False, str(error), str(self.output_dir)
            )


class BatchGenerator(QThread):
    log = Signal(str)
    job_finished = Signal(bool, str, str, int, int)
    finished_batch = Signal(bool)

    def __init__(self, prompts, batch_dir):
        super().__init__()
        self.prompts = prompts
        self.batch_dir = Path(batch_dir)
        self.stop_requested = False

    def request_stop(self):
        self.stop_requested = True

    def run(self):
        self.batch_dir.mkdir(parents=True, exist_ok=True)

        total = len(self.prompts)
        stopped = False

        for index, prompt in enumerate(self.prompts, start=1):
            if self.stop_requested:
                stopped = True
                self.log.emit("Batch stopped before the next job.")
                break

            slug = re.sub(
                r"[^a-zA-Z0-9]+", "_", prompt.lower()
            ).strip("_")[:35] or "model"

            job_dir = self.batch_dir / f"{index:02d}_{slug}"

            self.log.emit("")
            self.log.emit("=" * 42)
            self.log.emit(f"JOB {index}/{total}")
            self.log.emit(f"Prompt: {prompt}")
            self.log.emit(f"Output: {job_dir}")
            self.log.emit("Starting generation...")

            try:
                job_dir.mkdir(parents=True, exist_ok=True)
                code = run_generator(prompt, job_dir, self.log)

                if code == 0:
                    self.log.emit(f"JOB {index}: COMPLETED")
                    self.job_finished.emit(
                        True, prompt, str(job_dir), index, total
                    )
                else:
                    self.log.emit(
                        f"JOB {index}: FAILED (exit code {code})"
                    )
                    self.job_finished.emit(
                        False, prompt, str(job_dir), index, total
                    )

            except Exception as error:
                self.log.emit(f"JOB {index}: ERROR - {error}")
                self.job_finished.emit(
                    False, prompt, str(job_dir), index, total
                )

        self.finished_batch.emit(stopped)


class ShapeForge(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ShapeForge - Local Text-to-3D")
        self.resize(1200, 800)

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #20232b;
                color: #e8eaf0;
                font-size: 14px;
            }
            QTextEdit {
                background: #151820;
                border: 1px solid #414858;
                border-radius: 7px;
                padding: 10px;
                color: white;
            }
            QPushButton {
                background: #3478f6;
                border: none;
                border-radius: 7px;
                padding: 11px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover { background: #5590ff; }
            QPushButton:disabled { background: #414858; }
        """)

        self.mesh_item = None
        self.worker = None
        self.batch_worker = None
        self.latest_output_dir = OUTPUT

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        header = QLabel("◆  ShapeForge")
        header.setStyleSheet("font-size: 25px; font-weight: bold;")
        layout.addWidget(header)

        subtitle = QLabel(
            "Local AI 3D generation  •  Powered by Shap-E"
        )
        subtitle.setStyleSheet("color: #aab4c8;")
        layout.addWidget(subtitle)

        splitter = QSplitter()
        layout.addWidget(splitter, 1)

        # Left panel
        left = QWidget()
        left_layout = QVBoxLayout(left)

        left_layout.addWidget(QLabel("Single Generation"))

        self.prompt = QTextEdit()
        self.prompt.setPlaceholderText(
            "Describe one 3D model..."
        )
        self.prompt.setMaximumHeight(90)
        left_layout.addWidget(self.prompt)

        self.generate_button = QPushButton("Generate 3D Model")
        self.generate_button.clicked.connect(self.generate)
        left_layout.addWidget(self.generate_button)

        left_layout.addWidget(QLabel("Batch Build — up to 10 prompts"))
        self.batch_prompts = QTextEdit()
        self.batch_prompts.setPlaceholderText(
            "One prompt per line:\n"
            "A low-poly penguin\n"
            "A medieval treasure chest\n"
            "A small fantasy castle"
        )
        self.batch_prompts.setMaximumHeight(150)
        left_layout.addWidget(self.batch_prompts)
        self.batch_prompts.setPlaceholderText(BATCH_HELP)
        self.batch_count = QLabel()
        left_layout.addWidget(self.batch_count)
        self.batch_prompts.textChanged.connect(self.update_batch_count)
        self.update_batch_count()

        batch_buttons = QHBoxLayout()

        self.batch_button = QPushButton("Start Batch")
        self.batch_button.clicked.connect(self.start_batch)

        self.stop_button = QPushButton("Stop After Current")
        self.stop_button.clicked.connect(self.stop_batch)
        self.stop_button.setEnabled(False)

        batch_buttons.addWidget(self.batch_button)
        batch_buttons.addWidget(self.stop_button)
        left_layout.addLayout(batch_buttons)

        left_layout.addWidget(QLabel("Generation log"))

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        left_layout.addWidget(self.log, 1)

        self.status = QLabel("Ready • RTX 3060")
        self.status.setStyleSheet("color: #6ee7a0;")
        left_layout.addWidget(self.status)

        splitter.addWidget(left)

        # Right panel: 3D viewport
        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor("#171a22")
        self.view.setCameraPosition(
            distance=4, elevation=20, azimuth=45
        )

        grid = gl.GLGridItem()
        grid.setSize(4, 4)
        grid.setSpacing(0.25, 0.25)
        self.view.addItem(grid)

        right_layout.addWidget(self.view, 1)
        self.results = QListWidget()
        self.results.setViewMode(QListView.IconMode)
        self.results.setFlow(QListView.LeftToRight)
        self.results.setWrapping(False)
        self.results.setMovement(QListView.Static)
        self.results.setIconSize(QSize(96, 96))
        self.results.setGridSize(QSize(112, 134))
        self.results.setSpacing(4)
        self.results.setFixedHeight(156)
        self.results.setHorizontalScrollMode(QListWidget.ScrollPerPixel)
        self.results.setStyleSheet(
            "QListWidget { background: #151820; border: 1px solid #414858;"
            " border-radius: 7px; }"
            "QListWidget::item { color: #e8eaf0; }"
            "QListWidget::item:selected { background: #3478f6;"
            " border-radius: 6px; }"
        )
        self.results.itemClicked.connect(self.result_clicked)
        right_layout.addWidget(self.results)

        export_row = QHBoxLayout()

        self.obj_button = QPushButton("Export OBJ")
        self.obj_button.clicked.connect(
            lambda: self.export("obj")
        )
        self.obj_button.setEnabled(False)

        self.ply_button = QPushButton("Export PLY")
        self.ply_button.clicked.connect(
            lambda: self.export("ply")
        )
        self.ply_button.setEnabled(False)

        self.folder_button = QPushButton("Open Output Folder")
        self.folder_button.clicked.connect(self.open_output_folder)

        export_row.addWidget(self.obj_button)
        export_row.addWidget(self.ply_button)
        export_row.addWidget(self.folder_button)
        right_layout.addLayout(export_row)

        splitter.addWidget(right)
        splitter.setSizes([390, 810])

        if not ENGINE.exists():
            self.log.append(
                f"WARNING: Generator not found: {ENGINE}"
            )

    def set_running(self, running):
        self.generate_button.setEnabled(not running)
        self.batch_button.setEnabled(not running)
        self.stop_button.setEnabled(False)

    def generate(self):
        prompt = self.prompt.toPlainText().strip()

        if not prompt:
            QMessageBox.warning(
                self, "Missing prompt",
                "Please describe an object first."
            )
            return

        if not ENGINE.exists():
            QMessageBox.critical(
                self, "Missing engine",
                f"Could not find Shap-E generator:\n{ENGINE}"
            )
            return

        self.set_running(True)
        self.obj_button.setEnabled(False)
        self.ply_button.setEnabled(False)

        self.log.clear()
        self.log.append("Starting single generation...")
        self.status.setText("Generating on RTX 3060...")

        self.latest_output_dir = OUTPUT
        self.worker = Generator(prompt, OUTPUT)
        self.worker.log.connect(self.log.append)
        self.worker.finished_generation.connect(
            self.generation_finished
        )
        self.worker.start()

    def start_batch(self):
        prompts = parse_prompts(self.batch_prompts.toPlainText())

        if not prompts:
            QMessageBox.warning(
                self, "No prompts",
                "Enter at least one prompt."
            )
            return

        if len(prompts) > 10:
            QMessageBox.warning(
                self, "Too many prompts",
                "Batch Build currently supports a maximum of 10 prompts."
            )
            return

        if not ENGINE.exists():
            QMessageBox.critical(
                self, "Missing engine",
                f"Could not find Shap-E generator:\n{ENGINE}"
            )
            return

        batch_name = datetime.now().strftime(
            "batch_%Y%m%d_%H%M%S"
        )
        batch_dir = BATCH_OUTPUT / batch_name

        self.set_running(True)
        self.stop_button.setEnabled(True)
        self.results.clear()
        self.obj_button.setEnabled(False)
        self.ply_button.setEnabled(False)

        self.log.clear()
        self.log.append(
            f"Starting batch with {len(prompts)} prompts."
        )
        self.log.append("Jobs will run sequentially.")

        self.batch_worker = BatchGenerator(prompts, batch_dir)
        self.batch_worker.log.connect(self.log.append)
        self.batch_worker.job_finished.connect(
            self.batch_job_finished
        )
        self.batch_worker.finished_batch.connect(
            self.batch_finished
        )
        self.batch_worker.start()

        self.status.setText(
            f"Batch running • 0/{len(prompts)} completed"
        )

    def update_batch_count(self):
        count = len(parse_prompts(self.batch_prompts.toPlainText()))
        if count == 0:
            text, color = "No prompts yet", "#8892a6"
        elif count > 10:
            text, color = (
                f"{count} prompts detected - over the limit of 10", "#ff8a8a"
            )
        else:
            noun = "prompt" if count == 1 else "prompts"
            text, color = f"{count} {noun} detected (limit 10)", "#6ee7a0"
        self.batch_count.setText(text)
        self.batch_count.setStyleSheet(f"color: {color};")

    def add_result_item(self, success, prompt, output_dir, index):
        short = prompt if len(prompt) <= 13 else prompt[:12] + "..."
        try:
            if success:
                image = render_thumbnail(
                    Path(output_dir) / "generated.obj", 96
                )
            else:
                image = placeholder_image(96)
        except Exception:
            image = placeholder_image(96, "?", "#33384a")

        item = QListWidgetItem(
            QIcon(QPixmap.fromImage(image)), f"{index:02d} {short}"
        )
        item.setToolTip(prompt if success else f"FAILED: {prompt}")
        item.setData(Qt.UserRole, str(output_dir) if success else "")
        self.results.addItem(item)
        if success:
            self.results.setCurrentItem(item)
        self.results.scrollToItem(item)

    def result_clicked(self, item):
        directory = item.data(Qt.UserRole)
        if not directory:
            return
        self.latest_output_dir = Path(directory)
        try:
            self.load_mesh(self.latest_output_dir / "generated.obj")
            self.obj_button.setEnabled(True)
            self.ply_button.setEnabled(True)
            self.status.setText(f"Previewing: {item.toolTip()}")
        except Exception as error:
            self.log.append(f"Preview error: {error}")

    def stop_batch(self):
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.request_stop()
            self.stop_button.setEnabled(False)
            self.status.setText(
                "Stop requested • finishing current generation..."
            )
            self.log.append(
                "Stop requested. Current job will finish first."
            )

    def batch_job_finished(
        self, success, prompt, output_dir, index, total
    ):
        self.add_result_item(success, prompt, output_dir, index)
        self.status.setText(
            f"Batch progress • Job {index}/{total} finished"
        )

        if success:
            self.latest_output_dir = Path(output_dir)

            try:
                self.load_mesh(
                    self.latest_output_dir / "generated.obj"
                )
                self.obj_button.setEnabled(True)
                self.ply_button.setEnabled(True)
            except Exception as error:
                self.log.append(
                    f"Preview error for job {index}: {error}"
                )

    def batch_finished(self, stopped):
        self.set_running(False)

        if stopped:
            self.status.setText("Batch stopped.")
            self.log.append("Batch stopped.")
        else:
            self.status.setText("Batch complete!")
            self.log.append("All batch jobs have finished.")

    def generation_finished(self, success, message, output_dir):
        self.set_running(False)
        self.log.append(message)

        if success:
            self.latest_output_dir = Path(output_dir)

            try:
                self.load_mesh(
                    self.latest_output_dir / "generated.obj"
                )
                self.obj_button.setEnabled(True)
                self.ply_button.setEnabled(True)
                self.status.setText("Generation complete!")
            except Exception as error:
                self.status.setText(
                    "Generated, but preview failed."
                )
                self.log.append(f"Preview error: {error}")
        else:
            self.status.setText("Generation failed.")
            QMessageBox.warning(
                self, "Generation failed", message
            )

    def load_mesh(self, path):
        mesh = trimesh.load(str(path), force="mesh")

        if mesh.is_empty:
            raise ValueError("The generated mesh is empty.")

        vertices = np.asarray(
            mesh.vertices, dtype=np.float32
        )
        faces = np.asarray(mesh.faces, dtype=np.int32)

        vertices -= vertices.mean(axis=0)
        extent = np.max(np.ptp(vertices, axis=0))

        if extent > 0:
            vertices *= 2.0 / extent

        if self.mesh_item is not None:
            self.view.removeItem(self.mesh_item)

        mesh_data = gl.MeshData(
            vertexes=vertices, faces=faces
        )

        self.mesh_item = gl.GLMeshItem(
            meshdata=mesh_data,
            smooth=False,
            drawFaces=True,
            drawEdges=False,
            shader="shaded",
            color=(0.72, 0.78, 0.92, 1.0)
        )

        self.view.addItem(self.mesh_item)
        self.view.setCameraPosition(
            distance=4, elevation=20, azimuth=45
        )

    def export(self, extension):
        source = (
            self.latest_output_dir / f"generated.{extension}"
        )

        if not source.exists():
            QMessageBox.warning(
                self, "File missing",
                f"Could not find {source}"
            )
            return

        destination, _ = QFileDialog.getSaveFileName(
            self,
            f"Export {extension.upper()}",
            f"my_model.{extension}",
            f"{extension.upper()} files (*.{extension})"
        )

        if destination:
            shutil.copy2(source, destination)
            self.log.append(f"Exported: {destination}")

    def open_output_folder(self):
        BATCH_OUTPUT.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(BATCH_OUTPUT))
        )

    def closeEvent(self, event):
        workers = [self.worker, self.batch_worker]

        if any(
            worker is not None and worker.isRunning()
            for worker in workers
        ):
            QMessageBox.warning(
                self, "Generation in progress",
                "Wait for generation to finish before closing ShapeForge."
            )
            event.ignore()
            return

        ENGINE_PROC.shutdown()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ShapeForge()
    window.show()
    sys.exit(app.exec())
