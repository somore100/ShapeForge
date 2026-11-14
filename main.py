import sys
import subprocess
import shutil
from pathlib import Path

import trimesh
import numpy as np
import pyqtgraph.opengl as gl

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QLineEdit, QFileDialog, QMessageBox, QSplitter
)

APP_DIR = Path(__file__).resolve().parent
SHAPE_E_HOME = Path(
    os.environ.get("SHAPE_E_HOME", str(Path.home() / "shap-e"))
).expanduser().resolve()

ENGINE = APP_DIR / "engine" / "generate.py"
OUTPUT = APP_DIR / "generated"


class Generator(QThread):
    log = Signal(str)
    finished_generation = Signal(bool, str)

    def __init__(self, prompt):
        super().__init__()
        self.prompt = prompt

    def run(self):
        try:
            command = [
                sys.executable,
                str(ENGINE),
                "--prompt", self.prompt,
                "--output", str(OUTPUT)
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
                self.log.emit(line.rstrip())

            process.wait()

            if process.returncode == 0:
                self.finished_generation.emit(
                    True, "Generation completed!"
                )
            else:
                self.finished_generation.emit(
                    False, f"Generator exited with code {process.returncode}"
                )

        except Exception as error:
            self.finished_generation.emit(False, str(error))


class ShapeForge(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ShapeForge - Local Text-to-3D")
        self.resize(1200, 760)

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #20232b;
                color: #e8eaf0;
                font-size: 14px;
            }
            QLineEdit, QTextEdit {
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

        left_layout.addWidget(QLabel("Describe your 3D model"))

        self.prompt = QTextEdit()
        self.prompt.setPlaceholderText(
            "Example: A cute low-poly penguin with two eyes..."
        )
        self.prompt.setMaximumHeight(145)
        left_layout.addWidget(self.prompt)

        self.generate_button = QPushButton("Generate 3D Model")
        self.generate_button.clicked.connect(self.generate)
        left_layout.addWidget(self.generate_button)

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
        self.view.setCameraPosition(distance=4, elevation=20, azimuth=45)

        grid = gl.GLGridItem()
        grid.setSize(4, 4)
        grid.setSpacing(0.25, 0.25)
        self.view.addItem(grid)

        right_layout.addWidget(self.view, 1)

        export_row = QHBoxLayout()

        self.obj_button = QPushButton("Export OBJ")
        self.obj_button.clicked.connect(lambda: self.export("obj"))
        self.obj_button.setEnabled(False)

        self.ply_button = QPushButton("Export PLY")
        self.ply_button.clicked.connect(lambda: self.export("ply"))
        self.ply_button.setEnabled(False)

        self.folder_button = QPushButton("Open Output Folder")
        self.folder_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(OUTPUT)))
        )

        export_row.addWidget(self.obj_button)
        export_row.addWidget(self.ply_button)
        export_row.addWidget(self.folder_button)
        right_layout.addLayout(export_row)

        splitter.addWidget(right)
        splitter.setSizes([360, 840])

        if not ENGINE.exists():
            self.log.append(f"WARNING: Generator not found: {ENGINE}")

    def generate(self):
        prompt = self.prompt.toPlainText().strip()

        if not prompt:
            QMessageBox.warning(
                self, "Missing prompt", "Please describe an object first."
            )
            return

        if not ENGINE.exists():
            QMessageBox.critical(
                self, "Missing engine",
                f"Could not find Shap-E generator:\n{ENGINE}"
            )
            return

        self.generate_button.setEnabled(False)
        self.obj_button.setEnabled(False)
        self.ply_button.setEnabled(False)

        self.log.clear()
        self.log.append("Starting ShapeForge...")
        self.status.setText("Generating on RTX 3060...")

        self.worker = Generator(prompt)
        self.worker.log.connect(self.log.append)
        self.worker.finished_generation.connect(self.generation_finished)
        self.worker.start()

    def generation_finished(self, success, message):
        self.generate_button.setEnabled(True)
        self.log.append(message)

        if success:
            try:
                self.load_mesh(OUTPUT / "generated.obj")
                self.obj_button.setEnabled(True)
                self.ply_button.setEnabled(True)
                self.status.setText("Generation complete!")
            except Exception as error:
                self.status.setText("Generated, but preview failed.")
                self.log.append(f"Preview error: {error}")
        else:
            self.status.setText("Generation failed.")
            QMessageBox.warning(self, "Generation failed", message)

    def load_mesh(self, path):
        mesh = trimesh.load(str(path), force="mesh")

        if mesh.is_empty:
            raise ValueError("The generated mesh is empty.")

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)

        # Center and normalize the mesh for preview
        vertices -= vertices.mean(axis=0)
        extent = np.max(np.ptp(vertices, axis=0))
        if extent > 0:
            vertices *= 2.0 / extent

        if self.mesh_item is not None:
            self.view.removeItem(self.mesh_item)

        mesh_data = gl.MeshData(vertexes=vertices, faces=faces)

        self.mesh_item = gl.GLMeshItem(
            meshdata=mesh_data,
            smooth=False,
            drawFaces=True,
            drawEdges=False,
            shader="shaded",
            color=(0.72, 0.78, 0.92, 1.0)
        )

        self.view.addItem(self.mesh_item)
        self.view.setCameraPosition(distance=4, elevation=20, azimuth=45)

    def export(self, extension):
        source = OUTPUT / f"generated.{extension}"

        if not source.exists():
            QMessageBox.warning(
                self, "File missing", f"Could not find {source}"
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

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(
                self, "Generation in progress",
                "Wait for generation to finish before closing ShapeForge."
            )
            event.ignore()
            return
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ShapeForge()
    window.show()
    sys.exit(app.exec())
