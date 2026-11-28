"""Vertex-colour preview. Shap-E has no image textures: it stores one RGB
colour per vertex, which the OBJ/PLY files already contain."""

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtWidgets import QPushButton

TOGGLE_STYLE = (
    "QPushButton { background: #2b3140; padding: 6px 10px; }"
    "QPushButton:hover { background: #3a4256; }"
    "QPushButton:checked { background: #3478f6; }"
    "QPushButton:disabled { background: #252a36; color: #6b7388; }"
)


def vertex_colors_of(mesh):
    """Float RGBA (N, 4) in 0..1 aligned with mesh.vertices, or None."""
    try:
        if mesh.visual.kind != "vertex":
            return None
        colors = np.asarray(mesh.visual.vertex_colors, dtype=np.float32)
    except Exception:
        return None
    if colors.ndim != 2 or len(colors) != len(mesh.vertices):
        return None
    if colors.shape[1] == 3:
        colors = np.hstack([colors, np.full((len(colors), 1), 255, np.float32)])
    return np.ascontiguousarray(colors[:, :4] / 255.0, dtype=np.float32)


def make_meshdata(win):
    colors = win.mesh_colors if getattr(win, "use_colors", False) else None
    if colors is not None and len(colors) == len(win.mesh_vertices):
        return gl.MeshData(vertexes=win.mesh_vertices, faces=win.mesh_faces,
                           vertexColors=colors)
    return gl.MeshData(vertexes=win.mesh_vertices, faces=win.mesh_faces)


def set_mesh_colors(win, mesh):
    win.mesh_colors = vertex_colors_of(mesh)
    has = win.mesh_colors is not None
    win.color_button.setEnabled(has)
    win.color_button.setChecked(has and win.use_colors)
    if not has:
        win.color_button.setToolTip("This model has no colors.")


def add_color_toggle(win, row):
    win.use_colors = True
    win.mesh_colors = None
    win.color_button = QPushButton("Colors")
    win.color_button.setCheckable(True)
    win.color_button.setChecked(True)
    win.color_button.setEnabled(False)
    win.color_button.setStyleSheet(TOGGLE_STYLE)

    def toggled(checked):
        win.use_colors = bool(checked)
        win.show_mesh()

    win.color_button.clicked.connect(toggled)
    row.addWidget(win.color_button)
