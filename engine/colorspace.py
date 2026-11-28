"""Colour-space fixes for Shap-E meshes.

Shap-E stores mesh vertex colours as LINEAR values, but OBJ and PLY viewers
(and Blender's importer) expect sRGB, so raw files look dark and muddy.
New generations are converted to sRGB when written; GLB (glTF) wants linear
vertex colours, so exports to GLB convert back.
"""

import numpy as np


def linear_to_srgb(values):
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return np.where(values <= 0.0031308, 12.92 * values,
                    1.055 * np.power(values, 1 / 2.4) - 0.055)


def srgb_to_linear(values):
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return np.where(values <= 0.04045, values / 12.92,
                    np.power((values + 0.055) / 1.055, 2.4))


def mesh_colors_to_srgb(tri_mesh):
    """In place, on a shap_e TriMesh (vertex_channels R, G, B in 0..1)."""
    channels = getattr(tri_mesh, "vertex_channels", None)
    if not channels or not all(name in channels for name in "RGB"):
        return False
    for name in "RGB":
        channels[name] = linear_to_srgb(channels[name]).astype(np.float32)
    return True


def trimesh_colors_srgb_to_linear(mesh):
    """In place, on a trimesh mesh with uint8 vertex colours (for GLB)."""
    try:
        if mesh.visual.kind != "vertex":
            return False
        colors = np.array(mesh.visual.vertex_colors, dtype=np.uint8)
    except Exception:
        return False
    rgb = srgb_to_linear(colors[:, :3] / 255.0)
    colors[:, :3] = np.round(rgb * 255.0).astype(np.uint8)
    mesh.visual.vertex_colors = colors
    return True
