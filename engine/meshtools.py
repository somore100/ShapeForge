"""Mesh cleanup and export helpers (CPU only)."""

from pathlib import Path

import numpy as np
import trimesh

CLEAN_OBJ = "generated_clean.obj"
CLEAN_PLY = "generated_clean.ply"
FRAGMENT_FRACTION = 0.05

DECIMATE_CHOICES = (
    ("Off", None),
    ("20,000 faces", 20000),
    ("10,000 faces", 10000),
    ("5,000 faces", 5000),
    ("2,000 faces", 2000),
)
EXPORT_FORMATS = ("obj", "ply", "stl", "glb")

# Rotate -90 degrees about X: Z-up -> Y-up, (x, y, z) -> (x, z, -y).
Z_UP_TO_Y_UP = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])


def decimation_available():
    try:
        import fast_simplification  # noqa: F401
    except ImportError:
        return False
    return True


def best_mesh(folder):
    """The cleaned mesh if one exists in this folder, else the original."""
    folder = Path(folder)
    clean = folder / CLEAN_OBJ
    return clean if clean.exists() else folder / "generated.obj"


def purge_clean_files(folder):
    for name in (CLEAN_OBJ, CLEAN_PLY):
        try:
            (Path(folder) / name).unlink()
        except FileNotFoundError:
            pass


def _vertex_colors(mesh):
    """Per-vertex RGBA colors, or None if the mesh has none."""
    try:
        if mesh.visual.kind == "vertex":
            return np.array(mesh.visual.vertex_colors)
    except Exception:
        pass
    return None


def _restore_colors(mesh, old_vertices, old_colors):
    """Decimation drops colors; copy them from the nearest original vertex."""
    try:
        from scipy.spatial import cKDTree
        _, nearest = cKDTree(old_vertices).query(mesh.vertices)
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh, vertex_colors=old_colors[nearest]
        )
    except Exception:
        pass
    return mesh


def _stats(mesh):
    return {
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "closed": bool(mesh.is_watertight),
    }


def cleanup(folder, remove_fragments=True, fill_holes=True, smooth=10,
            decimate=None, fragment_fraction=FRAGMENT_FRACTION):
    """Clean <folder>/generated.obj into generated_clean.obj/.ply.

    The original is never modified. Returns a report dict.
    """
    folder = Path(folder)
    source = folder / "generated.obj"
    if not source.exists():
        raise FileNotFoundError(f"Could not find {source}")

    mesh = trimesh.load(str(source), force="mesh")
    if mesh.is_empty:
        raise ValueError("The mesh is empty.")

    before = _stats(mesh)
    notes = []
    removed = 0

    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()

    if remove_fragments:
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            biggest = max(len(part.faces) for part in parts)
            kept = [p for p in parts
                    if len(p.faces) >= fragment_fraction * biggest]
            removed = len(parts) - len(kept)
            if removed:
                mesh = trimesh.util.concatenate(kept)

    if fill_holes and not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)

    if smooth and smooth > 0:
        trimesh.smoothing.filter_taubin(mesh, iterations=int(smooth))

    if decimate:
        if not decimation_available():
            notes.append("decimation skipped (fast-simplification missing)")
        elif decimate >= len(mesh.faces):
            notes.append("already below the decimation target")
        else:
            was_closed = mesh.is_watertight
            old_vertices = np.array(mesh.vertices)
            old_colors = _vertex_colors(mesh)
            mesh = mesh.simplify_quadric_decimation(face_count=int(decimate))
            if old_colors is not None:
                mesh = _restore_colors(mesh, old_vertices, old_colors)
            if was_closed and not mesh.is_watertight:
                notes.append("decimation opened holes; try more faces")

    mesh.remove_unreferenced_vertices()
    try:
        if mesh.is_watertight:
            trimesh.repair.fix_inversion(mesh)
    except Exception:
        pass

    mesh.export(str(folder / CLEAN_OBJ))
    mesh.export(str(folder / CLEAN_PLY))

    return {
        "before": before, "after": _stats(mesh),
        "fragments_removed": removed, "notes": notes,
    }


def summarize(report):
    before, after = report["before"], report["after"]
    parts = [f"{before['faces']:,} -> {after['faces']:,} faces"]
    if report["fragments_removed"]:
        count = report["fragments_removed"]
        parts.append(f"removed {count} fragment{'s' if count != 1 else ''}")
    parts.append("closed surface" if after["closed"] else "open surface")
    parts.extend(report["notes"])
    return ", ".join(parts)


def export_mesh(source, destination, orientation="auto"):
    """Export any supported format. orientation: auto | keep | yup."""
    destination = Path(destination)
    extension = destination.suffix.lower().lstrip(".")
    if extension not in EXPORT_FORMATS:
        raise ValueError(f"Unsupported format: .{extension}")

    mesh = trimesh.load(str(source), force="mesh")
    if orientation == "yup" or (orientation == "auto" and extension == "glb"):
        mesh.apply_transform(Z_UP_TO_Y_UP)
    mesh.export(str(destination))
