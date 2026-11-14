"""Batch Build helpers: smart prompt parsing and result thumbnails."""

import re

import numpy as np

MAX_PROMPTS_PARSED = 500

BATCH_HELP = (
    "One prompt per line, or paste a numbered / bulleted list.\n"
    "Or separate prompts with ;  on one line.\n"
    "Variants: a {wooden|iron} sword   (makes 2 prompts)\n"
    "Repeats: a low-poly tree x3   (makes 3 variations)\n"
    "# starts a comment. Indent a line to continue the one above."
)

_MARKER = re.compile(r"^\s*(?:\d{1,3}[.)\]:]|[-*\u2022\u2013])\s+")
_BRACES = re.compile(r"\{([^{}]*\|[^{}]*)\}")
_REPEAT = re.compile(r"^(.*\S)\s+[x\u00d7](\d{1,2})$")
_QUOTES = "\"'`\u201c\u201d\u2018\u2019"


def _clean(text):
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) >= 2 and text[0] in _QUOTES and text[-1] in _QUOTES:
        text = text[1:-1].strip()
    return text


def _join_lines(lines):
    """Raw lines -> raw prompts, handling lists and continuation lines."""
    marked = any(_MARKER.match(line) for line in lines if line.strip())
    prompts = []
    current = None
    seen_marker = False
    carry = False
    start_indent = 0

    for line in lines:
        text = line.strip()
        if not text:
            continue

        if marked:
            if _MARKER.match(line):
                seen_marker = True
                if current:
                    prompts.append(current)
                current = _MARKER.sub("", line, count=1).strip()
            elif not seen_marker:
                if not text.endswith(":"):
                    prompts.append(text)
            else:
                current = f"{current} {text}"
            continue

        indent = len(line) - len(line.lstrip())
        if current is not None and (carry or indent > start_indent):
            current = f"{current} {text}"
        else:
            if current is not None:
                prompts.append(current)
            current = text
            start_indent = indent
        carry = current.endswith("\\")
        if carry:
            current = current[:-1].rstrip()

    if current:
        prompts.append(current)
    return prompts


def _expand(prompt):
    """a {red|blue} cube  ->  ['a red cube', 'a blue cube']"""
    results = [prompt]
    changed = True
    while changed:
        changed = False
        expanded = []
        for item in results:
            match = _BRACES.search(item)
            if not match:
                expanded.append(item)
                continue
            changed = True
            for option in match.group(1).split("|"):
                expanded.append(
                    item[:match.start()] + option.strip() + item[match.end():]
                )
        results = expanded
        if len(results) > MAX_PROMPTS_PARSED:
            return results[:MAX_PROMPTS_PARSED]
    return results


def parse_prompts(text):
    lines = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = raw.strip()
        if stripped.startswith("#") or stripped.startswith("//"):
            continue
        lines.append(raw.rstrip())

    results = []
    for chunk in _join_lines(lines):
        for part in chunk.split(";"):
            part = _clean(part)
            if not part:
                continue
            count = 1
            match = _REPEAT.match(part)
            if match:
                part, count = match.group(1), int(match.group(2))
            for variant in _expand(part):
                variant = _clean(variant)
                if variant:
                    results.extend([variant] * count)
            if len(results) > MAX_PROMPTS_PARSED:
                return results[:MAX_PROMPTS_PARSED]
    return results


# ---------------------------------------------------------------- thumbnails

def _camera(azimuth=45.0, elevation=20.0):
    az, el = np.radians(azimuth), np.radians(elevation)
    eye = np.array([
        np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)
    ])
    right = np.cross([0.0, 0.0, 1.0], eye)
    right /= np.linalg.norm(right)
    up = np.cross(eye, right)
    return eye, right, up


def render_thumbnail(path, size=96, scale=2):
    """Software-render a small shaded preview of a mesh as a QImage."""
    import trimesh
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage

    mesh = trimesh.load(str(path), force="mesh")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    if len(verts) == 0 or len(faces) == 0:
        raise ValueError("empty mesh")

    eye, right, up = _camera()
    vx, vy = verts @ right, verts @ up
    res = size * scale
    k = 0.9 * res / max(np.ptp(vx), np.ptp(vy), 1e-9)
    cx, cy = (vx.min() + vx.max()) / 2, (vy.min() + vy.max()) / 2

    # Sample each triangle densely enough to cover its pixels.
    edges = np.concatenate([
        faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]
    ])
    edge_px = np.hypot(vx[edges[:, 0]] - vx[edges[:, 1]],
                       vy[edges[:, 0]] - vy[edges[:, 1]]) * k
    steps = int(np.clip(np.ceil(np.median(edge_px) * 1.6), 1, 10))
    while steps > 1 and len(faces) * (steps + 1) * (steps + 2) // 2 > 2_500_000:
        steps -= 1
    weights = np.array([
        (i / steps, j / steps, 1 - (i + j) / steps)
        for i in range(steps + 1) for j in range(steps + 1 - i)
    ])
    tri = verts[faces]
    points = np.einsum("pk,fkd->fpd", weights, tri).reshape(-1, 3)
    normals = np.repeat(np.asarray(mesh.face_normals), len(weights), axis=0)

    x, y, depth = points @ right, points @ up, points @ eye
    px = np.clip(np.round((x - cx) * k + res / 2), 0, res - 1).astype(int)
    py = np.clip(np.round(res / 2 - (y - cy) * k), 0, res - 1).astype(int)
    idx = py * res + px

    zbuf = np.full(res * res, -np.inf)
    np.maximum.at(zbuf, idx, depth)
    keep = depth >= zbuf[idx] - 1e-9

    light = eye + 0.5 * up - 0.3 * right
    light /= np.linalg.norm(light)
    shade = 0.30 + 0.70 * np.abs(normals @ light)

    base = np.array([0.72, 0.78, 0.92])
    canvas = np.zeros((res * res, 3))
    filled = np.zeros(res * res, dtype=bool)
    canvas[idx[keep]] = base * shade[keep, None]
    filled[idx[keep]] = True
    canvas = canvas.reshape(res, res, 3)
    filled = filled.reshape(res, res)

    # Fill small holes: empty pixels with most neighbours filled.
    pad_c = np.pad(canvas, ((1, 1), (1, 1), (0, 0)))
    pad_f = np.pad(filled, 1).astype(float)
    shifts = [(dy, dx) for dy in (0, 1, 2) for dx in (0, 1, 2)
              if (dy, dx) != (1, 1)]
    count = sum(pad_f[dy:dy + res, dx:dx + res] for dy, dx in shifts)
    total = sum(
        pad_c[dy:dy + res, dx:dx + res] * pad_f[dy:dy + res, dx:dx + res, None]
        for dy, dx in shifts
    )
    holes = (~filled) & (count >= 5)
    canvas[holes] = total[holes] / count[holes, None]
    filled |= holes

    background = np.array([0x15, 0x18, 0x20]) / 255.0
    canvas[~filled] = background
    rgb = np.ascontiguousarray((canvas * 255).astype(np.uint8))

    image = QImage(rgb.data, res, res, res * 3, QImage.Format_RGB888).copy()
    return image.scaled(
        size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )


def placeholder_image(size=96, text="\u2717", color="#4a2830"):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QFont, QImage, QPainter

    image = QImage(size, size, QImage.Format_RGB32)
    image.fill(QColor(color))
    painter = QPainter(image)
    painter.setPen(QColor("#ff8a8a"))
    font = QFont()
    font.setPixelSize(size // 2)
    painter.setFont(font)
    painter.drawText(image.rect(), Qt.AlignCenter, text)
    painter.end()
    return image
