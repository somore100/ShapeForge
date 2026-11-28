"""Beginner-friendly UI helpers: hover help for every control, a maximized
startup window, and a thumbnail strip that only shows when it has content."""

import textwrap

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QLabel, QWidget

DEFAULT_HELP = "Hover over any control to see what it does."

# attribute name on the window -> plain-language explanation
HELP = {
    "generate_button": "Creates one 3D model from the prompt above. Takes about 20 seconds on a good GPU.",
    "batch_button": "Creates every prompt in the batch box, one after another, each in its own folder.",
    "stop_button": "Stops the batch after the model that is currently being made.",
    "enh_single_button": "Rewrites the prompt above so Shap-E understands it better. Ctrl+Z undoes it.",
    "enh_batch_button": "Rewrites every prompt line in the batch box. Comments, {a|b} variants and xN are left alone.",
    "enh_style": "Adds a style hint (low-poly, cartoon...) to enhanced prompts. None adds nothing.",
    "enh_model": "Rules = instant built-in cleanup, no AI needed. Other entries are AI models running on your PC via Ollama.",
    "enh_action": "Set up AI: install Ollama. Get model: download a small AI model. Refresh: look again.",
    "clean_fragments": "Shap-E often leaves small floating bits around a model. This deletes loose pieces smaller than the % on the right.",
    "clean_fragment_size": "How small a loose piece must be, compared with the main body, to be deleted. 5% is a safe start.",
    "clean_holes": "Closes gaps in the surface so the model is solid (needed for 3D printing).",
    "clean_smooth": "Evens out the lumpy surface. More passes = smoother but softer details. 0 turns it off.",
    "clean_decimate": "Lowers the triangle count for smaller files and faster games. Low targets lose detail.",
    "clean_button": "Applies these cleanup settings to the model on screen. Your original is always kept.",
    "show_original_button": "Preview the model exactly as Shap-E made it.",
    "show_cleaned_button": "Preview the cleaned-up version.",
    "auto_clean": "Cleans every batch model automatically as it finishes, using these settings.",
    "color_button": "Show the model's own colors. Shap-E paints colors on the surface instead of using image textures.",
    "obj_button": "Export as OBJ. Works everywhere; colors are stored per point.",
    "ply_button": "Export as PLY. The best choice for keeping colors (the Blender add-on uses it).",
    "stl_button": "Export for 3D printing. STL cannot store colors.",
    "glb_button": "Export for game engines and Blender. Faces up (Y-up) as those tools expect.",
    "export_orientation": "Which way is up in the exported file. Auto is right for most uses.",
    "results": "Finished batch models. Click one to preview it.",
}
LABEL_HELP = {"Smooth:": "clean_smooth", "Decimate:": "clean_decimate"}
MODE_HELP = {
    "Smooth": "Smooth shading: hides the polygon edges.",
    "Flat": "Flat shading: shows each polygon, useful to judge real detail.",
    "Wireframe": "Only the polygon edges.",
    "Solid + Edges": "Solid surface with the polygon edges drawn on top.",
}
TOOLTIP_CSS = (
    "QToolTip { background: #3a4256; color: #ffffff; border: 1px solid #5590ff;"
    " padding: 6px; font-size: 13px; }"
)


class _HelpFilter(QObject):
    def __init__(self, label):
        super().__init__(label)
        self.label = label

    def eventFilter(self, obj, event):
        kind = event.type()
        if kind == QEvent.Enter:
            text = obj.property("help_text")
            if text:
                self.label.setText(text)
        elif kind == QEvent.Leave:
            self.label.setText(DEFAULT_HELP)
        return False


def _attach(widget, text, flt):
    if widget is None or not text:
        return
    if not widget.isEnabled() and widget.toolTip() and "pip install" in widget.toolTip():
        text_tip = widget.toolTip()  # keep the install hint on a disabled Decimate
    else:
        text_tip = textwrap.fill(text, 58)
    widget.setToolTip(text_tip)
    widget.setProperty("help_text", text)
    widget.installEventFilter(flt)


def install_help(window):
    """Give every known control a tooltip and a hover line under the model."""
    window.setStyleSheet(window.styleSheet() + TOOLTIP_CSS)
    label = QLabel(DEFAULT_HELP)
    label.setStyleSheet("color: #8892a6; font-size: 12px;")
    label.setFixedHeight(18)
    anchor = getattr(window, "results", None)
    layout = anchor.parentWidget().layout() if anchor is not None else None
    if layout is not None:
        layout.insertWidget(layout.indexOf(anchor), label)
    window.help_label = label
    # The three text boxes each demand 90 px, which forces a tall window.
    for name in ("prompt", "batch_prompts", "log"):
        box = getattr(window, name, None)
        if box is not None:
            box.setMinimumHeight(52)
    _setup_results_strip(window)
    flt = _HelpFilter(label)
    window._help_filter = flt

    for name, text in HELP.items():
        _attach(getattr(window, name, None), text, flt)
    for mode, button in getattr(window, "mode_buttons", {}).items():
        _attach(button, MODE_HELP.get(mode), flt)
    for widget in window.findChildren(QLabel):
        target = LABEL_HELP.get(widget.text())
        if target:
            _attach(widget, HELP[target], flt)
    for button in window.findChildren(QWidget):
        if getattr(button, "text", None) and button.text() == "Reset Camera":
            _attach(button, "Puts the camera back to the default view.", flt)


def _setup_results_strip(window):
    """Hide the empty thumbnail strip until a batch produces results."""
    results = getattr(window, "results", None)
    if results is None:
        return
    def sync(*_):
        try:
            results.setVisible(results.count() > 0)
        except RuntimeError:  # window already closing
            pass
    model = results.model()
    for signal in (model.rowsInserted, model.rowsRemoved, model.modelReset):
        signal.connect(sync)
    sync()


def fit_window(window, avail=None):
    """Always open maximized: the window manager knows the real screen size,
    which Qt can misreport on scaled or multi-monitor displays."""
    window.showMaximized()
    return "maximized"
