"""Qt side of prompt enhancement: buttons, pickers and the Ollama setup flow.

ShapeForge mixes this in and calls build_enhance_ui(layout), enhance_generating(
running) and shutdown_enhance(). Enhance and Generate are never active together.
"""

from PySide6.QtCore import QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QMessageBox, QPushButton,
)

from engine import enhance as core

BUTTON_STYLE = (
    "QPushButton { background: #2b3140; padding: 6px 10px; }"
    "QPushButton:hover { background: #3a4256; }"
    "QPushButton:disabled { background: #252a36; color: #6b7388; }"
)
COMBO_STYLE = (
    "QComboBox { background: #2b3140; border: 1px solid #414858;"
    " border-radius: 6px; padding: 4px 8px; color: #e8eaf0; }"
    "QComboBox QAbstractItemView { background: #151820; color: #e8eaf0;"
    " selection-background-color: #3478f6; }"
)

INSTALL_HELP = (
    "Prompt rewriting uses Ollama, which runs locally.\n\n"
    "1. Install it:  curl -fsSL https://ollama.com/install.sh | sh\n"
    "2. Make sure it is running:  ollama serve\n"
    "3. Click Refresh here.\n\n"
    "Enhance still works without it (basic cleanup + style hint)."
)


def _replace_text(edit, text):
    """Replace the whole text but keep Ctrl+Z working."""
    cursor = edit.textCursor()
    cursor.beginEditBlock()
    cursor.select(QTextCursor.Document)
    cursor.insertText(text)
    cursor.endEditBlock()
    edit.setTextCursor(cursor)


class EnhanceMixin:
    def build_enhance_ui(self, layout):
        self._enh_task = None
        self._enh_state = "unknown"
        self._enh_busy = False
        self._enh_gen = False
        self._enh_host = core.host_url()

        layout.addWidget(QLabel("Prompt enhancement"))

        row = QHBoxLayout()
        self.enh_single_button = QPushButton("Enhance Prompt")
        self.enh_batch_button = QPushButton("Enhance Batch")
        for button in (self.enh_single_button, self.enh_batch_button):
            button.setStyleSheet(BUTTON_STYLE)
            row.addWidget(button)
        self.enh_single_button.clicked.connect(self.enhance_single)
        self.enh_batch_button.clicked.connect(self.enhance_batch)
        layout.addLayout(row)

        pickers = QHBoxLayout()
        self.enh_style = QComboBox()
        self.enh_style.addItems(list(core.STYLES))
        self.enh_style.setToolTip("Style added to enhanced prompts")
        self.enh_model = QComboBox()
        self.enh_model.setToolTip("Local Ollama model used for rewriting")
        for combo in (self.enh_style, self.enh_model):
            combo.setStyleSheet(COMBO_STYLE)
            pickers.addWidget(combo, 1)
        layout.addLayout(pickers)

        status_row = QHBoxLayout()
        self.enh_status = QLabel("Checking for Ollama...")
        self.enh_status.setStyleSheet("color: #8892a6;")
        self.enh_status.setWordWrap(True)
        self.enh_action = QPushButton("Refresh")
        self.enh_action.setStyleSheet(BUTTON_STYLE)
        self.enh_action.clicked.connect(self.enhance_action)
        status_row.addWidget(self.enh_status, 1)
        status_row.addWidget(self.enh_action)
        layout.addLayout(status_row)

        self._refresh_enhance_controls()
        QTimer.singleShot(400, self.enhance_probe)

    # -------------------------------------------------------------- state --

    def _set_enh_status(self, text, color="#8892a6"):
        self.enh_status.setText(text)
        self.enh_status.setStyleSheet(f"color: {color};")

    def _refresh_enhance_controls(self):
        idle = not self._enh_busy and not self._enh_gen
        self.enh_single_button.setEnabled(idle)
        self.enh_batch_button.setEnabled(idle)
        self.enh_action.setEnabled(not self._enh_busy)
        self.enh_style.setEnabled(idle)
        self.enh_model.setEnabled(idle and self._enh_state == "ready")
        self.enh_action.setText({
            "no_server": "Set up AI", "no_model": "Get model",
        }.get(self._enh_state, "Refresh"))

    def enhance_generating(self, running):
        self._enh_gen = bool(running)
        self._refresh_enhance_controls()

    def _lock_generation(self, locked):
        """Rewrites use the GPU/CPU; keep Generate off while one runs."""
        if locked:
            self.generate_button.setEnabled(False)
            self.batch_button.setEnabled(False)
        elif not self._enh_gen:
            self.generate_button.setEnabled(True)
            self.batch_button.setEnabled(True)

    # -------------------------------------------------------------- tasks --

    def _start_enh_task(self, kind, **kwargs):
        if self._enh_task is not None and self._enh_task.isRunning():
            return False
        self._enh_busy = True
        self._lock_generation(kind == "rewrite")
        self._refresh_enhance_controls()
        task = core.EnhanceTask(kind, self._enh_host, **kwargs)
        task.progress.connect(lambda text: self._set_enh_status(text, "#aab4c8"))
        task.finished_task.connect(self._enh_task_done)
        self._enh_task = task
        task.start()
        return True

    def _enh_task_done(self, kind, ok, result):
        task, self._enh_task = self._enh_task, None
        if task is not None:
            task.wait()
        self._enh_busy = False
        self._lock_generation(False)
        if kind == "probe":
            self._apply_probe(result if ok else
                              {"state": "no_server", "models": [], "error": str(result)})
        elif kind == "pull":
            if ok:
                self.log.append(f"Model ready: {result}")
            else:
                self.log.append(f"Model download failed: {result}")
            self.enhance_probe()
        else:
            self._apply_rewrite(ok, result)
        self._refresh_enhance_controls()

    def enhance_probe(self):
        self._set_enh_status("Checking for Ollama...")
        self._start_enh_task("probe")

    def _apply_probe(self, info):
        self._enh_state = info["state"]
        current = self.enh_model.currentText()
        self.enh_model.clear()
        if info["state"] == "ready":
            self.enh_model.addItems(info["models"])
            pick = current if current in info["models"] else info["models"][0]
            self.enh_model.setCurrentText(pick)
            self._set_enh_status(f"AI ready: {pick}", "#6ee7a0")
        elif info["state"] == "no_model":
            self._set_enh_status(
                f"Ollama is running but has no small chat model. "
                f"Get {core.DEFAULT_MODEL} (about 5 GB).", "#f0c674")
        else:
            self._set_enh_status(
                "Ollama not found - basic cleanup only.", "#f0c674")

    def enhance_action(self):
        if self._enh_state == "no_server":
            QMessageBox.information(self, "Set up AI", INSTALL_HELP)
            self.enhance_probe()
        elif self._enh_state == "no_model":
            answer = QMessageBox.question(
                self, "Get model",
                f"Download {core.DEFAULT_MODEL} with Ollama? "
                "It is about 5 GB and runs locally.")
            if answer == QMessageBox.Yes:
                self._set_enh_status(f"Starting download of {core.DEFAULT_MODEL}...",
                                     "#aab4c8")
                self._start_enh_task("pull", model=core.DEFAULT_MODEL)
        else:
            self.enhance_probe()

    # ------------------------------------------------------------ enhance --

    def _enhance_targets(self, edit, batch):
        text = edit.toPlainText()
        if batch:
            entries = core.split_batch_lines(text)
            bodies = [body for _, body, _ in entries if body is not None]
            return entries, bodies
        body = text.strip()
        return None, ([body] if body else [])

    def enhance_single(self):
        self._run_enhance(self.prompt, batch=False)

    def enhance_batch(self):
        self._run_enhance(self.batch_prompts, batch=True)

    def _run_enhance(self, edit, batch):
        entries, bodies = self._enhance_targets(edit, batch)
        if not bodies:
            QMessageBox.warning(self, "Nothing to enhance",
                                "Type a prompt first." if not batch else
                                "No enhanceable lines found in the batch box.")
            return
        style = self.enh_style.currentText()
        self._enh_pending = (edit, entries, bodies, style)
        if self._enh_state == "ready" and self.enh_model.currentText():
            self._set_enh_status("Enhancing...", "#aab4c8")
            self._start_enh_task("rewrite", model=self.enh_model.currentText(),
                                 items=bodies, style=style)
        else:
            self._finish_enhance([core.basic_polish(b, style) for b in bodies],
                                 "basic cleanup (no AI model available)")

    def _apply_rewrite(self, ok, result):
        edit, entries, bodies, style = self._enh_pending
        if ok:
            self._finish_enhance(result, f"AI ({self.enh_model.currentText()})")
        else:
            self.log.append(f"Enhance failed: {result}")
            self._finish_enhance([core.basic_polish(b, style) for b in bodies],
                                 "basic cleanup (AI request failed)")
            self.enhance_probe()

    def _finish_enhance(self, new_bodies, how):
        edit, entries, bodies, style = self._enh_pending
        if entries is None:
            _replace_text(edit, new_bodies[0])
        else:
            _replace_text(edit, core.join_batch_lines(entries, new_bodies))
        self.log.append(f"Enhanced {len(new_bodies)} prompt(s) with {how}. "
                        "Ctrl+Z in the box undoes it.")
        if self._enh_state == "ready":
            self._set_enh_status(f"AI ready: {self.enh_model.currentText()}", "#6ee7a0")

    # ----------------------------------------------------------- shutdown --

    def shutdown_enhance(self):
        task = getattr(self, "_enh_task", None)
        if task is not None and task.isRunning():
            task.requestInterruption()
            task.wait(1500)
        if getattr(self, "_enh_state", "") == "ready":
            model = self.enh_model.currentText()
            if model:
                core.unload(self._enh_host, model)
