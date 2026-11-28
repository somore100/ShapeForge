"""Prompt enhancement for ShapeForge: offline polish + optional local Ollama rewrite.

Pure helpers (rule-based polish, split/join batch lines, model filtering) have no
Qt dependency. EnhanceTask (QThread) wraps the Ollama calls for the UI.
Only the standard library is used for HTTP, with proxies bypassed.
"""

import json
import os
import re
import urllib.error
import urllib.request
from urllib.parse import urlparse

from PySide6.QtCore import QThread, Signal

from engine.rules import RULES_LABEL, STYLES, rule_enhance  # noqa: F401

DEFAULT_MODEL = "llama3.1:8b"
MAX_MODEL_BYTES = int(6.5e9)
KEEP_ALIVE = "30s"

SYSTEM_PROMPT = (
    "You rewrite short descriptions into prompts for Shap-E, a text-to-3D "
    "model. Shap-E works best with one simple, concrete object. Rules: keep "
    "the user's subject and every detail they gave; do not invent colours, "
    "brands, accessories or styling they did not ask for; if the object has "
    "parts that make it recognisable (wheels, legs, wings, handle), mention "
    "them briefly; describe one object only, with no scene, background, "
    "lighting or people; at most 12 words; one line. Work any requested "
    "style in naturally. Reply with the prompt only."
)

_REJECT_TOKENS = {
    "embed", "embedding", "coder", "code", "vl", "vision", "llava", "r1",
    "reasoning", "qwq", "qwen3", "rerank", "moondream", "minicpm",
}
_PREFERRED = ("llama3.1", "llama3.2", "llama3", "qwen2.5", "mistral",
              "gemma", "phi")
_QUOTES = "\"'`\u201c\u201d\u2018\u2019"
_LIST_MARKER = re.compile(r"^(\s*(?:\d{1,3}[.)\]:]|[-*\u2022\u2013])\s+)")
_REPEAT = re.compile(r"^(.*\S)(\s+[x\u00d7]\d{1,2})$")


class OllamaError(Exception):
    pass


# ---------------------------------------------------------------- polish --

def basic_polish(text, style="None"):
    """Offline, rule-based cleanup (see engine/rules.py)."""
    return rule_enhance(text, style)


def clean_reply(reply, fallback):
    reply = re.sub(r"<think>.*?</think>", "", reply or "", flags=re.S)
    for line in reply.splitlines():
        line = line.strip().strip(_QUOTES).strip()
        line = re.sub(r"^(prompt|output)\s*:\s*", "", line, flags=re.I)
        line = line.strip().strip(_QUOTES).strip()
        if line:
            return line[:240]
    return fallback


# ------------------------------------------------------- batch line split --

def split_batch_lines(text):
    """-> list of (prefix, body, suffix). body is None for lines left alone."""
    entries = []
    for line in text.splitlines():
        stripped = line.strip()
        skip = (
            not stripped or stripped.startswith("#") or line[:1].isspace()
            or any(ch in line for ch in "{};") or stripped.endswith("\\")
        )
        if skip:
            entries.append((line, None, ""))
            continue
        prefix = ""
        match = _LIST_MARKER.match(line)
        body = line
        if match:
            prefix = match.group(1)
            body = line[match.end():]
        suffix = ""
        repeat = _REPEAT.match(body.rstrip())
        if repeat:
            body, suffix = repeat.group(1), repeat.group(2)
        entries.append((prefix, body.strip(), suffix))
    return entries


def join_batch_lines(entries, new_bodies):
    """Rebuild text; new_bodies are consumed in order for non-skipped lines."""
    new_bodies = iter(new_bodies)
    lines = []
    for prefix, body, suffix in entries:
        lines.append(prefix if body is None else prefix + next(new_bodies) + suffix)
    return "\n".join(lines)


# ------------------------------------------------------------ model picks --

def usable_models(tags):
    """Filter /api/tags entries: plain chat models up to MAX_MODEL_BYTES."""
    names = []
    for entry in tags:
        name = entry.get("name") or entry.get("model") or ""
        size = int(entry.get("size") or 0)
        tokens = set(re.split(r"[:\-_/.\s]+", name.lower()))
        if not name or tokens & _REJECT_TOKENS:
            continue
        if size and size > MAX_MODEL_BYTES:
            continue
        names.append(name)

    def rank(name):
        for i, key in enumerate(_PREFERRED):
            if name.lower().startswith(key):
                return (i, name)
        return (len(_PREFERRED), name)

    return sorted(names, key=rank)


# --------------------------------------------------------------- Ollama --

def host_url():
    raw = os.environ.get("OLLAMA_HOST", "").strip() or "127.0.0.1:11434"
    if "://" not in raw:
        raw = "http://" + raw
    parts = urlparse(raw)
    host = parts.hostname or "127.0.0.1"
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    return f"{parts.scheme}://{host}:{parts.port or 11434}"


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _call(host, path, payload=None, timeout=10, stream=False):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        host + path, data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        response = _OPENER.open(request, timeout=timeout)
        if stream:
            return response
        with response:
            return json.loads(response.read().decode() or "{}")
    except urllib.error.HTTPError as error:
        raise OllamaError(f"Ollama returned HTTP {error.code}") from error
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise OllamaError(f"Cannot reach Ollama: {error}") from error


def probe(host):
    """-> {'state': no_server|no_model|ready, 'models': [...], 'error': str}"""
    try:
        tags = _call(host, "/api/tags", timeout=4).get("models", [])
    except OllamaError as error:
        return {"state": "no_server", "models": [], "error": str(error)}
    models = usable_models(tags)
    return {"state": "ready" if models else "no_model",
            "models": models, "error": ""}


def rewrite(host, model, text, style="None"):
    hint = STYLES.get(style, "")
    user = f"Object: {text}" + (f"\nStyle: {hint}" if hint else "")
    reply = _call(host, "/api/chat", {
        "model": model, "stream": False, "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0.4, "num_predict": 96},
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": user}],
    }, timeout=180)
    content = (reply.get("message") or {}).get("content", "")
    return clean_reply(content, basic_polish(text, style))


def unload(host, model):
    try:
        _call(host, "/api/generate", {"model": model, "keep_alive": 0}, timeout=2)
    except OllamaError:
        pass


def pull(host, model, progress):
    response = _call(host, "/api/pull", {"model": model, "stream": True},
                     timeout=60, stream=True)
    with response:
        for raw in response:
            try:
                event = json.loads(raw.decode())
            except ValueError:
                continue
            if event.get("error"):
                raise OllamaError(event["error"])
            total, done = event.get("total"), event.get("completed")
            if total and done is not None:
                progress(f"Downloading {model} - {int(done * 100 / total)}%")
            elif event.get("status"):
                progress(f"{model}: {event['status']}")


# ------------------------------------------------------------------ task --

class EnhanceTask(QThread):
    progress = Signal(str)
    finished_task = Signal(str, bool, object)  # kind, ok, result-or-message

    def __init__(self, kind, host, model="", items=None, style="None"):
        super().__init__()
        self.kind, self.host, self.model = kind, host, model
        self.items, self.style = list(items or []), style

    def run(self):
        try:
            if self.kind == "probe":
                self.finished_task.emit("probe", True, probe(self.host))
            elif self.kind == "pull":
                pull(self.host, self.model, self.progress.emit)
                self.finished_task.emit("pull", True, self.model)
            else:
                self.finished_task.emit("rewrite", True, self._rewrite())
        except OllamaError as error:
            self.finished_task.emit(self.kind, False, str(error))
        except Exception as error:  # never let a thread die silently
            self.finished_task.emit(self.kind, False, f"{type(error).__name__}: {error}")

    def _rewrite(self):
        out, succeeded = [], False
        for index, text in enumerate(self.items, start=1):
            if self.isInterruptionRequested():
                raise OllamaError("Cancelled")
            self.progress.emit(f"Enhancing {index}/{len(self.items)}...")
            try:
                out.append(rewrite(self.host, self.model, text, self.style))
                succeeded = True
            except OllamaError:
                if not succeeded:
                    raise
                out.append(basic_polish(text, self.style))
        return out
