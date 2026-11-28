"""Client for the persistent ShapeForge engine (engine/server.py).

Keeps one server process alive so Shap-E loads once, not once per job.
run() returns an exit-code style int: 0 ok, 1 job failed, 2 engine could not
start (caller may fall back to the one-shot generator), 3 engine died.
"""

import json
import os
import re
import subprocess
import sys
import threading

from engine.runtime import engine_python

_PERCENT = re.compile(r"^\s*(\d+)%\|")


class PersistentEngine:
    def __init__(self, server_path, shap_e_home):
        self.server_path = str(server_path)
        self.home = str(shap_e_home)
        self.proc = None
        self.lock = threading.Lock()
        self._log = lambda text: None
        self._last_percent = -1
        self._job_id = 0
        self.options = {}  # quality settings merged into every job

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def _forward_stderr(self, proc):
        for line in proc.stderr:
            line = line.rstrip()
            if not line or "FutureWarning" in line:
                continue
            if line.lstrip().startswith("def "):
                continue
            match = _PERCENT.match(line)
            if match:
                percent = int(match.group(1))
                if percent == self._last_percent or percent % 25:
                    continue
                self._last_percent = percent
            self._log(line)

    def _read_event(self):
        for raw in self.proc.stdout:
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except ValueError:
                self._log(raw)
                continue
            if event.get("event") == "log":
                self._log(event.get("message", ""))
                continue
            return event
        return None

    def _start(self):
        env = os.environ.copy()
        prior = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = self.home + (os.pathsep + prior if prior else "")
        env["SHAPE_E_HOME"] = self.home
        self.proc = subprocess.Popen(
            [engine_python(), self.server_path],
            cwd=self.home, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        threading.Thread(
            target=self._forward_stderr, args=(self.proc,), daemon=True
        ).start()

        while True:
            event = self._read_event()
            if event is None:
                self.proc = None
                return False, "Engine exited during startup."
            kind = event.get("event")
            if kind == "ready":
                return True, ""
            if kind == "fatal":
                self.shutdown()
                return False, event.get("error", "unknown error")

    def run(self, prompt, output_dir, log_fn):
        with self.lock:
            self._log = log_fn
            self._last_percent = -1

            if not self.alive():
                self.proc = None
                log_fn("Starting engine (loading Shap-E, one time)...")
                ok, error = self._start()
                if not ok:
                    log_fn(f"Engine failed to start: {error}")
                    return 2

            self._job_id += 1
            job_id = self._job_id
            job = {"id": job_id, "prompt": prompt, "output": str(output_dir)}
            job.update(self.options)
            try:
                self.proc.stdin.write(json.dumps(job) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self.proc = None
                log_fn("Engine connection lost; it will restart next job.")
                return 3

            while True:
                event = self._read_event()
                if event is None:
                    self.proc = None
                    log_fn("Engine stopped unexpectedly; it will restart next job.")
                    return 3
                if event.get("id") != job_id:
                    continue
                if event.get("event") == "done":
                    if event.get("ok"):
                        log_fn(f"Generated in {event.get('seconds')}s")
                        return 0
                    log_fn(f"Generation failed: {event.get('error')}")
                    return 1

    def shutdown(self):
        proc, self.proc = self.proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.stdin.write('{"cmd": "quit"}\n')
            proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
