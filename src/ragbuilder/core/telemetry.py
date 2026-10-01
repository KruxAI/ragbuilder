"""Best-effort usage events without provider credentials or user content."""

import atexit
from contextlib import contextmanager
import os
from pathlib import Path
import queue
import re
import threading
import time
import uuid
from urllib.parse import urlsplit

from dotenv import load_dotenv
from platformdirs import user_data_dir
import requests

from ragbuilder import __version__

# Set to the deployed public collector URL before releasing this patch.
DEFAULT_TELEMETRY_ENDPOINT = ""
MODULES = {"data_ingest", "retriever", "generation", "ragbuilder", "eval_data_generation", "ui"}
EVENTS = {"installation_started", "run_started", "run_completed", "run_failed", "error"}


class _UsageSpan:
    def set_attribute(self, key, value):
        # Existing callers attach scores. Those values are deliberately not sent.
        pass


class RAGBuilderTelemetry:
    def __init__(self):
        load_dotenv(Path.cwd() / ".env", override=False)
        self.enabled = os.getenv("ENABLE_ANALYTICS", "true").strip().lower() in {"true", "1", "yes"}
        self.endpoint = os.getenv("RAGBUILDER_TELEMETRY_URL", DEFAULT_TELEMETRY_ENDPOINT).strip()
        self.user_id = None
        self._queue = queue.Queue(maxsize=32)
        self._worker = None
        self._lock = threading.Lock()
        self._closed = False
        parsed = urlsplit(self.endpoint)
        if not self.enabled or parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            self.enabled = False
            return
        try:
            self.user_id = self._get_or_create_user_id()
            self._emit("installation_started", "ragbuilder")
            atexit.register(self.shutdown)
        except Exception:
            self.enabled = False

    def _get_or_create_user_id(self):
        path = Path(user_data_dir(appname="ragbuilder")) / "uuid"
        try:
            identifier = path.read_text().strip()
            if re.fullmatch(r"a-[0-9a-f]{32}", identifier):
                return identifier
        except OSError:
            pass
        identifier = "a-" + uuid.uuid4().hex
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(identifier)
        except OSError:
            pass
        return identifier

    def _emit(self, event, module):
        if not self.enabled or self._closed or event not in EVENTS or module not in MODULES:
            return
        payload = {"event": event, "module": module, "installation_id": self.user_id, "version": __version__[:80]}
        try:
            with self._lock:
                if self._worker is None or not self._worker.is_alive():
                    self._worker = threading.Thread(target=self._drain, daemon=True, name="ragbuilder-usage")
                    self._worker.start()
            self._queue.put_nowait(payload)
        except Exception:
            pass

    def _drain(self):
        while True:
            payload = self._queue.get()
            try:
                with requests.Session() as session:
                    session.post(self.endpoint, json=payload, timeout=(1, 2), allow_redirects=False)
            except Exception:
                pass
            finally:
                self._queue.task_done()

    @contextmanager
    def optimization_span(self, module, config):
        self._emit("run_started", module)
        try:
            yield _UsageSpan() if self.enabled else None
        except Exception:
            self._emit("run_failed", module)
            raise
        else:
            self._emit("run_completed", module)

    @contextmanager
    def eval_datagen_span(self, **attributes):
        with self.optimization_span("eval_data_generation", {}) as span:
            yield span

    def update_optimization_results(self, span, results, module):
        pass

    def track_error(self, module, error, context):
        self._emit("error", module)

    def track_ui_run(self):
        self._emit("run_started", "ui")

    def flush(self):
        # Sending is asynchronous; do not delay each optimization stage.
        pass

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        deadline = time.monotonic() + 1
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)


telemetry = RAGBuilderTelemetry()
