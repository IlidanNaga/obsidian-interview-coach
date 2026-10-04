#!/usr/bin/env python3
"""Local browser bridge: one worker, private snapshots, ephemeral answers.

CoachApp(state_dir, endpoint, model) exposes config/sessions/session/job/start/action
with dictionary requests and responses. make_server(config, app=None) (also
create_server(config)) returns {ok, server, url}; call server.serve_forever(),
then server.server_close(). No controller work happens on GET or startup.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import stat
import threading
import urllib.parse
import uuid

import interview

MAX_BODY_BYTES = 64 * 1024
MAX_JOBS = 128
MAX_REQUEST_IDS = 256
PENDING_PHASES = {"need_question", "indexing", "discovery", "expansion"}
ACTIONS = {"answer", "skip", "next", "pause", "finish", "resume", "retry"}


def _error(code):
    return {"ok": False, "error": code}


def _id(value):
    return isinstance(value, str) and len(value) == 32 and all(c in "0123456789abcdef" for c in value)


class CoachApp:
    def __init__(self, state_dir=interview.DEFAULT_STATE_DIR, endpoint="http://127.0.0.1:11434", model="qwen3.5:9b"):
        self.endpoint = interview._endpoint(endpoint)
        if not isinstance(model, str) or not model.strip() or len(model.encode("utf-8")) > 256:
            raise interview.SessionError("invalid_model")
        self.model = model
        self.state_dir = interview._state_dir(state_dir, allow_missing=True)
        self.token = secrets.token_hex(32)
        self.lock = threading.RLock()
        self.jobs = OrderedDict()
        self.requests = OrderedDict()
        self.active_job_id = None
        self.worker = None
        self.closed = False
        self.owner_fd = None
        # Lock a sibling so startup and read-only listing need not create state_dir.
        try:
            self.state_dir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor = os.open(
                self.state_dir.with_name(self.state_dir.name + ".web.lock"),
                os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK,
                0o600,
            )
            self.owner_fd = descriptor
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_nlink != 1
                or info.st_mode & 0o077
            ):
                raise interview.SessionError("invalid_state_dir")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise interview.SessionError("web_busy") from None
        except (OSError, interview.SessionError) as error:
            if self.owner_fd is not None:
                os.close(self.owner_fd)
                self.owner_fd = None
            if isinstance(error, interview.SessionError):
                raise
            raise interview.SessionError("state_unavailable") from None

    def _active(self):
        return copy.deepcopy(self.jobs.get(self.active_job_id))

    def config(self, request: dict) -> dict:
        with self.lock:
            return {
                "ok": True,
                "token": self.token,
                "endpoint": self.endpoint,
                "model": self.model,
                "active_job": self._active(),
            }

    def sessions(self, request: dict) -> dict:
        return interview.list_sessions({"state_dir": self.state_dir})

    def session(self, request: dict) -> dict:
        if not isinstance(request, dict) or set(request) != {"session_id"}:
            return _error("invalid_request")
        with self.lock:
            result = interview.read_session({**request, "state_dir": self.state_dir})
            result["active_job"] = self._active()
            return result

    def job(self, request: dict) -> dict:
        if not isinstance(request, dict) or set(request) != {"job_id"} or not _id(request["job_id"]):
            return _error("invalid_request")
        with self.lock:
            job = self.jobs.get(request["job_id"])
            return {"ok": True, "job": copy.deepcopy(job)} if job else _error("job_not_found")

    def start(self, request: dict) -> dict:
        return self._submit("start", request)

    def action(self, request: dict) -> dict:
        return self._submit("action", request)

    def _validate_request(self, kind, request):
        if not isinstance(request, dict):
            return "invalid_request"
        if not _id(request.get("request_id")):
            return "invalid_request_id"
        if kind == "start":
            required = {"request_id", "vault", "endpoint", "model"}
            if not required <= set(request) or set(request) - required not in ({"note"}, {"topic"}):
                return "invalid_request"
            if any(not isinstance(value, str) for value in request.values()):
                return "invalid_request"
        else:
            required = {"request_id", "session_id", "action"}
            if not required <= set(request) or set(request) - required - {"answer"}:
                return "invalid_request"
            if not _id(request["session_id"]):
                return "invalid_session_id"
            if not isinstance(request["action"], str) or request["action"] not in ACTIONS:
                return "invalid_action"
            if request["action"] == "answer":
                if not isinstance(request.get("answer"), str) or not request["answer"].strip():
                    return "invalid_answer"
            elif "answer" in request:
                return "invalid_request"
        return None

    def _remember(self, request_id, fingerprint, job_id):
        self.requests[request_id] = (fingerprint, job_id)
        while len(self.requests) > MAX_REQUEST_IDS:
            self.requests.popitem(last=False)
        while len(self.jobs) > MAX_JOBS:
            oldest = next(iter(self.jobs))
            if oldest == self.active_job_id:
                break
            self.jobs.pop(oldest)
            self.requests = OrderedDict((key, value) for key, value in self.requests.items() if value[1] != oldest)

    def _submit(self, kind, request):
        error = self._validate_request(kind, request)
        if error:
            return _error(error)
        try:
            canonical = json.dumps(
                [kind, request], sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
            fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        except (TypeError, ValueError, UnicodeError):
            return _error("invalid_request")
        # One lock linearizes acceptance, deduplication, busy and stop intents.
        # Controller/model work runs outside it, leaving polling and stop available.
        with self.lock:
            if self.closed:
                return _error("server_closed")
            known = self.requests.get(request["request_id"])
            if known:
                if known[0] != fingerprint:
                    return _error("request_id_conflict")
                return {"ok": True, "job": copy.deepcopy(self.jobs[known[1]])}
            active = self.jobs.get(self.active_job_id)
            if active:
                if (
                    kind == "action"
                    and request["session_id"] == active["session_id"]
                    and request["action"] in {"pause", "finish"}
                ):
                    if active["stop_requested"] != "finish":
                        active["stop_requested"] = request["action"]
                    self._remember(request["request_id"], fingerprint, active["job_id"])
                    return {"ok": True, "job": copy.deepcopy(active)}
                return {**_error("job_busy"), "job": copy.deepcopy(active)}
            if kind == "start":
                prepared = interview.start_session({**request, "state_dir": self.state_dir, "defer_inference": True})
                if "phase" not in prepared:
                    return prepared
                session_id = prepared["session_id"]
                operation = None
            else:
                current = interview.read_session({"session_id": request["session_id"], "state_dir": self.state_dir})
                if "meta" not in current:
                    return current
                session_id = request["session_id"]
                operation = {key: value for key, value in request.items() if key != "request_id"}
            job = {
                "job_id": uuid.uuid4().hex,
                "session_id": session_id,
                "status": "running",
                "stop_requested": None,
                "result": None,
                "error": None,
            }
            self.active_job_id = job["job_id"]
            self.jobs[job["job_id"]] = job
            self._remember(request["request_id"], fingerprint, job["job_id"])
            self.worker = threading.Thread(target=self._run, args=(job["job_id"], operation), daemon=True)
            response = {"ok": True, "job": copy.deepcopy(job)}
            self.worker.start()
            return response

    def _stopping(self, job_id):
        with self.lock:
            return self.jobs[job_id]["stop_requested"] is not None

    def _run(self, job_id, operation):
        session_id = self.jobs[job_id]["session_id"]
        base = {"session_id": session_id, "state_dir": self.state_dir}
        stopped = lambda: self._stopping(job_id)
        try:
            result = (
                interview.dispatch({**operation, "state_dir": self.state_dir}, _stop_requested=stopped)
                if operation is not None and not stopped()
                else interview.read_session(base)
            )
            # Explicit actions run once. Only active preparation progresses automatically.
            while (
                result.get("ok")
                and result.get("status") == "active"
                and result.get("phase") in PENDING_PHASES
                and not stopped()
            ):
                result = interview.dispatch({**base, "action": "retry"}, _stop_requested=stopped)
        except KeyboardInterrupt:
            result = _error("interrupted")
        except Exception:
            # Never retain an exception/traceback containing a request or private path.
            result = _error("internal_error")
        finally:
            operation = None
        with self.lock:
            job = self.jobs[job_id]
            if job["stop_requested"]:
                try:
                    result = interview.dispatch({**base, "action": job["stop_requested"]})
                except Exception:
                    result = _error("internal_error")
            job.update(status="done", result=result, error=result.get("error"))
            self.active_job_id = None

    def close(self):
        with self.lock:
            self.closed = True
            active = self.jobs.get(self.active_job_id)
            if active and active["stop_requested"] is None:
                active["stop_requested"] = "pause"
            worker = self.worker
        if worker is not None:
            worker.join()
        with self.lock:
            if self.owner_fd is not None:
                os.close(self.owner_fd)
                self.owner_fd = None


def _status(result):
    if result.get("ok"):
        return 202 if result.get("job", {}).get("status") == "running" else 200
    code = result.get("error")
    if code in {"session_not_found", "job_not_found"}:
        return 404
    if code in {"job_busy", "session_busy", "request_id_conflict", "web_busy"}:
        return 409
    if code in {"corrupt_state", "state_unavailable", "persistence_failure", "internal_error", "server_closed"}:
        return 503
    return 400


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_error(self, code, message=None, explain=None):
        self._json(code, _error("http_error"))

    def _send(self, code, data, content_type):
        self.close_connection = True
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
        )
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _json(self, code, value):
        self._send(code, interview._json(value).encode("utf-8"), "application/json; charset=utf-8")

    def _protected(self, post=False):
        if self.headers.get_all("Host", []) != [self.server.host_header]:
            self._json(403, _error("invalid_host"))
            return False
        origins = self.headers.get_all("Origin", [])
        if (post or origins) and origins != [self.server.url]:
            self._json(403, _error("invalid_origin"))
            return False
        if post:
            tokens = self.headers.get_all("X-Coach-Token", [])
            if (
                len(tokens) != 1
                or not tokens[0].isascii()
                or not secrets.compare_digest(tokens[0], self.server.app.token)
            ):
                self._json(403, _error("invalid_token"))
                return False
        return True

    def _route(self):
        if not self.path.startswith("/") or self.path.startswith("//"):
            raise ValueError
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.fragment:
            raise ValueError
        pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
        query = dict(pairs)
        if len(pairs) != len(query):
            raise ValueError
        return parsed.path, query

    def do_GET(self):
        if not self._protected():
            return
        try:
            route, query = self._route()
            app = self.server.app
            if route in {"/", "/web.html"} and not query:
                try:
                    data = Path(__file__).with_name("web.html").read_bytes()
                except OSError:
                    self._json(503, _error("ui_unavailable"))
                    return
                self._send(200, data, "text/html; charset=utf-8")
                return
            if route == "/api/config" and not query:
                result = app.config({})
            elif route == "/api/sessions" and not query:
                result = app.sessions({})
            elif route == "/api/session" and set(query) == {"session_id"}:
                result = app.session(query)
                self._json(200 if "meta" in result else _status(result), result)
                return
            elif route == "/api/job" and set(query) == {"job_id"}:
                result = app.job(query)
            else:
                self._json(404, _error("route_not_found"))
                return
            self._json(_status(result), result)
        except (ValueError, UnicodeError):
            self._json(400, _error("invalid_request"))

    def do_POST(self):
        if not self._protected(post=True):
            return
        try:
            route, query = self._route()
            if route not in {"/api/start", "/api/action"} or query:
                self._json(404, _error("route_not_found"))
                return
            content_types = self.headers.get_all("Content-Type", [])
            if len(content_types) != 1 or content_types[0].lower() not in {
                "application/json",
                "application/json; charset=utf-8",
            }:
                self._json(415, _error("json_required"))
                return
            lengths = self.headers.get_all("Content-Length", [])
            if (
                self.headers.get_all("Transfer-Encoding")
                or len(lengths) != 1
                or not lengths[0].isascii()
                or not lengths[0].isdigit()
            ):
                self._json(400, _error("invalid_content_length"))
                return
            length = int(lengths[0])
            if length > MAX_BODY_BYTES:
                self._json(413, _error("request_too_large"))
                return
            self.connection.settimeout(5)
            data = self.rfile.read(length)
            if len(data) != length:
                raise ValueError
            request = json.loads(
                data.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_invalid_constant
            )
            result = self.server.app.start(request) if route == "/api/start" else self.server.app.action(request)
            self._json(_status(result), result)
        except TimeoutError:
            self._json(408, _error("request_timeout"))
        except (ValueError, UnicodeError, RecursionError):
            self._json(400, _error("invalid_request"))

    def do_OPTIONS(self):
        if self._protected():
            self._json(405, _error("method_not_allowed"))


def _unique_object(pairs):
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("duplicate JSON key")
    return value


def _invalid_constant(value):
    raise ValueError("non-finite JSON number")


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        pass

    def server_close(self):
        super().server_close()
        self.app.close()


def make_server(request: dict, app=None) -> dict:
    """Bind only 127.0.0.1. Port 0 is allowed for ephemeral test servers."""
    created = False
    try:
        if not isinstance(request, dict) or set(request) - {"port", "state_dir", "endpoint", "model"}:
            return _error("invalid_request")
        port = request.get("port", 8765)
        if type(port) is not int or not 0 <= port <= 65535:
            return _error("invalid_port")
        if app is None:
            app = CoachApp(
                request.get("state_dir", interview.DEFAULT_STATE_DIR),
                request.get("endpoint", "http://127.0.0.1:11434"),
                request.get("model", "qwen3.5:9b"),
            )
            created = True
        server = _Server(("127.0.0.1", port), _Handler, bind_and_activate=False)
        server.app = app
        try:
            server.server_bind()
            server.server_activate()
        except OSError:
            # A failed bind must not keep the state-owner lock.
            server.socket.close()
            raise
        server.host_header = f"127.0.0.1:{server.server_port}"
        server.url = "http://" + server.host_header
        return {"ok": True, "server": server, "url": server.url}
    except interview.SessionError as error:
        return _error(error.code)
    except (OSError, RuntimeError, ValueError, TypeError):
        if created:
            app.close()
        return _error("server_unavailable")


def create_server(request: dict) -> dict:
    return make_server(request)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--state-dir", default=str(interview.DEFAULT_STATE_DIR))
    result = create_server(vars(parser.parse_args()))
    if not result["ok"]:
        print("Error:", result["error"])
        return 1
    server = result["server"]
    print(result["url"], flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
