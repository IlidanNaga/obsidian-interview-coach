"""Synthetic threaded controller and real loopback HTTP checks; no model service."""

import http.client
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import uuid

import interview
import web


class WebTests(unittest.TestCase):
    def setUp(self):
        # Private temporary fixture state, following existing tests' external-state pattern.
        self.root = Path(tempfile.mkdtemp())
        self.vault = self.root / "vault"
        self.vault.mkdir()
        self.note = self.vault / "cache.md"
        self.note.write_text(
            "# Cache\nA cache stores reusable results.\n# Removal\nInvalidation removes stale entries.\n"
        )
        self.state = self.root / "sessions"
        self.calls = []
        self.failure = None
        self.admit = True
        self.expansion = ["cache"]
        self.gate = None
        self.active_calls = self.peak_calls = 0
        self.call_lock = threading.Lock()
        self.mock = patch("interview.chat", side_effect=self.model)
        self.mock.start()
        self.app = web.CoachApp(self.state)
        created = web.make_server({"port": 0}, self.app)
        self.assertTrue(created["ok"], created)
        self.server, self.url = created["server"], created["url"]
        self.http_thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01})
        self.http_thread.start()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        if self.gate:
            self.gate[1].set()
        self.server.shutdown()
        self.http_thread.join(5)
        self.server.server_close()
        self.app.close()
        self.mock.stop()
        # Remove only this test's files, without recursive force removal.
        for directory, subdirs, files in os.walk(self.root, topdown=False):
            for name in files:
                (Path(directory) / name).unlink()
            for name in subdirs:
                child = Path(directory) / name
                if child.is_symlink():
                    child.unlink()
                else:
                    child.rmdir()
        self.root.rmdir()

    def model(self, request):
        with self.call_lock:
            self.calls.append(request)
            self.active_calls += 1
            self.peak_calls = max(self.peak_calls, self.active_calls)
        try:
            self.assertLessEqual(interview._body_size(request), request["max_request_bytes"])
            if self.gate:
                self.gate[0].set()
                self.assertTrue(self.gate[1].wait(5), "model gate was not released")
            if self.failure:
                return {"ok": False, "error": self.failure}
            payload = json.loads(request["messages"][-1]["content"])
            if request["format"] == interview.EXPANSION_SCHEMA:
                content = {"terms": self.expansion}
            else:
                quote = next(
                    line for line in payload["source"].split("\n") if line.strip() and not line.startswith("#")
                )
                if request["format"] == interview.ADMISSION_SCHEMA:
                    content = {
                        "decision": "admit" if self.admit else "reject",
                        "question": "What does the note say?" if self.admit else "",
                        "quote": quote if self.admit else "",
                    }
                elif "answer" in payload:
                    content = {
                        "label": "supported",
                        "feedback": "This agrees with the supplied note, not an objective grade.",
                        "quote": quote,
                    }
                else:
                    content = {"question": "What does the note say?", "quote": quote}
            return {"ok": True, "content": content}
        finally:
            with self.call_lock:
                self.active_calls -= 1

    def start_request(self, **fields):
        return {
            "request_id": uuid.uuid4().hex,
            "vault": str(self.vault),
            "note": "cache.md",
            "endpoint": self.app.endpoint,
            "model": self.app.model,
            **fields,
        }

    def action_request(self, session_id, action, **fields):
        return {"request_id": uuid.uuid4().hex, "session_id": session_id, "action": action, **fields}

    def complete(self, response):
        self.assertTrue(response["ok"], response)
        worker = self.app.worker
        worker.join(5)
        self.assertFalse(worker.is_alive(), "worker stalled")
        result = self.app.job({"job_id": response["job"]["job_id"]})["job"]
        self.assertEqual(result["status"], "done")
        return result

    def start(self, **fields):
        return self.complete(self.app.start(self.start_request(**fields)))

    def action(self, session_id, action, **fields):
        return self.complete(self.app.action(self.action_request(session_id, action, **fields)))

    def http(self, method, path, payload=None, headers=None, raw=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        defaults = {}
        if method == "POST":
            defaults = {"Origin": self.url, "X-Coach-Token": self.app.token, "Content-Type": "application/json"}
        defaults.update(headers or {})
        body = raw if raw is not None else (json.dumps(payload).encode() if payload is not None else None)
        connection.request(method, path, body=body, headers=defaults)
        response = connection.getresponse()
        data = response.read()
        self.assertEqual(response.getheader("Cache-Control"), "no-store")
        self.assertIsNone(response.getheader("Access-Control-Allow-Origin"))
        status, response_headers = response.status, dict(response.getheaders())
        connection.close()
        return status, json.loads(data), response_headers

    def snapshot(self, session_id):
        return self.state / (session_id + ".json")

    def block_model(self):
        self.gate = (threading.Event(), threading.Event())

    def test_deferred_note_and_topic_do_no_inference_or_indexing(self):
        request = {**self.start_request(), "state_dir": self.state, "defer_inference": True}
        with patch("interview.build_index", side_effect=AssertionError("deferred indexing")):
            note = interview.start_session(request)
            topic_request = {key: value for key, value in request.items() if key != "note"}
            topic_request["topic"] = "cache"
            topic = interview.start_session(topic_request)
            repeated = interview.start_session(topic_request)
        self.assertEqual((note["phase"], topic["phase"]), ("need_question", "indexing"))
        self.assertEqual(topic["session_id"], repeated["session_id"])
        self.assertEqual(self.calls, [])
        self.assertFalse((self.state / (topic["session_id"] + ".search")).exists())
        self.assertEqual(json.loads(self.snapshot(note["session_id"]).read_text())["version"], 1)
        self.assertEqual(json.loads(self.snapshot(topic["session_id"]).read_text())["version"], 2)
        self.assertEqual(interview.start_session({**request, "defer_inference": "yes"})["error"], "invalid_request")

    def test_missing_list_and_http_reads_do_not_create_state(self):
        self.assertFalse(self.state.exists())
        with patch("interview.dispatch", side_effect=AssertionError("read dispatched")):
            self.assertEqual(
                interview.list_sessions({"state_dir": self.state}), {"ok": True, "sessions": [], "unreadable": 0}
            )
            self.assertEqual(self.http("GET", "/api/sessions")[1]["sessions"], [])
            config = self.http("GET", "/api/config")[1]
        self.assertEqual(config["token"], self.app.token)
        self.assertIsNone(config["active_job"])
        self.assertFalse(self.state.exists())
        self.assertEqual(self.calls, [])

    def test_polling_active_model_reads_atomic_checkpoint_without_lock_or_resume(self):
        self.block_model()
        response = self.app.start(self.start_request())
        session_id = response["job"]["session_id"]
        self.assertTrue(self.gate[0].wait(5))
        before = self.snapshot(session_id).read_bytes()
        names = set(self.state.iterdir())
        with patch("interview.dispatch", side_effect=AssertionError("poll dispatched")):
            for _ in range(3):
                status, view, _ = self.http("GET", "/api/session?session_id=" + session_id)
                self.assertEqual((status, view["phase"]), (200, "need_question"))
                self.assertEqual(view["active_job"]["job_id"], response["job"]["job_id"])
                self.assertEqual(
                    self.http("GET", "/api/job?job_id=" + response["job"]["job_id"])[1]["job"]["status"], "running"
                )
                self.assertEqual(len(self.http("GET", "/api/sessions")[1]["sessions"]), 1)
        self.assertEqual(self.snapshot(session_id).read_bytes(), before)
        self.assertEqual(set(self.state.iterdir()), names)
        self.assertEqual(len(self.calls), 1)
        self.gate[1].set()
        self.assertEqual(self.complete(response)["result"]["phase"], "await_answer")

    def test_readonly_meta_unreadable_count_and_source_independence(self):
        job = self.start()
        session_id = job["session_id"]
        snapshot = self.snapshot(session_id)
        before = snapshot.read_bytes(), snapshot.stat().st_mtime_ns, snapshot.stat().st_mode
        bad = self.state / ("f" * 32 + ".json")
        bad.write_text("broken private JSON")
        linked = self.state / ("e" * 32 + ".json")
        linked.symlink_to(snapshot)
        malformed = self.state / "bad-name.json"
        malformed.write_text("{}")
        with patch("interview.load_note", side_effect=AssertionError("read Vault")), patch(
            "interview.chat", side_effect=AssertionError("read inferred")
        ):
            view = interview.read_session({"state_dir": self.state, "session_id": session_id})
            listed = interview.list_sessions({"state_dir": self.state})
        self.assertEqual(listed["unreadable"], 3)
        self.assertEqual(listed["sessions"], [view["meta"]])
        self.assertEqual(view["meta"]["note"], "cache.md")
        self.assertIsNone(view["meta"]["topic"])
        self.assertEqual(
            set(view["meta"]),
            {"session_id", "vault", "topic", "note", "status", "phase", "updated_at", "model", "endpoint"},
        )
        self.assertEqual((snapshot.read_bytes(), snapshot.stat().st_mtime_ns, snapshot.stat().st_mode), before)
        snapshot.chmod(0o644)
        self.assertEqual(
            self.http("GET", "/api/session?session_id=" + session_id)[:2],
            (503, {"ok": False, "error": "corrupt_state", "active_job": None}),
        )
        self.assertEqual(snapshot.stat().st_mode & 0o777, 0o644)

    def test_http_host_origin_and_token_guards_reject_before_controller(self):
        request = self.start_request()
        for headers, error in (
            ({"Host": "localhost:" + str(self.server.server_port)}, "invalid_host"),
            ({"Host": "evil.example"}, "invalid_host"),
            ({"Origin": "https://evil.example"}, "invalid_origin"),
            ({"Origin": "null"}, "invalid_origin"),
            ({"Origin": self.url + "/"}, "invalid_origin"),
        ):
            for method, path, body in (
                ("GET", "/api/config", None),
                ("GET", "/api/sessions", None),
                ("POST", "/api/start", request),
            ):
                with self.subTest(method=method, headers=headers):
                    status, result, _ = self.http(method, path, body, headers)
                    self.assertEqual((status, result), (403, {"ok": False, "error": error}))
        for token in ("", "wrong", "caf\xe9"):
            status, result, _ = self.http("POST", "/api/start", request, {"X-Coach-Token": token})
            self.assertEqual((status, result["error"]), (403, "invalid_token"))
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        connection.request(
            "POST",
            "/api/start",
            body=json.dumps(request),
            headers={"Content-Type": "application/json", "X-Coach-Token": self.app.token},
        )
        response = connection.getresponse()
        self.assertEqual((response.status, json.loads(response.read())["error"]), (403, "invalid_origin"))
        connection.close()
        self.assertFalse(self.state.exists())
        self.assertEqual(self.calls, [])

    def test_http_json_only_body_bound_and_no_arbitrary_file_routes(self):
        for path in (
            "/interview.py",
            "/BIBLE.md",
            "/../.env",
            "/api/session?session_id=x&session_id=y",
            "/api/config?extra=1",
        ):
            status, result, _ = self.http("GET", path)
            self.assertIn(status, {400, 404})
            self.assertFalse(result["ok"])
        for raw in (b"not JSON", b"[]", b'{"request_id":"a","request_id":"b"}', b'{"value":NaN}', b"\xff"):
            self.assertEqual(self.http("POST", "/api/start", raw=raw)[0], 400)
        self.assertEqual(
            self.http("POST", "/api/start", self.start_request(), {"Content-Type": "text/plain"})[:2],
            (415, {"ok": False, "error": "json_required"}),
        )
        self.assertEqual(
            self.http("POST", "/api/start", raw=b"x" * (web.MAX_BODY_BYTES + 1))[:2],
            (413, {"ok": False, "error": "request_too_large"}),
        )
        self.assertEqual(self.http("OPTIONS", "/api/action")[0], 405)
        self.assertFalse(self.state.exists())
        self.assertEqual(self.calls, [])

    def test_http_rejects_duplicate_headers_and_transfer_encoding(self):
        for header, value, error in (
            ("Host", self.server.host_header, "invalid_host"),
            ("Origin", self.url, "invalid_origin"),
            ("X-Coach-Token", self.app.token, "invalid_token"),
            ("Content-Length", "2", "invalid_content_length"),
            ("Transfer-Encoding", "chunked", "invalid_content_length"),
        ):
            connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
            connection.putrequest("POST", "/api/start")
            for key, text in (
                ("Origin", self.url),
                ("X-Coach-Token", self.app.token),
                ("Content-Type", "application/json"),
                ("Content-Length", "2"),
                (header, value),
            ):
                connection.putheader(key, text)
            connection.endheaders(b"{}")
            response = connection.getresponse()
            self.assertEqual(json.loads(response.read())["error"], error)
            connection.close()
        self.assertFalse(self.state.exists())

    def test_start_validation_state_dir_and_defer_are_server_owned(self):
        for fields, code in (
            ({"state_dir": str(self.root)}, "invalid_request"),
            ({"defer_inference": False}, "invalid_request"),
            ({"topic": "cache"}, "invalid_request"),
            ({"endpoint": "http://localhost:11434"}, "invalid_endpoint"),
            ({"model": ""}, "invalid_model"),
            ({"note": "../outside.md"}, "invalid_source"),
            ({"request_id": "A" * 32}, "invalid_request_id"),
        ):
            with self.subTest(fields=fields):
                self.assertEqual(self.http("POST", "/api/start", self.start_request(**fields))[1]["error"], code)
        self.assertEqual(self.calls, [])
        self.assertFalse(self.state.exists())

    def test_http_start_returns_session_and_inflight_stop_job(self):
        self.block_model()
        status, response, _ = self.http("POST", "/api/start", self.start_request())
        self.assertEqual(status, 202)
        self.assertTrue(self.gate[0].wait(5))
        session_id = response["job"]["session_id"]
        self.assertTrue(self.snapshot(session_id).exists())
        status, paused, _ = self.http("POST", "/api/action", self.action_request(session_id, "pause"))
        self.assertEqual((status, paused["job"]["stop_requested"]), (202, "pause"))
        self.assertEqual(paused["job"]["job_id"], response["job"]["job_id"])
        self.gate[1].set()
        job = self.complete(response)
        self.assertEqual((job["result"]["status"], job["result"]["phase"]), ("paused", "await_answer"))
        self.assertEqual(self.http("GET", "/api/session?session_id=" + session_id)[1]["status"], "paused")
        self.assertEqual(
            self.http("GET", "/api/job?job_id=" + "f" * 32)[:2], (404, {"ok": False, "error": "job_not_found"})
        )

    def test_dedup_running_and_done_payload_order_and_conflict(self):
        self.block_model()
        request = self.start_request()
        response = self.app.start(request)
        self.assertTrue(self.gate[0].wait(5))
        duplicate = self.app.start(dict(reversed(list(request.items()))))
        self.assertEqual(duplicate, response)
        self.assertEqual(self.app.start({**request, "model": "different"})["error"], "request_id_conflict")
        self.gate[1].set()
        job = self.complete(response)
        self.assertEqual(self.app.start(request)["job"], job)
        answer = self.action_request(job["session_id"], "answer", answer="unique ephemeral response")
        answered = self.complete(self.app.action(answer))
        self.assertEqual(self.app.action(answer)["job"], answered)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(answered["result"]["phase"], "await_next")
        next_request = self.action_request(job["session_id"], "next")
        advanced = self.complete(self.app.action(next_request))
        self.assertEqual(self.app.action(next_request)["job"], advanced)
        self.assertEqual(advanced["result"]["cursor"], 1)
        self.assertEqual(len(self.calls), 3)

    def test_busy_has_active_handle_no_queue_and_calls_are_serial(self):
        first = self.start()
        second = self.start()
        self.block_model()
        response = self.app.action(self.action_request(first["session_id"], "answer", answer="ephemeral input"))
        self.assertTrue(self.gate[0].wait(5))
        busy_request = self.action_request(second["session_id"], "skip")
        for attempt in (
            lambda: self.app.action(busy_request),
            lambda: self.app.start(self.start_request()),
            lambda: self.app.action(self.action_request(second["session_id"], "finish")),
        ):
            busy = attempt()
            self.assertEqual((busy["error"], busy["job"]["job_id"]), ("job_busy", response["job"]["job_id"]))
        self.assertEqual(len(self.calls), 3)
        self.gate[1].set()
        self.complete(response)
        self.assertEqual(self.complete(self.app.action(busy_request))["result"]["cursor"], 1)
        self.assertEqual(self.peak_calls, 1)

    def test_simultaneous_duplicate_starts_launch_one_worker(self):
        self.block_model()
        request = self.start_request()
        barrier = threading.Barrier(6)
        responses = []

        def submit():
            barrier.wait(5)
            responses.append(self.app.start(request))

        clients = [threading.Thread(target=submit) for _ in range(6)]
        for client in clients:
            client.start()
        for client in clients:
            client.join(5)
            self.assertFalse(client.is_alive())
        self.assertEqual(len(responses), 6)
        self.assertTrue(all(response["ok"] for response in responses))
        self.assertEqual(len({response["job"]["job_id"] for response in responses}), 1)
        self.assertTrue(self.gate[0].wait(5))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.app.sessions({})["sessions"]), 1)
        self.gate[1].set()
        self.complete(responses[0])

    def test_stop_during_expansion_prevents_following_candidate(self):
        self.block_model()
        request = self.start_request()
        request.pop("note")
        request["topic"] = "absent literal topic"
        response = self.app.start(request)
        self.assertTrue(self.gate[0].wait(5))
        self.assertEqual(self.calls[0]["format"], interview.EXPANSION_SCHEMA)
        self.app.action(self.action_request(response["job"]["session_id"], "pause"))
        self.gate[1].set()
        done = self.complete(response)
        self.assertEqual((done["result"]["status"], done["result"]["phase"]), ("paused", "discovery"))
        self.assertEqual(len(self.calls), 1)
        resumed = self.action(done["session_id"], "resume")
        self.assertEqual(resumed["result"]["phase"], "await_answer")
        self.assertEqual(len(self.calls), 2)

    def test_pause_and_finish_stop_rejected_topic_candidate_and_finish_wins(self):
        self.admit = False
        self.block_model()
        request = self.start_request()
        request.pop("note")
        request["topic"] = "cache"
        response = self.app.start(request)
        self.assertTrue(self.gate[0].wait(5))
        session_id = response["job"]["session_id"]
        pause_request = self.action_request(session_id, "pause")
        paused = self.app.action(pause_request)
        self.assertEqual(paused["job"]["stop_requested"], "pause")
        self.assertEqual(self.app.action(pause_request), paused)
        finished = self.app.action(self.action_request(session_id, "finish"))
        self.assertEqual(finished["job"]["stop_requested"], "finish")
        self.assertEqual(self.app.action(self.action_request(session_id, "pause"))["job"]["stop_requested"], "finish")
        self.gate[1].set()
        done = self.complete(response)
        self.assertEqual((done["result"]["status"], done["result"]["cursor"]), ("finished", 1))
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(self.snapshot(session_id).exists())

    def test_stop_during_index_batch_prevents_first_candidate_search(self):
        started, release = threading.Event(), threading.Event()
        build = interview.build_index

        def held_index(request):
            result = build(request)
            started.set()
            self.assertTrue(release.wait(5))
            return result

        request = self.start_request()
        request.pop("note")
        request["topic"] = "cache"
        with patch("interview.build_index", side_effect=held_index), patch(
            "interview.search_topic", side_effect=AssertionError("search after stop")
        ):
            response = self.app.start(request)
            self.assertTrue(started.wait(5))
            self.app.action(self.action_request(response["job"]["session_id"], "pause"))
            release.set()
            done = self.complete(response)
        self.assertEqual(done["result"]["status"], "paused")
        self.assertEqual(self.calls, [])
        self.assertTrue(done["result"]["scan"]["scan_complete"])

    def test_stop_during_search_prevents_admission_and_preserves_candidate(self):
        started, release = threading.Event(), threading.Event()
        search = interview.search_topic

        def held_search(request):
            result = search(request)
            started.set()
            self.assertTrue(release.wait(5))
            return result

        request = self.start_request()
        request.pop("note")
        request["topic"] = "cache"
        with patch("interview.search_topic", side_effect=held_search):
            response = self.app.start(request)
            self.assertTrue(started.wait(5))
            self.app.action(self.action_request(response["job"]["session_id"], "pause"))
            release.set()
            done = self.complete(response)
        self.assertEqual((done["result"]["status"], done["result"]["cursor"]), ("paused", 0))
        self.assertEqual(self.calls, [])
        resumed = self.action(done["session_id"], "resume")
        self.assertEqual((resumed["result"]["phase"], resumed["result"]["cursor"]), ("await_answer", 1))
        self.assertEqual(len(self.calls), 1)

    def test_stop_applied_after_failed_call_and_after_unexpected_exception(self):
        for failure in ("ollama request timed out", RuntimeError("private exception body")):
            with self.subTest(failure=type(failure).__name__):
                self.block_model()
                if isinstance(failure, str):
                    self.failure = failure
                    response = self.app.start(self.start_request())
                else:

                    def exploding(request):
                        self.gate[0].set()
                        self.assertTrue(self.gate[1].wait(5))
                        raise failure

                    with patch("interview.chat", side_effect=exploding):
                        response = self.app.start(self.start_request())
                        self.assertTrue(self.gate[0].wait(5))
                        self.app.action(self.action_request(response["job"]["session_id"], "finish"))
                        self.gate[1].set()
                        done = self.complete(response)
                    self.assertEqual(done["result"]["status"], "finished")
                    self.assertNotIn("private exception", json.dumps(done))
                    continue
                self.assertTrue(self.gate[0].wait(5))
                self.app.action(self.action_request(response["job"]["session_id"], "pause"))
                self.gate[1].set()
                done = self.complete(response)
                self.assertEqual((done["result"]["status"], done["result"]["error"]), ("paused", "timeout"))
                self.assertEqual(self.action(done["session_id"], "resume")["result"]["status"], "blocked")

    def test_failed_assessment_retry_and_restart_do_not_replay_answer(self):
        current = self.start()
        session_id = current["session_id"]
        self.failure = "ollama request timed out"
        answer = "EPHEMERAL-PRIVATE-ANSWER-123"
        failed = self.action(session_id, "answer", answer=answer)
        self.assertEqual((failed["result"]["status"], failed["result"]["phase"]), ("blocked", "await_answer"))
        self.assertEqual(self.http("GET", "/api/session?session_id=" + session_id)[0], 200)
        self.assertNotIn(answer, self.snapshot(session_id).read_text())
        self.assertNotIn(answer, json.dumps(failed))
        self.assertNotIn(answer, repr(self.app.requests))
        old_token = self.app.token
        self.app.close()
        self.app = web.CoachApp(self.state)
        self.server.app = self.app
        self.assertNotEqual(self.app.token, old_token)
        self.assertEqual(self.app.job({"job_id": failed["job_id"]})["error"], "job_not_found")
        before = len(self.calls)
        self.assertEqual(self.app.session({"session_id": session_id})["question"], current["result"]["question"])
        self.failure = None
        retried = self.action(session_id, "retry")
        self.assertEqual((retried["result"]["status"], retried["result"]["phase"]), ("active", "await_answer"))
        self.assertEqual(len(self.calls), before)
        self.assertEqual(
            self.action(session_id, "answer", answer="new ephemeral answer")["result"]["phase"], "await_next"
        )
        self.assertEqual(len(self.calls), before + 1)

    def test_restart_pending_note_needs_explicit_retry(self):
        pending = interview.start_session({**self.start_request(), "state_dir": self.state, "defer_inference": True})
        session_id = pending["session_id"]
        self.app.close()
        self.app = web.CoachApp(self.state)
        self.server.app = self.app
        self.assertEqual(self.app.sessions({})["sessions"][0]["phase"], "need_question")
        self.assertEqual(self.app.session({"session_id": session_id})["cursor"], 0)
        self.assertIsNone(self.app.config({})["active_job"])
        self.assertEqual(self.calls, [])
        self.assertEqual(self.action(session_id, "retry")["result"]["phase"], "await_answer")

    def test_oversized_answer_unchanged_checkpoint_and_no_inference(self):
        current = self.start()
        session_id = current["session_id"]
        before = self.snapshot(session_id).read_bytes()
        answer = "\u042f" * 8000
        status, response, _ = self.http("POST", "/api/action", self.action_request(session_id, "answer", answer=answer))
        self.assertEqual(status, 202)
        failed = self.complete(response)
        self.assertEqual(failed["error"], "answer_too_large")
        self.assertEqual(self.snapshot(session_id).read_bytes(), before)
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn(answer, json.dumps(failed))
        self.assertEqual(self.action(session_id, "answer", answer="shorter response")["result"]["phase"], "await_next")

    def test_topic_auto_progress_batches_expansion_refusal_and_no_auto_retry(self):
        for n in range(6):
            (self.vault / f"extra-{n}.md").write_text(f"# Cache {n}\nCache has example number {n}.\n")
        request = self.start_request()
        request.pop("note")
        request["topic"] = "absent literal topic"
        build = interview.build_index
        with patch("interview.build_index", wraps=build) as indexed:
            admitted = self.complete(self.app.start(request))
        self.assertGreater(indexed.call_count, 1)
        self.assertEqual(admitted["result"]["phase"], "await_answer")
        self.assertEqual(
            [call["format"] for call in self.calls], [interview.EXPANSION_SCHEMA, interview.ADMISSION_SCHEMA]
        )
        self.admit = False
        self.expansion = ["also absent"]
        request = {**request, "request_id": uuid.uuid4().hex}
        refused = self.complete(self.app.start(request))
        self.assertEqual((refused["result"]["phase"], refused["error"]), ("refused", "topic_not_found"))
        self.assertTrue(refused["result"]["suggestions"])
        self.failure = "ollama request timed out"
        request = {**request, "request_id": uuid.uuid4().hex, "topic": "cache"}
        before = len(self.calls)
        blocked = self.complete(self.app.start(request))
        self.assertEqual((blocked["result"]["status"], blocked["error"]), ("blocked", "timeout"))
        self.assertEqual(len(self.calls), before + 1)

    def test_topic_budget_stops_at_64_and_skip_can_exhaust(self):
        self.note.write_text("".join(f"# Cache {n}\nCache stores synthetic example {n}.\n" for n in range(70)))
        self.admit = False
        request = self.start_request()
        request.pop("note")
        request["topic"] = "cache"
        incomplete = self.complete(self.app.start(request))
        self.assertEqual((incomplete["result"]["cursor"], incomplete["result"]["phase"]), (64, "search_incomplete"))
        self.assertEqual(len(self.calls), 64)
        self.assertEqual(self.action(incomplete["session_id"], "retry")["result"]["cursor"], 64)
        self.assertEqual(len(self.calls), 64)
        self.assertEqual(self.action(incomplete["session_id"], "finish")["result"]["status"], "finished")
        self.note.write_text("# Cache\nCache stores one example.\n")
        self.admit = True
        admitted = self.complete(self.app.start({**request, "request_id": uuid.uuid4().hex}))
        exhausted = self.action(admitted["session_id"], "skip")
        self.assertEqual(exhausted["result"]["phase"], "exhausted")
        self.assertEqual(exhausted["result"]["status"], "active")

    def test_pause_resume_skip_finish_and_source_change_recovery(self):
        first = self.start()
        session_id = first["session_id"]
        self.assertEqual(self.action(session_id, "pause")["result"]["status"], "paused")
        before = self.snapshot(session_id).read_bytes()
        self.assertEqual(self.action(session_id, "skip")["error"], "skip_not_expected")
        self.assertEqual(self.snapshot(session_id).read_bytes(), before)
        self.assertEqual(self.action(session_id, "resume")["result"]["question"], first["result"]["question"])
        skipped = self.action(session_id, "skip")
        self.assertEqual((skipped["result"]["cursor"], skipped["result"]["phase"]), (1, "await_answer"))
        self.assertTrue(all("answer" not in json.loads(call["messages"][-1]["content"]) for call in self.calls))
        self.note.write_text("# Changed\nChanged source text.\n")
        self.assertEqual(self.action(session_id, "answer", answer="ephemeral")["error"], "source_changed")
        self.assertEqual(self.action(session_id, "finish")["result"]["status"], "finished")
        self.assertEqual(self.action(session_id, "resume")["error"], "session_finished")
        self.assertTrue(self.snapshot(session_id).exists())

    def test_bounded_jobs_and_idempotency_storage(self):
        current = self.start()
        with patch.object(web, "MAX_JOBS", 3), patch.object(web, "MAX_REQUEST_IDS", 4):
            for _ in range(8):
                self.action(current["session_id"], "pause")
        self.assertLessEqual(len(self.app.jobs), 3)
        self.assertLessEqual(len(self.app.requests), 4)
        self.assertEqual(self.app.job({"job_id": current["job_id"]})["error"], "job_not_found")
        self.assertTrue(all(value[1] in self.app.jobs for value in self.app.requests.values()))

    def test_second_server_same_state_refused_and_close_releases_lock(self):
        self.assertEqual(web.create_server({"port": 0, "state_dir": self.state})["error"], "web_busy")
        self.app.close()
        replacement = web.create_server({"port": 0, "state_dir": self.state})
        self.assertTrue(replacement["ok"], replacement)
        replacement["server"].server_close()
        self.assertFalse(self.state.exists())

    def test_http_logging_contains_no_paths_answers_or_token(self):
        with patch("sys.stderr", new_callable=io.StringIO) as log:
            response = self.http("POST", "/api/start", self.start_request())[1]
            current = self.complete(response)
            self.http("GET", "/api/session?session_id=" + current["session_id"])
            self.complete(
                self.http(
                    "POST",
                    "/api/action",
                    self.action_request(current["session_id"], "answer", answer="PRIVATE-EPHEMERAL"),
                )[1]
            )
        self.assertEqual(log.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
