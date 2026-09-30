"""Synthetic, offline checks for the explicit-note interview controller."""

import fcntl
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import interview
from vault_source import load_note


class InterviewTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parent / "tmp"
        scratch.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(dir=scratch))
        # Session data must be external to the repository, even for a fixture.
        self.external = Path(tempfile.mkdtemp())
        self.vault = self.root / "vault"
        (self.vault / "nested").mkdir(parents=True)
        self.note = self.vault / "nested" / "cache.md"
        self.note.write_text(
            "# Cache\nA cache stores reusable results.\n# Invalidation\nInvalidation removes stale entries.\n"
        )
        self.original = self.note.read_bytes()
        self.calls = []
        self.failure = None
        self.proposal = None
        self.mock = patch("interview.chat", side_effect=self.model)
        self.mock.start()
        self.addCleanup(self.mock.stop)
        self.addCleanup(self.clean_files)
        self.request = {
            "vault": str(self.vault),
            "note": "nested/cache.md",
            "endpoint": "http://127.0.0.1:11434",
            "state_dir": str(self.external / "sessions"),
        }

    def clean_files(self):
        # Only remove files created by this test; no recursive force removal.
        for root in (self.root, self.external):
            for directory, subdirs, files in os.walk(root, topdown=False):
                for name in files:
                    (Path(directory) / name).unlink()
                for name in subdirs:
                    child = Path(directory) / name
                    if child.is_symlink():
                        child.unlink()
                    else:
                        child.rmdir()
            root.rmdir()

    def model(self, request):
        self.calls.append(request)
        self.assertLessEqual(interview._body_size(request), 3072)
        if self.failure:
            return {"ok": False, "content": None, "error": self.failure, "metrics": {}}
        if self.proposal is not None:
            return {"ok": True, "content": self.proposal, "error": None, "metrics": {}}
        payload = json.loads(request["messages"][-1]["content"])
        quote = next(line for line in payload["source"].split("\n") if line.strip() and not line.startswith("#"))
        content = (
            {
                "label": "supported",
                "feedback": "This agrees with the supplied note; it is not an objective grade.",
                "quote": quote,
            }
            if "answer" in payload
            else {"question": "What does the note say?", "quote": quote}
        )
        return {"ok": True, "content": content, "error": None, "metrics": {}}

    def start(self, **changes):
        result = interview.start_session({**self.request, **changes})
        if "session_id" in result:
            self.session_id = result["session_id"]
        return result

    def send(self, action, **fields):
        return interview.dispatch(
            {"session_id": self.session_id, "state_dir": self.request["state_dir"], "action": action, **fields}
        )

    def snapshot(self):
        return Path(self.request["state_dir"]) / (self.session_id + ".json")

    def test_full_flow_absolute_path_pause_resume_and_no_vault_writes(self):
        current = self.start(note=str(self.note))
        self.assertTrue(current["ok"])
        self.assertEqual(current["phase"], "await_answer")
        self.assertEqual(current["question"]["citation"]["path"], "nested/cache.md")
        self.assertEqual(current["question"]["citation"]["start_line"], 2)
        question = current["question"]
        self.assertEqual(self.send("pause")["status"], "paused")
        resumed = self.send("resume")
        self.assertEqual(resumed["question"], question)
        self.assertEqual(len(self.calls), 1)
        secret = "EPHEMERAL-ANSWER-777"
        assessed = self.send("answer", answer=secret)
        self.assertEqual(assessed["phase"], "await_next")
        self.assertEqual(assessed["feedback"]["label"], "supported")
        self.assertNotIn(secret, self.snapshot().read_text())
        self.assertEqual(self.send("next")["cursor"], 1)
        self.send("answer", answer="stale entries")
        exhausted = self.send("next")
        self.assertEqual(exhausted["phase"], "exhausted")
        self.assertEqual(exhausted["status"], "active")
        self.assertEqual(self.send("finish")["status"], "finished")
        self.assertFalse(self.send("resume")["ok"])
        self.assertEqual(self.note.read_bytes(), self.original)
        self.assertEqual(set(self.vault.iterdir()), {self.vault / "nested"})
        self.assertEqual(set((self.vault / "nested").iterdir()), {self.note})

    def test_paths_and_state_directories_refuse_before_inference(self):
        outside = self.root / "outside.md"
        outside.write_text("Synthetic outside note")
        link = self.vault / "escape.md"
        link.symlink_to(outside)
        for note in ("../outside.md", "escape.md", str(outside)):
            with self.subTest(note=note):
                self.assertFalse(self.start(note=note)["ok"])
        for state_dir in (self.vault / "state", self.root / "state"):
            with self.subTest(state_dir=state_dir):
                self.assertEqual(self.start(state_dir=str(state_dir))["error"], "invalid_state_dir")
                self.assertFalse(state_dir.exists())
        self.assertFalse(self.start(endpoint="http://localhost:11434")["ok"])
        self.assertEqual(self.calls, [])

    def test_existing_state_directory_permissions_are_not_changed(self):
        directory = self.external / "public-state"
        directory.mkdir(mode=0o755)
        self.assertEqual(self.start(state_dir=str(directory))["error"], "invalid_state_dir")
        self.assertEqual(directory.stat().st_mode & 0o777, 0o755)

    def test_fabricated_repeated_and_malformed_proposals_block(self):
        self.note.write_text("# Cache\nrepeat repeat\n")
        for proposal, error in (
            ({"question": "Q?", "quote": "fabricated"}, "invalid_citation"),
            ({"question": "Q?", "quote": "repeat"}, "invalid_citation"),
            ({"question": "Q?", "quote": "repeat repeat", "action": "finish"}, "invalid_output"),
        ):
            self.proposal = proposal
            blocked = self.start()
            self.assertEqual(blocked["error"], error)
            self.assertEqual(blocked["status"], "blocked")
            self.assertEqual(blocked["cursor"], 0)
            self.assertIsNone(blocked["question"])
            self.assertEqual(self.send("finish")["status"], "finished")

    def test_failed_assessment_requires_answer_reentry(self):
        question = self.start()["question"]
        self.failure = "ollama request timed out"
        result = self.send("answer", answer="EPHEMERAL-FAILED-ANSWER")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["phase"], "await_answer")
        self.assertEqual(result["question"], question)
        self.assertNotIn("EPHEMERAL-FAILED-ANSWER", self.snapshot().read_text())
        count = len(self.calls)
        self.assertFalse(self.send("answer", answer="must retry first")["ok"])
        self.failure = None
        retried = self.send("retry")
        self.assertEqual(retried["phase"], "await_answer")
        self.assertEqual(retried["status"], "active")
        self.assertEqual(len(self.calls), count)
        self.assertEqual(self.send("answer", answer="reentered")["phase"], "await_next")

    def test_failed_question_pause_resume_preserves_block_then_retry(self):
        self.failure = "model unavailable"
        result = self.start()
        self.assertEqual(result["phase"], "need_question")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(self.send("pause")["status"], "paused")
        self.assertEqual(self.send("resume")["status"], "blocked")
        self.failure = None
        self.assertEqual(self.send("retry")["phase"], "await_answer")

    def test_pause_finish_need_no_source_or_model(self):
        self.start()
        self.note.rename(self.note.with_suffix(".gone"))
        self.failure = "model unavailable"
        count = len(self.calls)
        self.assertEqual(self.send("pause")["status"], "paused")
        self.assertEqual(self.send("finish")["status"], "finished")
        self.assertEqual(len(self.calls), count)

    def test_source_change_blocks_without_resetting_progress(self):
        current = self.start()
        self.send("pause")
        self.note.write_text(self.original.decode() + "Source changed.\n")
        resumed = self.send("resume")
        self.assertEqual(resumed["error"], "source_changed")
        self.assertEqual(resumed["question"], current["question"])
        self.assertEqual(resumed["cursor"], 0)
        self.assertFalse(self.send("retry")["ok"])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.send("finish")["status"], "finished")

    def test_corrupt_snapshot_refuses_without_overwrite(self):
        self.start()
        valid = json.loads(self.snapshot().read_text())
        for corrupt in (
            b"{",
            json.dumps({**valid, "cursor": 999}).encode(),
            json.dumps({**valid, "status": "unknown"}).encode(),
        ):
            self.snapshot().write_bytes(corrupt)
            self.assertEqual(self.send("resume")["error"], "corrupt_state")
            self.assertEqual(self.snapshot().read_bytes(), corrupt)
        self.assertEqual(len(self.calls), 1)

    def test_oversized_answer_is_not_truncated_or_sent(self):
        self.start()
        count = len(self.calls)
        result = self.send("answer", answer="PRIVATE-OVERSIZED" * 1000)
        self.assertEqual(result["error"], "answer_too_large")
        self.assertEqual(result["phase"], "await_answer")
        self.assertEqual(len(self.calls), count)
        self.assertNotIn("PRIVATE-OVERSIZED", self.snapshot().read_text())

    def test_private_atomic_snapshot_lock_and_persistence_failure(self):
        self.start()
        directory = Path(self.request["state_dir"])
        self.assertEqual(directory.stat().st_mode & 0o777, 0o700)
        self.assertEqual(self.snapshot().stat().st_mode & 0o777, 0o600)
        with (directory / (self.session_id + ".lock")).open("rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(self.send("finish")["error"], "session_busy")
        before = self.snapshot().read_bytes()
        with patch("interview.os.replace", side_effect=OSError("private diagnostic")):
            result = self.send("pause")
        self.assertEqual(result["error"], "persistence_failure")
        self.assertEqual(self.snapshot().read_bytes(), before)
        self.assertEqual(self.send("resume")["status"], "active")

    def test_sections_fences_and_long_unicode_are_retained(self):
        text = "Preamble\n\nATX\n===\n```\n# Not a heading\n```\n# Last\n" + "é\\\"" * 300
        self.note.write_text(text)
        note = load_note({"vault": str(self.vault), "note": str(self.note), "max_chunk_bytes": 64})
        self.assertEqual("".join(chunk["text"] for chunk in note["chunks"]), text)
        self.assertEqual(set(chunk["heading"] for chunk in note["chunks"]), {"", "ATX", "Last"})
        # Inspect bounded requests without assuming a semantic model judgement.
        self.proposal = {"question": "What is in this section?", "quote": "Preamble"}
        self.assertTrue(self.start()["ok"])

    def test_thematic_breaks_and_cr_lines_keep_section_boundaries(self):
        for text, headings in (
            ("Intro\n***\nTitle\n===\nBody\n", ["", "Title"]),
            ("Intro\n***\n---\nBody\n", [""]),
            ("# One\rBody\r# Two\rOther\r", ["One", "Two"]),
        ):
            with self.subTest(text=text):
                self.note.write_bytes(text.encode())
                chunks = load_note({"vault": str(self.vault), "note": str(self.note), "max_chunk_bytes": 4})["chunks"]
                self.assertEqual("".join(chunk["text"] for chunk in chunks), text)
                self.assertEqual(list(dict.fromkeys(chunk["heading"] for chunk in chunks)), headings)

    def test_interrupt_during_request_retains_session_id_and_pending_phase(self):
        with patch("interview.chat", side_effect=KeyboardInterrupt):
            result = self.start()
        self.assertEqual(result["error"], "interrupted")
        self.assertEqual(result["phase"], "need_question")
        self.assertEqual(self.send("retry")["phase"], "await_answer")
        saved = self.snapshot().read_bytes()
        with patch("interview.chat", side_effect=KeyboardInterrupt):
            result = self.send("answer", answer="PRIVATE-INTERRUPTED")
        self.assertEqual(result["error"], "interrupted")
        self.assertEqual(self.snapshot().read_bytes(), saved)

    def test_json_boundary_rejects_invalid_action_and_unicode(self):
        self.start()
        self.assertEqual(self.send([])["error"], "invalid_action")
        self.assertEqual(self.send("answer", answer="\ud800")["error"], "invalid_request")

    def test_model_cannot_persist_an_echoed_answer(self):
        self.start()
        self.proposal = {
            "label": "supported",
            "feedback": "You wrote PRIVATE-ECHO-ANSWER.",
            "quote": "A cache stores reusable results.",
        }
        result = self.send("answer", answer="private-echo-answer \n")
        self.assertEqual(result["status"], "blocked")
        self.assertNotIn("PRIVATE-ECHO-ANSWER", self.snapshot().read_text())

    def test_source_change_during_inference_does_not_acknowledge_question(self):
        def mutate(request):
            result = self.model(request)
            self.note.write_text(self.original.decode() + "Changed during request\n")
            return result

        with patch("interview.chat", side_effect=mutate):
            result = self.start()
        self.assertEqual(result["error"], "source_changed")
        self.assertIsNone(result["question"])
        self.assertEqual(self.send("finish")["status"], "finished")

    def test_cli_keeps_session_after_unknown_command(self):
        commands = [":unknown", "answer", ":pause"]
        with patch(
            "sys.argv",
            [
                "interview.py",
                "start",
                "--vault",
                str(self.vault),
                "--note",
                str(self.note),
                "--endpoint",
                self.request["endpoint"],
                "--state-dir",
                self.request["state_dir"],
            ],
        ), patch("builtins.input", side_effect=commands), patch("builtins.print"):
            self.assertEqual(interview.main(), 0)
        states = list(Path(self.request["state_dir"]).glob("*.json"))
        self.assertEqual(len(states), 1)
        state = json.loads(states[0].read_text())
        self.assertEqual(state["status"], "paused")
        self.assertEqual(state["phase"], "await_next")

    def test_dispatch_rejects_vault_state_before_any_write(self):
        vault = self.external / "external-vault"
        vault.mkdir()
        (vault / "note.md").write_bytes(self.original)
        self.start(vault=str(vault), note="note.md")
        directory = vault / "private-state"
        directory.mkdir(mode=0o700)
        snapshot = directory / self.snapshot().name
        snapshot.write_bytes(self.snapshot().read_bytes())
        snapshot.chmod(0o600)
        result = interview.dispatch({"session_id": self.session_id, "action": "finish", "state_dir": str(directory)})
        self.assertEqual(result["error"], "invalid_state_dir")
        self.assertEqual(list(directory.iterdir()), [snapshot])
        self.assertEqual(snapshot.read_bytes(), self.snapshot().read_bytes())

    def test_start_persistence_failure_returns_recoverable_session_id(self):
        replace = os.replace
        calls = 0

        def fail_second_replace(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("synthetic persistence failure")
            return replace(*args, **kwargs)

        with patch("interview.os.replace", side_effect=fail_second_replace):
            result = self.start()
        self.assertEqual(result["error"], "persistence_failure")
        self.assertIn("session_id", result)
        self.assertEqual(self.send("resume")["phase"], "need_question")

    def test_replaced_vault_symlink_refuses_before_reading_new_source(self):
        self.start()
        self.vault.rename(self.root / "original-vault")
        replacement = self.external / "replacement-vault"
        (replacement / "nested").mkdir(parents=True)
        (replacement / "nested" / "cache.md").write_bytes(self.original)
        self.vault.symlink_to(replacement, target_is_directory=True)
        with patch("interview.load_note", wraps=load_note) as loader:
            result = self.send("resume")
        self.assertEqual(result["error"], "source_changed")
        loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
