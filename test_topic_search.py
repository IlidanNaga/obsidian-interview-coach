"""Synthetic, offline checks for private topic discovery."""

from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import topic_search
from vault_source import SourceError, cite, load_note


class TopicSearchTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parent / "tmp"
        scratch.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(dir=scratch))
        # Private fixture state must be outside the repository, as in production.
        self.external = Path(tempfile.mkdtemp())
        self.vault = self.root / "vault"
        self.vault.mkdir()
        self.state = self.external / "state"
        self.request = {"vault": str(self.vault), "state_dir": str(self.state)}
        self.addCleanup(self.clean_files)

    def clean_files(self):
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

    def note(self, path, text):
        note = self.vault / path
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text(text, encoding="utf-8")
        return note

    def build(self, **fields):
        return topic_search.build_index({**self.request, **fields})

    def search(self, topic, **fields):
        return topic_search.search_topic({**self.request, "topic": topic, **fields})

    def database(self):
        return next(self.state.glob("*.sqlite3"))

    def test_candidates_metadata_ranking_paging_and_real_suggestions(self):
        first = self.note(
            "nested/cache.md", "# Cache\nCache stores results.\n# Eviction\nCache eviction removes entries.\n"
        )
        self.note("plain.md", "Other synthetic material.\n")
        before = first.read_bytes()
        built = self.build()
        self.assertEqual((built["complete"], built["done"], built["total"], built["indexed"]), (True, 2, 2, 2))
        result = self.search("cache")
        self.assertTrue(result["complete"])
        self.assertEqual(result["total_candidates"], 2)
        self.assertEqual(result["candidates"], self.search("cache")["candidates"])
        source = load_note({"vault": str(self.vault), "note": str(first), "max_chunk_bytes": 512})
        for candidate in result["candidates"]:
            self.assertEqual(candidate["source"], {key: source[key] for key in source if key != "chunks"})
            self.assertIn(candidate["chunk"], source["chunks"])
            self.assertEqual(
                cite(candidate["chunk"], "Cache" if candidate["chunk"]["heading"] == "Eviction" else "stores")["path"],
                "nested/cache.md",
            )
        page = self.search("cache", limit=1)
        self.assertEqual(page["next_offset"], 1)
        last = self.search("cache", limit=1, offset=page["next_offset"])
        self.assertIsNone(last["next_offset"])
        self.assertEqual(page["candidates"] + last["candidates"], result["candidates"])
        absent = self.search("nonexistent astronomy")
        self.assertTrue(absent["complete"])
        self.assertEqual(absent["candidates"], [])
        self.assertEqual(
            {(item["path"], item["topic"]) for item in absent["suggestions"]},
            {
                ("nested/cache.md", "Cache"),
                ("nested/cache.md", "Eviction"),
                ("plain.md", "plain"),
            },
        )
        self.assertEqual(first.read_bytes(), before)
        json.dumps(result, ensure_ascii=False, allow_nan=False)

    def test_interruption_resume_skips_committed_notes(self):
        self.note("a.md", "# Alpha\nAlpha material.\n")
        self.note("b.md", "# Beta\nBeta material.\n")
        loader = topic_search.load_note
        calls = 0

        def interrupt_second(request):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt
            return loader(request)

        with patch("topic_search.load_note", side_effect=interrupt_second):
            interrupted = self.build()
        self.assertEqual(interrupted["error"], "interrupted")
        self.assertEqual((interrupted["done"], interrupted["total"]), (1, 2))
        self.assertFalse(self.search("absent")["complete"])
        self.assertEqual({path.suffix for path in self.state.iterdir()}, {".sqlite3", ".lock"})
        with patch("topic_search.load_note", wraps=loader) as mocked:
            resumed = self.build()
        self.assertTrue(resumed["complete"])
        self.assertEqual((resumed["indexed"], resumed["unchanged"]), (1, 1))
        self.assertEqual(mocked.call_count, 1)
        with patch("topic_search.load_note", side_effect=AssertionError("unchanged note read")):
            repeated = self.build()
        self.assertEqual((repeated["indexed"], repeated["unchanged"]), (0, 2))

    def test_interrupt_inside_transaction_rolls_back_one_note(self):
        self.note("a.md", "# Cache\nCache data.\n")
        self.note("b.md", "# Cache\nMore cache data.\n")
        loader = topic_search._bounded_note

        class InterruptedChunks:
            def __iter__(self):
                raise KeyboardInterrupt

        def interrupt_insert(vault, note, *args):
            result = loader(vault, note, *args)
            if note.name == "b.md":
                result["chunks"] = InterruptedChunks()
            return result

        with patch("topic_search._bounded_note", side_effect=interrupt_insert):
            interrupted = self.build()
        self.assertEqual(interrupted["done"], 1)
        with closing(sqlite3.connect(self.database())) as connection:
            self.assertEqual(connection.execute("SELECT path FROM notes").fetchall(), [("a.md",)])
        self.assertEqual(self.build()["indexed"], 1)

    def test_batched_progress_and_stale_rows_removed(self):
        old = self.note("a.md", "# Old\nOld body.\n")
        deleted = self.note("b.md", "# Deleted\nDeleted body.\n")
        first = self.build(max_notes=1)
        self.assertEqual((first["done"], first["total"], first["scan_complete"]), (1, 2, False))
        second = self.build(max_notes=1)
        self.assertEqual((second["done"], second["total"], second["scan_complete"]), (2, 2, True))
        old.write_text("# New\nReplacement body.\n")
        deleted.rename(deleted.with_suffix(".gone"))
        refreshed = self.build()
        self.assertEqual((refreshed["done"], refreshed["total"], refreshed["indexed"]), (1, 1, 1))
        for topic in ("Old", "Deleted"):
            self.assertEqual(self.search(topic)["candidates"], [])
        self.assertEqual(self.search("New")["candidates"][0]["chunk"]["heading"], "New")

    def test_excluded_utf8_oversized_unreadable_and_changed_notes(self):
        self.note("good.md", "# Valid\nValid body.\n")
        bad = self.note("bad.md", "temporary")
        bad.write_bytes(b"\xff\xfe")
        huge = self.note("huge.md", "")
        with huge.open("wb") as stream:
            stream.truncate(topic_search.MAX_NOTE_BYTES + 1)
        denied = self.note("denied.md", "# Denied\nUnreadable fixture.\n")
        changed = self.note("changed.md", "# Changed\nMutable fixture.\n")
        bounded = topic_search._bounded_note

        def exclude(vault, note, *args):
            self.assertNotEqual(note, huge, "oversized source reached full read")
            if note == denied:
                raise PermissionError("synthetic unreadable note")
            if note == changed:
                raise SourceError("source_changed")
            return bounded(vault, note, *args)

        with patch("topic_search._bounded_note", side_effect=exclude):
            built = self.build()
        self.assertTrue(built["scan_complete"])
        self.assertFalse(built["complete"])
        self.assertEqual((built["done"], built["total"], built["excluded"]), (5, 5, 4))
        negative = self.search("astronomy")
        self.assertFalse(negative["complete"])
        self.assertEqual(negative["excluded"], 4)
        self.assertEqual(negative["candidates"], [])
        self.assertEqual({item["path"] for item in negative["suggestions"]}, {"good.md"})
        bad.write_text("# Repaired\nRepaired UTF-8.\n")
        huge.write_text("# Small\nNow small.\n")
        self.assertTrue(self.build()["complete"])

    def test_growth_after_stat_is_bounded_and_excluded(self):
        note = self.note("grow.md", "# Small\nInitial body.\n")
        bounded = topic_search._bounded_note

        def grow(vault, selected, *args):
            with note.open("ab") as stream:
                stream.truncate(topic_search.MAX_NOTE_BYTES + 1)
            return bounded(vault, selected, *args)

        with patch("topic_search._bounded_note", side_effect=grow), patch("topic_search.load_note") as loader:
            result = self.build()
        loader.assert_not_called()
        self.assertEqual((result["excluded"], result["complete"]), (1, False))
        self.assertEqual(self.search("Small")["candidates"], [])

    def test_source_mutation_during_parse_excludes_old_chunks(self):
        note = self.note("mutable.md", "# Previous\nPrevious body.\n")
        self.build()
        note.write_text("# Current\nCurrent body.\n")
        loader = topic_search.load_note

        def mutate(request):
            result = loader(request)
            note.write_text("# Final\nFinal body.\n")
            return result

        with patch("topic_search.load_note", side_effect=mutate):
            result = self.build()
        self.assertEqual(result["excluded"], 1)
        for topic in ("Previous", "Current"):
            self.assertEqual(self.search(topic)["candidates"], [])
        self.assertTrue(self.build()["complete"])
        self.assertEqual(len(self.search("Final")["candidates"]), 1)

    def test_symlinks_containment_hidden_directories_cycles_and_deduplication(self):
        source = self.note("nested/cache.md", "# Cache\nCache body.\n")
        hidden = self.note(".hidden/private.md", "# Hidden\nHidden body.\n")
        outside = self.root / "outside.md"
        outside.write_text("# Outside\nOutside body.\n")
        (self.vault / "alias.md").symlink_to(source)
        (self.vault / "directory").symlink_to(source.parent, target_is_directory=True)
        (source.parent / "cycle").symlink_to(self.vault, target_is_directory=True)
        (self.vault / "hidden-alias.md").symlink_to(hidden)
        (self.vault / "hidden-directory").symlink_to(hidden.parent, target_is_directory=True)
        (self.vault / "escape.md").symlink_to(outside)
        (self.vault / "broken.md").symlink_to(self.root / "missing.md")
        (self.vault / "loop.md").symlink_to(self.vault / "loop.md")
        bounded = topic_search._bounded_note

        def contained(vault, note, *args):
            self.assertEqual(note, source)
            return bounded(vault, note, *args)

        with patch("topic_search._bounded_note", side_effect=contained) as loader:
            built = self.build()
        self.assertEqual(loader.call_count, 1)
        self.assertEqual((built["indexed"], built["excluded"], built["total"]), (1, 3, 4))
        self.assertEqual(len(self.search("cache")["candidates"]), 1)
        for topic in ("Hidden", "Outside"):
            self.assertEqual(self.search(topic)["candidates"], [])

    def test_fts_literal_punctuation_unicode_and_query_validation(self):
        self.note("unicode.md", '# Кэш\nКэш хранит результаты.\n# C++\nC++ and "quoted" text.\n# 缓存\n缓存 数据\n')
        self.assertTrue(self.build()["complete"])
        for topic in ("КЭШ", "C++", '"quoted"', "缓存"):
            self.assertTrue(self.search(topic)["candidates"], topic)
        for topic in ("!!!", 'cache OR "*"', "NOT", "'; DROP TABLE notes; --", "\0", "absent"):
            result = self.search(topic)
            self.assertTrue(result["ok"], topic)
            self.assertEqual(result["candidates"], [], topic)
        for topic in ("", " ", "x" * 257, "\ud800", 1):
            self.assertEqual(self.search(topic)["error"], "invalid_request")
        for fields in ({"limit": True}, {"limit": 101}, {"offset": -1}):
            self.assertEqual(self.search("cache", **fields)["error"], "invalid_request")

    def test_state_privacy_and_database_links_refuse_before_writes(self):
        self.note("cache.md", "# Cache\nCache body.\n")
        for directory in (self.vault / "state", self.root / "state"):
            self.assertEqual(self.build(state_dir=str(directory))["error"], "invalid_state_dir")
            self.assertFalse(directory.exists())
        public = self.external / "public"
        public.mkdir(mode=0o755)
        public.chmod(0o755)
        self.assertEqual(self.build(state_dir=str(public))["error"], "invalid_state_dir")
        self.assertEqual(public.stat().st_mode & 0o777, 0o755)
        self.assertTrue(self.build()["complete"])
        self.assertEqual(self.state.stat().st_mode & 0o777, 0o700)
        self.assertEqual(self.database().stat().st_mode & 0o777, 0o600)
        original = self.database()
        moved = self.external / "saved.sqlite3"
        original.rename(moved)
        original.symlink_to(moved)
        self.assertFalse(self.build()["ok"])
        original.unlink()
        os.link(moved, original)
        self.assertEqual(self.build()["error"], "invalid_index")
        original.unlink()
        moved.rename(original)
        sidecar = Path(str(original) + "-journal")
        sidecar.symlink_to(self.vault / "cache.md")
        self.assertEqual(self.build()["error"], "invalid_index")
        self.assertEqual((self.vault / "cache.md").read_text(), "# Cache\nCache body.\n")

    def test_empty_vault_missing_index_and_chunk_size_refresh(self):
        self.assertFalse(self.search("cache")["ok"])
        empty = self.build()
        self.assertEqual((empty["complete"], empty["done"], empty["total"]), (True, 0, 0))
        self.assertTrue(self.search("cache")["complete"])
        self.note("long.md", "# Cache\n" + "cache " * 40)
        self.build(max_chunk_bytes=64)
        candidates = self.search("cache")["candidates"]
        self.assertGreater(len(candidates), 1)
        self.assertTrue(all(len(item["chunk"]["text"].encode()) <= 64 for item in candidates))
        self.assertEqual(self.build(max_chunk_bytes=128)["indexed"], 1)


if __name__ == "__main__":
    unittest.main()
