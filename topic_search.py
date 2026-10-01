"""Private, resumable SQLite FTS5 discovery; no model calls or topic admission.

build_index({"vault": str, "state_dir": str, "max_chunk_bytes": int=512,
             "max_notes": int|null=None}) returns:
  {ok, error, complete, scan_complete, done, total, indexed, unchanged, excluded}.
max_notes bounds note transactions per call; repeat the same request until
scan_complete, displaying `done N/total` between calls. Unchanged notes are
skipped; exclusions are counted, and source_changed exclusions are retried.

search_topic({"vault": str, "state_dir": str, "topic": str,
              "limit": int=20, "offset": int=0}) returns:
  {ok, error, complete, scan_complete, done, total, excluded, candidates,
   suggestions, total_candidates, next_offset}.
Each candidate is {source, chunk, rank}: source has load_note's vault/note/path/
identity/sha256 fields, and chunk has exactly its chunk fields. Lower rank is
better; ties use path then chunk ID. Suggestions are {path, heading, topic},
where topic is an actual heading or the note's filename stem.

ok means the operation ran; complete means the last inventory finished with
no exclusions. An empty candidate list supports refusal only when complete;
it is never model admission. Re-read a candidate through load_note before use.
Failures return {ok: False, error: classified_code, complete: False}; no source
text or absolute paths appear in errors. state_dir is required, private, and
outside both Vault and repository. All returned values are JSON-shaped.
"""

from __future__ import annotations

from contextlib import contextmanager
import difflib
import fcntl
import hashlib
import heapq
import json
import os
from pathlib import Path
import sqlite3
import stat
import tempfile

from vault_source import SourceError, _identity, load_note

REPO = Path(__file__).resolve().parent
MAX_NOTE_BYTES = 8 * 1024 * 1024
MAX_HEADING_BYTES = 8 * 1024
CHUNK_FIELDS = ("id", "section_id", "path", "heading", "start_line", "end_line", "start_offset", "end_offset", "text")


def _paths(request, create=False):
    if not isinstance(request, dict) or any(
        not isinstance(request.get(key), str) or not request[key] for key in ("vault", "state_dir")
    ):
        raise SourceError("invalid_request")
    try:
        vault = Path(request["vault"]).resolve(strict=True)
        if not vault.is_dir():
            raise SourceError("invalid_source")
        directory = Path(request["state_dir"]).expanduser().resolve()
        if directory.is_relative_to(vault) or directory.is_relative_to(REPO):
            raise SourceError("invalid_state_dir")
        if create:
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = directory.stat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise SourceError("invalid_state_dir")
        return vault, directory
    except (OSError, RuntimeError):
        raise SourceError("state_unavailable") from None


@contextmanager
def _database(vault, directory, create=False):
    name = "topic-" + hashlib.sha256(str(vault).encode()).hexdigest() + ".sqlite3"
    path = directory / name
    descriptor = None
    lock_descriptor = None
    connection = None
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK | (os.O_CREAT if create else 0), 0o600)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1 or info.st_mode & 0o077:
            raise SourceError("invalid_index")
        # Lock a separate file: macOS SQLite may itself use flock on the DB.
        lock_descriptor = os.open(str(path) + ".lock", os.O_RDWR | os.O_NOFOLLOW | (os.O_CREAT if create else 0), 0o600)
        lock_info = os.fstat(lock_descriptor)
        if (
            not stat.S_ISREG(lock_info.st_mode)
            or lock_info.st_uid != os.getuid()
            or lock_info.st_nlink != 1
            or lock_info.st_mode & 0o077
        ):
            raise SourceError("invalid_index")
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SourceError("index_busy") from None
        # SQLite also opens sidecars by name; reject links before it can do so.
        for suffix in ("-journal", "-wal", "-shm"):
            sidecar = Path(str(path) + suffix)
            if sidecar.exists() or sidecar.is_symlink():
                side = sidecar.lstat()
                if (
                    not stat.S_ISREG(side.st_mode)
                    or side.st_uid != os.getuid()
                    or side.st_nlink != 1
                    or side.st_mode & 0o077
                ):
                    raise SourceError("invalid_index")
        connection = sqlite3.connect(path.as_uri() + ("?mode=rw" if create else "?mode=ro"), uri=True)
        connection.row_factory = sqlite3.Row
        yield connection
    except FileNotFoundError:
        raise SourceError("index_unavailable") from None
    finally:
        if connection is not None:
            connection.close()
        if lock_descriptor is not None:
            os.close(lock_descriptor)
        if descriptor is not None:
            os.close(descriptor)


@contextmanager
def _open_source(vault, note, directory=False):
    """Open a contained canonical path without following swapped components."""
    descriptors = []
    try:
        parts = note.relative_to(vault).parts
        descriptor = os.open(vault, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(descriptor)
        for i, part in enumerate(parts):
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if directory or i < len(parts) - 1:
                flags |= os.O_DIRECTORY
            descriptor = os.open(part, flags, dir_fd=descriptor)
            descriptors.append(descriptor)
        yield descriptor
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _inventory(vault):
    notes = {}
    visited = set()
    pending = [vault]
    while pending:
        directory = pending.pop()
        if directory in visited:
            continue
        visited.add(directory)
        try:
            with _open_source(vault, directory, directory=True) as descriptor, os.scandir(descriptor) as entries:
                names = sorted(entry.name for entry in entries)
        except OSError:
            notes[str(directory)] = (directory.relative_to(vault).as_posix(), "unreadable_directory")
            continue
        for name in names:
            requested = directory / name
            try:
                target = requested.resolve(strict=True)
                if not target.is_relative_to(vault):
                    if requested.suffix.lower() == ".md":
                        notes[str(requested)] = (requested.relative_to(vault).as_posix(), "invalid_source")
                    continue
                info = target.stat()
                relative = target.relative_to(vault)
                if stat.S_ISDIR(info.st_mode):
                    if not name.startswith(".") and not any(part.startswith(".") for part in relative.parts):
                        pending.append(target)
                elif requested.suffix.lower() == ".md":
                    if any(part.startswith(".") for part in relative.parts[:-1]):
                        continue
                    error = None if stat.S_ISREG(info.st_mode) and target.suffix.lower() == ".md" else "invalid_source"
                    notes[str(target)] = (relative.as_posix(), error)
            except (OSError, RuntimeError):
                if requested.suffix.lower() == ".md":
                    notes[str(requested)] = (requested.relative_to(vault).as_posix(), "invalid_source")
    return sorted(notes.items())


def _bounded_note(vault, note, directory, limit, before):
    """Parse a private bounded snapshot with load_note, retaining original IDs.

    A stat followed by load_note(original) alone races an unbounded read. Keep
    original descriptors open while parsing the snapshot and reject mutation.
    """
    temporary = None
    try:
        with _open_source(vault, note) as descriptor:
            if _identity(os.fstat(descriptor)) != before:
                raise SourceError("source_changed")
            with os.fdopen(os.dup(descriptor), "rb") as stream:
                data = stream.read(MAX_NOTE_BYTES + 1)
            if len(data) > MAX_NOTE_BYTES or len(data) != before["size"]:
                raise SourceError("source_changed")
            output, temporary = tempfile.mkstemp(suffix=".md", dir=directory)
            with os.fdopen(output, "wb") as stream:
                stream.write(data)
            result = load_note({"vault": str(directory), "note": temporary, "max_chunk_bytes": limit})
            if (
                _identity(os.fstat(descriptor)) != before
                or _identity(note.stat()) != before
                or note.resolve(strict=True) != note
                or vault.resolve(strict=True) != vault
            ):
                raise SourceError("source_changed")
        path = note.relative_to(vault).as_posix()
        note_id = hashlib.sha256((path + "\0" + result["sha256"]).encode()).hexdigest()
        for chunk in result["chunks"]:
            chunk.update(path=path)
            for key in ("id", "section_id"):
                chunk[key] = note_id + ":" + chunk[key].split(":", 1)[1]
        result.update(vault=str(vault), note=str(note), path=path, identity=before)
        return result
    finally:
        if temporary is not None:
            os.unlink(temporary)


def _counts(connection):
    meta = connection.execute("SELECT * FROM meta").fetchone()
    counts = connection.execute("SELECT count(*) AS done, count(error) AS excluded FROM notes").fetchone()
    scan_complete = bool(meta["scan_complete"])
    return {
        "scan_complete": scan_complete,
        "complete": scan_complete and counts["excluded"] == 0,
        "done": counts["done"],
        "total": meta["total"],
        "excluded": counts["excluded"],
    }


def build_index(request: dict) -> dict:
    """See the module's exact JSON contract; each note commits independently."""
    try:
        vault, directory = _paths(request, create=True)
        limit = request.get("max_chunk_bytes", 512)
        budget = request.get("max_notes")
        if (
            type(limit) is not int
            or not 4 <= limit <= 512
            or (budget is not None and (type(budget) is not int or budget < 1))
        ):
            raise SourceError("invalid_request")
        with _database(vault, directory, create=True) as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS meta (chunk_bytes INTEGER, scan_complete INTEGER, total INTEGER);
                CREATE TABLE IF NOT EXISTS notes
                    (note TEXT PRIMARY KEY, path TEXT, identity TEXT, sha256 TEXT, error TEXT);
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                    note UNINDEXED, id UNINDEXED, section_id UNINDEXED, path, heading,
                    start_line UNINDEXED, end_line UNINDEXED, start_offset UNINDEXED,
                    end_offset UNINDEXED, text
                );
            """)
            with connection:
                meta = connection.execute("SELECT * FROM meta").fetchone()
                if meta is None or meta["chunk_bytes"] != limit:
                    connection.execute("DELETE FROM chunks")
                    connection.execute("DELETE FROM notes")
                connection.execute("DELETE FROM meta")
                connection.execute("INSERT INTO meta VALUES (?, 0, 0)", (limit,))
            indexed = unchanged = attempts = 0
            try:
                inventory = _inventory(vault)
                keys = {note for note, _ in inventory}
                for row in connection.execute("SELECT note FROM notes").fetchall():
                    if row["note"] not in keys:
                        with connection:
                            connection.execute("DELETE FROM chunks WHERE note = ?", (row["note"],))
                            connection.execute("DELETE FROM notes WHERE note = ?", (row["note"],))
                with connection:
                    connection.execute("UPDATE meta SET total = ?", (len(inventory),))
                pending = False
                for name, (path, error) in inventory:
                    note = Path(name)
                    identity = None
                    result = None
                    try:
                        if error is None:
                            with _open_source(vault, note) as descriptor:
                                info = os.fstat(descriptor)
                                if not stat.S_ISREG(info.st_mode):
                                    raise SourceError("invalid_source")
                                identity = _identity(info)
                                if info.st_size > MAX_NOTE_BYTES:
                                    error = "note_too_large"
                        old = connection.execute("SELECT * FROM notes WHERE note = ?", (name,)).fetchone()
                        encoded = json.dumps(identity, sort_keys=True)
                        if old and old["identity"] == encoded and old["error"] == error:
                            unchanged += 1
                            continue
                        if budget is not None and attempts >= budget:
                            # Never expose old chunks as current after a partial refresh.
                            if old:
                                with connection:
                                    connection.execute("DELETE FROM chunks WHERE note = ?", (name,))
                                    connection.execute("DELETE FROM notes WHERE note = ?", (name,))
                            pending = True
                            continue
                        if error is None:
                            result = _bounded_note(vault, note, directory, limit, identity)
                            if any(
                                len(chunk["heading"].encode("utf-8")) > MAX_HEADING_BYTES for chunk in result["chunks"]
                            ):
                                raise SourceError("heading_too_large")
                    except SourceError as failure:
                        result = None
                        error = failure.code
                    except (OSError, RuntimeError):
                        error = "invalid_source"
                    with connection:
                        connection.execute("DELETE FROM chunks WHERE note = ?", (name,))
                        connection.execute(
                            "INSERT OR REPLACE INTO notes VALUES (?, ?, ?, ?, ?)",
                            (
                                name,
                                path,
                                json.dumps(identity, sort_keys=True),
                                result["sha256"] if result else None,
                                error,
                            ),
                        )
                        if result:
                            connection.executemany(
                                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                ((name, *(chunk[key] for key in CHUNK_FIELDS)) for chunk in result["chunks"]),
                            )
                            indexed += 1
                    attempts += 1
                with connection:
                    connection.execute("UPDATE meta SET scan_complete = ?", (int(not pending),))
                return {"ok": True, "error": None, **_counts(connection), "indexed": indexed, "unchanged": unchanged}
            except KeyboardInterrupt:
                return {
                    "ok": False,
                    "error": "interrupted",
                    **_counts(connection),
                    "indexed": indexed,
                    "unchanged": unchanged,
                }
    except SourceError as error:
        return {"ok": False, "error": error.code, "complete": False}
    except (OSError, sqlite3.Error):
        return {"ok": False, "error": "index_unavailable", "complete": False}
    except (ValueError, TypeError, UnicodeError):
        return {"ok": False, "error": "invalid_request", "complete": False}


def search_topic(request: dict) -> dict:
    """Literal FTS phrase lookup; paging is stable until the next index build."""
    try:
        vault, directory = _paths(request)
        topic = request.get("topic")
        limit, offset = request.get("limit", 20), request.get("offset", 0)
        if (
            not isinstance(topic, str)
            or not topic.strip()
            or len(topic.encode("utf-8")) > 256
            or type(limit) is not int
            or not 1 <= limit <= 100
            or type(offset) is not int
            or offset < 0
        ):
            raise SourceError("invalid_request")
        # Binding SQL alone does not escape the FTS query language. Quote the
        # whole literal phrase as well, including double quotes and NULs.
        query = '"' + topic.replace("\0", " ").replace('"', '""') + '"'
        with _database(vault, directory) as connection:
            counts = _counts(connection)
            total = connection.execute("SELECT count(*) FROM chunks WHERE chunks MATCH ?", (query,)).fetchone()[0]
            rows = connection.execute(
                """
                SELECT chunks.*, notes.identity, notes.sha256, bm25(chunks) AS rank
                FROM chunks JOIN notes ON chunks.note = notes.note
                WHERE chunks MATCH ? ORDER BY rank, chunks.path, chunks.id LIMIT ? OFFSET ?
            """,
                (query, limit, offset),
            )
            candidates = [
                {
                    "source": {
                        "vault": str(vault),
                        "note": row["note"],
                        "path": row["path"],
                        "identity": json.loads(row["identity"]),
                        "sha256": row["sha256"],
                    },
                    "chunk": {key: row[key] for key in CHUNK_FIELDS},
                    "rank": row["rank"],
                }
                for row in rows
            ]
            headings = connection.execute("SELECT DISTINCT path, heading FROM chunks")
            suggestions = heapq.nsmallest(
                5,
                (
                    {"path": row["path"], "heading": row["heading"], "topic": row["heading"] or Path(row["path"]).stem}
                    for row in headings
                ),
                key=lambda item: (
                    -difflib.SequenceMatcher(None, topic.casefold(), item["topic"].casefold()).ratio(),
                    item["path"],
                    item["heading"],
                ),
            )
            return {
                "ok": True,
                "error": None,
                **counts,
                "candidates": candidates,
                "suggestions": suggestions,
                "total_candidates": total,
                "next_offset": offset + len(candidates) if offset + len(candidates) < total else None,
            }
    except SourceError as error:
        return {"ok": False, "error": error.code, "complete": False}
    except (OSError, sqlite3.Error):
        return {"ok": False, "error": "index_unavailable", "complete": False}
    except (ValueError, TypeError, UnicodeError):
        return {"ok": False, "error": "invalid_request", "complete": False}
