"""Read one contained Markdown note and derive citations from its chunks."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from contextlib import ExitStack
from pathlib import Path


class SourceError(ValueError):
    """A classified failure, without note contents or machine-specific paths."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _identity(info: os.stat_result) -> dict:
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def _read(root: Path, note: Path, requested: Path) -> tuple[bytes, dict]:
    info = note.stat()
    before = _identity(info)
    if not stat.S_ISREG(info.st_mode):
        raise SourceError("invalid_source")
    root_identity = _identity(root.stat())
    with ExitStack() as stack:
        # Walk the resolved path using directory descriptors: a swapped symlink
        # cannot redirect an intermediate component outside the Vault.
        def open_at(path, flags, directory=None):
            descriptor = os.open(path, flags | os.O_NOFOLLOW, dir_fd=directory)
            stack.callback(os.close, descriptor)
            return descriptor

        directory = open_at(root, os.O_RDONLY | os.O_DIRECTORY)
        if _identity(os.fstat(directory)) != root_identity:
            raise SourceError("source_changed")
        parts = note.relative_to(root).parts
        for part in parts[:-1]:
            directory = open_at(part, os.O_RDONLY | os.O_DIRECTORY, directory)
        descriptor = open_at(parts[-1], os.O_RDONLY | os.O_NONBLOCK, directory)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode) or _identity(os.fstat(descriptor)) != before:
            raise SourceError("source_changed")
        if requested.resolve(strict=True) != note or root.resolve(strict=True) != root:
            raise SourceError("source_changed")
        with os.fdopen(os.dup(descriptor), "rb") as source:
            data = source.read()
        if (
            _identity(os.fstat(descriptor)) != before
            or _identity(note.stat()) != before
            or requested.resolve(strict=True) != note
            or _identity(root.stat()) != root_identity
            or len(data) != before["size"]
        ):
            raise SourceError("source_changed")
    return data, before


def _sections(text: str) -> list[tuple[int, str]]:
    """Return section character offsets and headings; retain all source text."""
    headings = []
    fence = None
    paragraph = None
    offset = 0
    lines = re.findall(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+$", text)
    for line in lines:
        body = line.rstrip("\r\n")
        marker = re.fullmatch(r" {0,3}(`{3,}|~{3,})(.*)", body)
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= fence[1] and not marker[2].strip():
                fence = None
            paragraph = None
        elif marker and (marker[1][0] != "`" or "`" not in marker[2]):
            fence = (marker[1][0], len(marker[1]))
            paragraph = None
        else:
            atx = re.fullmatch(r" {0,3}#{1,6}(?:[ \t]+(.*)|[ \t]*)", body)
            setext = re.fullmatch(r" {0,3}(?:=+|-+)[ \t]*", body)
            if atx:
                heading = re.sub(r"(?:^|[ \t]+)#+[ \t]*$", "", atx[1] or "").strip()
                headings.append((offset, heading))
                paragraph = None
            elif setext and paragraph is not None:
                heading = " ".join(part.strip() for part in re.split(r"\r\n|\r|\n", text[paragraph:offset].strip()))
                headings.append((paragraph, heading))
                paragraph = None
            elif (
                not body.strip()
                or re.fullmatch(r" {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})", body)
                or re.match(r"(?: {4}|\t| {0,3}(?:>|[-+*] |\d+[.)] ))", body)
            ):
                paragraph = None
            elif setext:
                paragraph = None
            elif paragraph is None:
                paragraph = offset
        offset += len(line)
    if not headings:
        return [(0, "")]
    if text[: headings[0][0]].strip():
        return [(0, "")] + headings
    # Leading blank lines belong to the first heading rather than disappearing.
    return [(0, headings[0][1])] + headings[1:]


def _parts(text: str, limit: int):
    """Prefer whole lines; split a long line only between Unicode characters."""
    start = 0
    while start < len(text):
        end = start
        size = 0
        newline = None
        while end < len(text):
            width = len(text[end].encode("utf-8"))
            if size + width > limit:
                break
            size += width
            end += 1
            if text[end - 1] in "\r\n" and not (text[end - 1] == "\r" and text[end : end + 1] == "\n"):
                newline = end
        if end < len(text) and newline is not None:
            end = newline
        elif text[end - 1 : end] == "\r" and text[end : end + 1] == "\n":
            end -= 1
        yield start, end
        start = end


def load_note(request: dict) -> dict:
    """Accept vault/note paths and optional max_chunk_bytes (default 1024).

    Return canonical vault/note paths, a relative path, identity, SHA-256, and
    ordered chunks. Character offsets are zero-based, end-exclusive; line spans
    are one-based, inclusive. The byte limit bounds source text, not HTTP JSON.
    """
    if not isinstance(request, dict) or any(
        not isinstance(request.get(key), str) or not request[key] for key in ("vault", "note")
    ):
        raise SourceError("invalid_request")
    limit = request.get("max_chunk_bytes", 1024)
    if type(limit) is not int or limit < 4:
        raise SourceError("invalid_request")
    try:
        root = Path(request["vault"]).resolve(strict=True)
        requested = Path(request["note"])
        if not requested.is_absolute():
            requested = root / requested
        note = requested.resolve(strict=True)
        if not root.is_dir() or not note.is_relative_to(root) or note.suffix.lower() != ".md":
            raise SourceError("invalid_source")
        data, identity = _read(root, note, requested)
        text = data.decode("utf-8")
    except SourceError:
        raise
    except (OSError, RuntimeError, ValueError, UnicodeError):
        raise SourceError("invalid_source") from None
    digest = hashlib.sha256(data).hexdigest()
    path = note.relative_to(root).as_posix()
    note_id = hashlib.sha256((path + "\0" + digest).encode()).hexdigest()
    sections = _sections(text)
    chunks = []
    line = 1
    for index, (start, heading) in enumerate(sections):
        end = sections[index + 1][0] if index + 1 < len(sections) else len(text)
        section_id = f"{note_id}:{index}"
        section = text[start:end]
        for part, (left, right) in enumerate(_parts(section, limit)):
            content = section[left:right]
            newlines = len(re.findall(r"\r\n|\r|\n", content))
            chunks.append(
                {
                    "id": f"{section_id}:{part}",
                    "section_id": section_id,
                    "path": path,
                    "heading": heading,
                    "start_line": line,
                    "end_line": line + newlines - int(content.endswith(("\r", "\n"))),
                    "start_offset": start + left,
                    "end_offset": start + right,
                    "text": content,
                }
            )
            line += newlines
    return {
        "vault": str(root),
        "note": str(note),
        "path": path,
        "identity": identity,
        "sha256": digest,
        "chunks": chunks,
    }


def cite(chunk: dict, quote: str) -> dict:
    """Validate a nonblank exact, unique quote in a controller-selected chunk."""
    if not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str):
        raise SourceError("invalid_citation")
    if not isinstance(quote, str) or not quote.strip():
        raise SourceError("invalid_citation")
    text = chunk["text"]
    start = text.find(quote)
    if start < 0 or text.find(quote, start + 1) >= 0:
        raise SourceError("invalid_citation")
    try:
        first_line = chunk["start_line"] + len(re.findall(r"\r\n|\r|\n", text[:start]))
        return {
            "path": chunk["path"],
            "heading": chunk["heading"],
            "start_line": first_line,
            "end_line": first_line + len(re.findall(r"\r\n|\r|\n", quote)) - int(quote.endswith(("\r", "\n"))),
            "quote": quote,
        }
    except (KeyError, TypeError):
        raise SourceError("invalid_citation") from None
