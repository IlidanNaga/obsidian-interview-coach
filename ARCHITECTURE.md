# Architecture

The current implementation is one explicit-note controller with standard-library source loading, local transport and JSON snapshots. [BIBLE.md](BIBLE.md) defines the broader contract; topic discovery and whole-Vault coverage are still planned. Usage is in [README.md](README.md).

## Modules and flow

| File | Current responsibility |
| --- | --- |
| [interview.py](interview.py) | `start_session(dict)` / `dispatch(dict)` return JSON-shaped results; deterministic controller, request construction, proposal validation, durable state and thin interactive CLI. |
| [vault_source.py](vault_source.py) | `load_note(dict)` resolves/reads one contained UTF-8 Markdown file, identifies sections/chunks and source identity; `cite(chunk, quote)` derives validated references. Source failures raise classified `SourceError`. |
| [ollama_smoke.py](ollama_smoke.py) | `chat(dict)` provides bounded Ollama transport; `smoke(dict)` checks structured output using a synthetic prompt; `--self-check` exercises mocked transport. |
| [test_interview.py](test_interview.py) | Synthetic offline checks of controller transitions, source boundaries, citations and persistence; patches `interview.chat`. |
| [AGENTS.md](AGENTS.md) / [BIBLE.md](BIBLE.md) | Working procedure / product invariants and delivery order. |
| [.gitignore](.gitignore) | Backstop for local artifacts, not a publication/privacy guarantee. |

```text
CLI / Python dictionary
  -> controller -> contained note -> ordered section chunks
  -> save pending session -> selected chunk -> local chat request
  -> validate proposal + quote + unchanged source -> save result -> caller

answer -> selected chunk + current question + ephemeral answer -> chat
       -> validated note-based feedback -> save result -> caller
```

The controller selects the source, advances the cursor, enforces budgets and owns commands/recovery. The model supplies only a question and quote, or a feedback label/text and quote; extra fields are rejected. It cannot choose another file, execute tools, or grant permissions. Each inference sees one chunk; assessment also receives the current question and answer, without conversation history. Temperature zero is a transport setting, not a guarantee of model determinism or correctness.

## Sources and citations

Source paths are resolved before containment checks; the target must be a regular `.md` file inside the canonical Vault. Resolved components are opened through directory descriptors with `O_NOFOLLOW`, with identity checks around the read to detect replacement. Symlinks resolving outside the Vault are rejected. The note is read completely, decoded as UTF-8, and identified by filesystem metadata and SHA-256.

Section parsing recognizes ATX/setext headings and ignores headings inside fenced code; preamble and headingless text are retained. Sections are split into ordered UTF-8 byte-limited chunks, preferring line boundaries and retaining all text. Sessions start with a 512-byte chunk limit and halve it, down to 4 bytes, if needed to fit the complete serialized request budget.

The model must return a nonblank exact quote occurring once in the selected chunk. Code derives the relative path, heading and one-based inclusive line span from that chunk, rather than accepting a model-supplied path. Proposals require exact fields/nonempty strings; questions are limited to 256 UTF-8 bytes and assessment labels to `supported`, `partial`, `not_supported`, `uncertain`. Source identity/hash and saved citations are rechecked on continuation, and the source is rechecked after inference before acknowledging a result. A valid span proves textual presence, not relevance or a correct assessment.

## Bounded local transport

`chat` posts to `/api/chat` only at an explicit `http://127.0.0.1:PORT` or `http://[::1]:PORT` base URL, with no credentials, extra path, query or fragment. Redirects and environment proxies are disabled; the application has no cloud fallback. Ollama's own cloud setting is outside this transport boundary and must be disabled separately for local-only execution.

Requests are non-streaming, with `think=false`, temperature `0`, `num_ctx=4096`, `num_predict=256`, a supplied JSON schema, a maximum 120-second socket timeout, a 3072-byte complete request body and a 256-KiB response cap. The transport rejects incomplete/length-truncated responses, malformed JSON, non-object content and non-finite numbers; the controller validates the actual proposal fields and citations. These are per-call limits, not an overall session duration or process memory quota. The per-session lock prevents concurrent operations on the same session; no global concurrency limiter is implemented.

## Durable transitions and recovery

State version 1 records source metadata, endpoint/model, limits, cursor/total, current question/feedback, error, status and phase. Status is `active`, `blocked`, `paused`, or `finished`; phase tracks work independently:

| Transition | Durable behavior |
| --- | --- |
| Start | Save `need_question` before inference; validated question moves to `await_answer`. |
| Answer | Validated feedback moves to `await_next`; the raw answer is never saved. |
| Next | Only after feedback: advance cursor, clear current results and save `need_question` before inference; at the end save `exhausted`. |
| Pause / resume | Pause remembers the prior active/blocked status without changing phase; resume validates the source and restores status without inference. |
| Retry | Revalidate source, clear the block, save; request a pending question or ask for answer re-entry after failed assessment. |
| Finish | Save `finished`, clear the error and close continuation; retain snapshot/current results. |

Exhaustion does not finish the session automatically. Pause/finish require readable valid state but neither the source nor the model. Model unavailability, timeouts, invalid output/citations and source changes block without advancing pending work. Changed source is not silently adopted: finish and start a new session. Oversized answers are rejected before transport, without truncation or a durable block. Interrupted requests leave the saved phase intact; resume and retry/re-enter the answer as appropriate.

State storage must be outside the repository and Vault, user-owned and private. Operations take a nonblocking `fcntl` session lock (`session_busy` on collision). Snapshots use a private temporary file, file `fsync`, atomic replacement and directory `fsync`, with a 128-KiB size cap. Reads validate schema, ownership, permissions, regular-file/symlink constraints and size. Corrupt snapshots are refused without overwrite; persistence failures are reported, and recovery uses the last readable valid snapshot rather than assuming an in-memory result was saved. The tests cover preservation of an earlier snapshot when replacement fails.

## Privacy and trust boundaries

Vault text and answers are untrusted data. Prompts say so, while deterministic source selection, proposal validation and the absence of model-executable tools enforce the operational boundary. The application opens notes for reading only. Local inference receives the selected source text and ephemeral answer; runtime storage/logging is outside the application's control.

Snapshots retain paths and derived content, including current questions, feedback and source quotes, but no raw answers/full transcript. Full-answer echoes in feedback are rejected case-insensitively; fragments/paraphrases are not comprehensively filtered. Errors use classified codes rather than note/answer text; CLI output intentionally displays the current question, feedback and source reference. Real Vaults and derived artifacts must remain outside Git, regardless of ignore rules.

## Planned extensions: delivery steps 3–5

- **Step 3 — Topic discovery:** extend controller-owned source selection beyond an explicit note, with retrieval grounded in Vault text, absent-topic refusal and nearby topics linked to notes. No topic index/retrieval or refusal flow exists yet.
- **Step 4 — Whole-Vault traversal:** extend source inventory and durable progress across eligible notes/sections, record covered/skipped sections with reasons and visit uncovered material before repeating. The current cursor covers only chunks of one immutable note; it is not a Vault coverage ledger.
- **Step 5 — Evaluation, then comparison:** evaluate usefulness, assessment interpretation, latency and peak memory on a small private sample; only then compare the same flow in an agent framework. The existing `chat(dict)` boundary and JSON controller are reuse points, not multiple implemented engines; the smoke helper supplies elapsed time/Ollama metrics but no peak-memory measurement or quality evaluation.

Offline fixtures demonstrate mechanical behavior, not local model quality or hardware suitability; see [verification and limits](README.md#verification-and-limits).
