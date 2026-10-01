# Architecture

The current implementation supports explicit-note and topic interviews with standard-library source loading, local transport and JSON snapshots. [BIBLE.md](BIBLE.md) defines the broader contract; whole-Vault coverage is still planned. Usage is in [README.md](README.md).

## Modules and flow

| File | Current responsibility |
| --- | --- |
| [interview.py](interview.py) | `start_session(dict)` / `dispatch(dict)` return JSON-shaped results; deterministic note/topic controller, request construction, proposal validation, durable state and thin interactive CLI. |
| [vault_source.py](vault_source.py) | `load_note(dict)` resolves/reads one contained UTF-8 Markdown file, identifies sections/chunks and source identity; `cite(chunk, quote)` derives validated references. Source failures raise classified `SourceError`. |
| [topic_search.py](topic_search.py) | Private resumable SQLite FTS5 index, contained Vault inventory, candidate search and suggestions from actual notes. |
| [ollama_smoke.py](ollama_smoke.py) | `chat(dict)` provides bounded Ollama transport; `smoke(dict)` checks structured output using a synthetic prompt; `--self-check` exercises mocked transport. |
| [test_interview.py](test_interview.py) / [test_topic_search.py](test_topic_search.py) | Synthetic offline checks of controller transitions, topic admission/refusal, source boundaries, indexing, citations and persistence. |
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

The controller selects the source, advances the cursor, enforces budgets and owns commands/recovery. The model supplies only a question and quote, a topic admission decision, retrieval terms, or a feedback label/text and quote; extra fields are rejected. It cannot choose another file, execute tools, or grant permissions. Question/admission and assessment inference see one chunk; assessment also receives the current question and answer, without conversation history. Query expansion sees only the topic. Temperature zero is a transport setting, not a guarantee of model determinism or correctness.

In topic mode, the controller inventories and indexes Markdown notes outside the Vault. It uses FTS5 to select candidate chunks, then asks the local model whether the **original topic** is supported by each selected chunk. Only an admitted question with a valid exact quote becomes visible. A rejected or uncertain candidate advances search progress without a question. After feedback, `:next` seeks another supported chunk and may cross a note boundary. One successful query expansion follows if the literal query yields no admitted material; a failed call can be retried. Expansion terms only retrieve candidates and never grant access to paths. The index and a per-session decision ledger live in a private state directory. Full Vault coverage accounting remains separate work.

Topic discovery caps examined candidates at 64 total across literal and expanded queries. The existing v2 cursor counts each examined visit, including a ledger deduplication; no snapshot fields or versions are added. A guard runs before further candidate search or admission inference, and a rejected/uncertain 64th visit immediately ends discovery with phase/error `search_incomplete` and `complete: false`. Its budget message differs from excluded-note incompleteness; neither supports an absence claim. Even exact exhaustion at 64 remains incomplete without a further completeness proof. An admitted 64th question can still be answered; subsequent `:next` stops discovery. Retry/pause/resume retain cursor, scan counts, suggestions and decisions without extending the cap; finish remains available. This bounds successful admission decisions to 64, plus at most one successful expansion; failed-call retries and assessments remain separate. A zero-exclusion search ending below the cap can still refuse or exhaust normally.

## Sources and citations

Source paths are resolved before containment checks; the target must be a regular `.md` file inside the canonical Vault. Resolved components are opened through directory descriptors with `O_NOFOLLOW`, with identity checks around the read to detect replacement. Symlinks resolving outside the Vault are rejected. The note is read completely, decoded as UTF-8, and identified by filesystem metadata and SHA-256.

Section parsing recognizes ATX/setext headings and ignores headings inside fenced code; preamble and headingless text are retained. Sections are split into ordered UTF-8 byte-limited chunks, preferring line boundaries and retaining all text. Sessions start with a 512-byte chunk limit and halve it, down to 4 bytes, if needed to fit the complete serialized request budget.

Topic mode reserves worst-case source escaping plus a 256-byte question and short answer before indexing. The old 3072-byte complete-body cap made ordinary topics fall to 128-byte chunks. With the new 8192-byte cap, a synthetic `retrieval augmented generation` topic and `qwen3.5:9b` model fit a 512-NUL-byte source: admission is 4447 bytes and assessment with 256-byte ASCII question/answer is 4972 bytes. Long or heavily escaped topic/model values can still shrink chunks; actual assessment bodies are measured again and oversized answers are refused without truncation.

The model must return a nonblank exact quote occurring once in the selected chunk. Code derives the relative path, heading and one-based inclusive line span from that chunk, rather than accepting a model-supplied path. Proposals require exact fields/nonempty strings; questions are limited to 256 UTF-8 bytes and assessment labels to `supported`, `partial`, `not_supported`, `uncertain`. Source identity/hash and saved citations are rechecked on continuation, and the source is rechecked after inference before acknowledging a result. A valid span proves textual presence, not relevance or a correct assessment.

## Bounded local transport

`chat` posts to `/api/chat` only at an explicit `http://127.0.0.1:PORT` or `http://[::1]:PORT` base URL, with no credentials, extra path, query or fragment. Redirects and environment proxies are disabled; the application has no cloud fallback. Ollama's own cloud setting is outside this transport boundary and must be disabled separately for local-only execution.

Requests are non-streaming, with `think=false`, temperature `0`, `num_ctx=4096`, `num_predict=256`, a supplied JSON schema, a maximum 120-second socket timeout, an 8192-byte complete request body and a 256-KiB response cap. The controller checks each call against the snapshot's stored cap and passes that cap to the transport, which independently enforces it and never permits more than 8192 bytes. Direct transport calls default to 8192 bytes. The transport rejects incomplete/length-truncated responses, malformed JSON, non-object content and non-finite numbers; the controller validates the actual proposal fields and citations. These are per-call limits, not an overall session duration or process memory quota. The per-session lock prevents concurrent operations on the same session; no global concurrency limiter is implemented.

## Durable transitions and recovery

State version 1 records explicit-note source metadata, endpoint/model, limits, cursor/total, current question/feedback, error, status and phase. Status is `active`, `blocked`, `paused`, or `finished`; phase tracks work independently:

| Transition | Durable behavior |
| --- | --- |
| Start | Save `need_question` before inference; validated question moves to `await_answer`. |
| Answer | Validated feedback moves to `await_next`; the raw answer is never saved. |
| Next | Only after feedback: advance cursor, clear current results and save `need_question` before inference; at the end save `exhausted`. |
| Skip | Only in active/blocked `await_answer`: clear question/feedback/error and advance once through the same saved transition as Next, without assessment or answer retention. Paused sessions must resume first; other phases/statuses return `skip_not_expected` without mutation. |
| Pause / resume | Pause remembers the prior active/blocked status without changing phase; resume validates the source and restores status without inference. |
| Retry | Revalidate source, clear the block, save; request a pending question or ask for answer re-entry after failed assessment. |
| Finish | Save `finished`, clear the error and close continuation; retain snapshot/current results. |

Exhaustion does not finish the session automatically. Pause/finish require readable valid state but neither the source nor the model. Model unavailability, timeouts, invalid output/citations and source changes block without advancing pending work. Skip saves the advanced position before next-question/discovery inference, so a failure retries that position rather than the skipped question. In topic mode the admitted candidate already advanced cursor/search offset and has a durable decision; skip clears the current source and enters discovery without counting that candidate again or extending the 64-candidate cap. When no material remains, the existing exhausted/incomplete outcomes apply. Changed source is not silently adopted: finish and start a new session. Oversized answers are rejected before transport, without truncation or a durable block. Interrupted requests leave the saved phase intact; resume and retry/re-enter the answer as appropriate. Neither v1 nor v2 snapshot schemas change.

Topic sessions use a separate strict version-2 snapshot. Their phases include indexing, discovery, expansion, awaiting an answer/next command, refusal, incomplete search and exhaustion. Indexing commits one note at a time and exposes `done N/total`; a repeated start or saved ID can resume an unfinished scan. Candidate decisions are committed before the corresponding snapshot advances, so a crash can replay a decision without repeating an accepted question. The search index is frozen per session after indexing so ranked offsets do not move when another session scans the Vault. A complete refusal means no admitted chunk was found in the indexed candidate stream; excluded files produce `search_incomplete` instead. Neither result proves semantic absence from the user's knowledge.

Both validators accept exactly the old 3072-byte or new 8192-byte stored request cap, with all other limits still checked strictly. Resume by ID keeps that cap and the stored chunk size, cursor, index and current question; it does not rechunk or migrate an existing v1/v2 session. New snapshots store 8192 bytes, and repeated-start index matching includes these limits.

State storage must be outside the repository and Vault, user-owned and private. Operations take a nonblocking `fcntl` session lock (`session_busy` on collision). Snapshots use a private temporary file, file `fsync`, atomic replacement and directory `fsync`, with a 128-KiB size cap. Reads validate schema, ownership, permissions, regular-file/symlink constraints and size. Corrupt snapshots are refused without overwrite; persistence failures are reported, and recovery uses the last readable valid snapshot rather than assuming an in-memory result was saved. The tests cover preservation of an earlier snapshot when replacement fails.

## Privacy and trust boundaries

Vault text and answers are untrusted data. Prompts say so, while deterministic source selection, proposal validation and the absence of model-executable tools enforce the operational boundary. The application opens notes for reading only. Local inference receives the selected source text and ephemeral answer; runtime storage/logging is outside the application's control.

Snapshots retain paths and derived content, including current questions, feedback and source quotes, but no raw answers/full transcript. Full-answer echoes in feedback are rejected case-insensitively; fragments/paraphrases are not comprehensively filtered. Explicit `skip` / `:skip` bypasses assessment, ignores any supplied answer and keeps this privacy check unchanged; free text is never interpreted as skip. Only the subsequent bounded question/discovery may call the model. Errors use classified codes rather than note/answer text; CLI output intentionally displays the current question, feedback and source reference. Real Vaults and derived artifacts must remain outside Git, regardless of ignore rules.

## Planned extensions: delivery steps 4–5

- **Step 4 — Whole-Vault traversal:** add durable coverage across eligible notes/sections, record covered/skipped sections with reasons and visit uncovered material before repeating. Topic-mode candidate progress is query-specific; it is not a Vault coverage ledger.
- **Step 5 — Evaluation, then comparison:** evaluate usefulness, assessment interpretation, latency and peak memory on a small private sample; only then compare the same flow in an agent framework. The existing `chat(dict)` boundary and JSON controller are reuse points, not multiple implemented engines; the smoke helper supplies elapsed time/Ollama metrics but no peak-memory measurement or quality evaluation.

Offline fixtures demonstrate mechanical behavior, not local model quality or hardware suitability; see [verification and limits](README.md#verification-and-limits).
