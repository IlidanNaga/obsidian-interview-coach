# Obsidian Interview Coach

Rehearse interview answers against your own Markdown notes using a local Ollama model. Questions and feedback cite supplied note text; feedback compares your answer with the notes, rather than giving an objective grade.

**Implemented:** explicit-note and topic interviews through a CLI and reusable JSON API, validated citations, skip, pause/resume, retry, and finish. A topic interview can move between notes and refuses to invent material when no supported chunk is found. **Planned:** whole-Vault coverage (step 4) and usefulness/resource evaluation before a framework comparison (step 5). See the product contract in [BIBLE.md](BIBLE.md) and the [architecture](ARCHITECTURE.md).

## Prerequisites

- Python 3.13 (offline transport checks verified with 3.13.11); application and tests use only the standard library.
- A macOS/POSIX environment supporting `fcntl` locks and the directory/file operations used for persistence.
- Local Ollama with a downloaded model. The default is `qwen3.5:9b`; check that it fits and runs on your machine. For that model, download it once with `ollama pull qwen3.5:9b` if needed.
- A readable UTF-8 `.md` note inside a Vault for explicit-note mode, or a Vault with Markdown notes for topic mode. No Obsidian plugin or special note layout is required. Topic indexing uses Python's bundled SQLite FTS5.

## CLI

Start Ollama with cloud access disabled in a separate terminal (or configure an existing Ollama service equivalently):

```sh
OLLAMA_NO_CLOUD=1 OLLAMA_HOST=127.0.0.1:11434 ollama serve
```

Run the interview from the repository root. These examples use pyenv's installed Python 3.13.11 and disable bytecode writes; replace the generic Vault and note paths with your own.

```sh
export PYENV_VERSION=3.13.11
export PYTHONDONTWRITEBYTECODE=1

pyenv exec python interview.py start \
  --vault "/path/to/vault" \
  --note "topics/cache.md" \
  --endpoint "http://127.0.0.1:11434" \
  --model "qwen3.5:9b"
```

For topic mode, replace `--note` with `--topic`:

```sh
pyenv exec python interview.py start \
  --vault "/path/to/vault" \
  --topic "cache invalidation" \
  --endpoint "http://127.0.0.1:11434" \
  --model "qwen3.5:9b"
```

Topic mode inventories readable Markdown notes in small batches, indexes them privately, searches for candidates, and asks the local model to admit each candidate against the original topic. The first result may say discovery is pending; enter `:retry` to continue. The CLI prints `done N/total` while indexing. A rejected or uncertain candidate produces no question. After feedback, `:next` searches for another supported chunk, including in a different note. There is no required number of questions. An absent-topic result includes suggestions drawn from indexed headings and paths; excluded or unreadable notes cause a distinct `search_incomplete` result instead of a complete refusal.

Each topic session examines at most 32 candidates across literal and expanded queries, including visits deduplicated by the saved decision ledger. Reaching that ceiling returns `search_incomplete` with a candidate-budget message and `complete: false`, not a claim that the topic is absent; excluded-note incompleteness has a separate message. Pause/resume and `:retry` preserve progress and cannot extend the ceiling. You can answer a supported 32nd candidate before further discovery stops, use `:finish`, or start a narrower topic. A complete search with zero exclusions that runs out before 32 can still refuse; reaching exactly 32 conservatively remains incomplete.

The note path may be Vault-relative or absolute, but must resolve inside the Vault. Only explicit HTTP loopback addresses with a port are accepted: `127.0.0.1` or `[::1]`; `localhost` is rejected.

Enter an answer at `>`, or use a command:

| Command | Effect |
| --- | --- |
| `:next` | After feedback, advance to the next chunk in note mode or seek another supported chunk in topic mode. |
| `:skip` | While a question awaits an answer, including a failed assessment, advance once without assessing or saving an answer. Resume a paused session first. |
| `:pause` | Save the current position/question and exit the CLI. |
| `:retry` | Retry a pending/failed question; after failed assessment, clear the block so you can re-enter the answer. |
| `:finish` | Close the session; it cannot be resumed. The saved snapshot remains. |

Use `:skip` explicitly to move past a question you cannot answer. Free text such as “I don't know” remains an answer: if model feedback echoes it, the privacy check still rejects that feedback, and `:skip` lets you continue without `:retry`. Skip clears the current question, feedback and error before the next bounded question/discovery step; a failure there preserves the advanced position for retry. It uses the existing exhausted/incomplete outcomes when material runs out and never extends the 32-candidate ceiling. In other phases, including paused or finished sessions, skip returns `skip_not_expected` without changing state; finish remains available.

Copy the printed session ID to resume; this continues the saved position without automatically repeating inference:

```sh
read -r OIC_SESSION_ID
pyenv exec python interview.py resume "$OIC_SESSION_ID"
```

Paste the ID when `read` waits for input. If you used `--state-dir` at start, pass the same option on resume. Source changes block continuation; finish that session and start a new one for the changed note. After interruption, resume with the ID and use `:retry` if a question or discovery step is pending; answers may need re-entry. Repeating the same topic `start` command resumes an unfinished index scan. Exhaustion leaves the session open until pause or finish.

Resume older sessions by their saved ID: v1/v2 snapshots retain their original 3072-byte request cap, chunk size, cursor, questions and topic index. New sessions use 8192 bytes; starting again with the new limits does not upgrade an old index/session.

## Python / Jupyter

From a Python process or notebook with the repository on its import path, pass dictionaries and receive dictionaries; the CLI alone renders prose.

```python
from interview import start_session, dispatch

current = start_session({
    "vault": "/path/to/vault",
    "note": "topics/cache.md",
    "endpoint": "http://127.0.0.1:11434",
    "model": "qwen3.5:9b",
})
assert current["ok"], current
session_id = current["session_id"]  # Keep this ID for another process.
print(current["question"])

feedback = dispatch({
    "session_id": session_id,
    "action": "answer",
    "answer": "A cache stores reusable results.",  # Invented example answer.
})
print(feedback)
paused = dispatch({"session_id": session_id, "action": "pause"})
```

For topic mode, pass `"topic": "cache invalidation"` in place of `"note"`. If the result has `phase` `indexing`, `discovery`, or `expansion`, call `dispatch({"session_id": session_id, "action": "retry"})` to advance one bounded step. Topic results also include `outcome`, index `scan` counts, `complete`, and real-note `suggestions` when search ends without enough evidence.

To move past a pending question without an assessment, use `dispatch({"session_id": session_id, "action": "skip"})`; no `answer` field is needed, and any supplied answer is ignored. This also works directly after a blocked failed assessment.

In a later process, import `dispatch` and supply the saved ID:

```python
from interview import dispatch

session_id = input("Saved session ID: ").strip()
current = dispatch({"session_id": session_id, "action": "resume"})
print(current)
finished = dispatch({"session_id": session_id, "action": "finish"})
```

Optional `state_dir` must be included in every call when using custom storage. Successful session views expose `ok`, `error`, `session_id`, `status`, `phase`, `cursor`, `total`, `question`, `feedback`, and `message`; early failures may return only `ok`/`error`. Check results before advancing. Feedback labels are `supported`, `partial`, `not_supported`, or `uncertain`.

## Local state and privacy

The Vault is read-only. State defaults to `~/Library/Application Support/ObsidianInterviewCoach/sessions`, outside the repository and Vault. New state directories use mode `0700`, snapshots/locks `0600`; an existing directory must already be private and owned by the current user.

Snapshots retain source paths, identity/hash, endpoint/model, limits, position, current question/feedback and quoted citations. Topic mode also stores the topic and a private SQLite search index and decision ledger under the state directory. They are sensitive derived data. Raw answers and full transcripts are not stored by this application; a check rejects feedback containing the complete answer verbatim (case-insensitively), but does not guarantee removal of answer fragments or paraphrases. The local runtime receives the selected chunk and, for assessment, the question and answer; its own retention is outside this controller.

The application has no cloud fallback and disables proxies and redirects. The loopback URL alone does not prove the Ollama runtime executes locally: keep cloud disabled in Ollama too. Keep real Vaults, state, indexes, traces, credentials and model files outside Git; [`.gitignore`](.gitignore) is only a backstop.

## Verification and limits

With the Python environment above:

```sh
pyenv exec python ollama_smoke.py --self-check
pyenv exec python -m unittest -v test_interview test_topic_search
pyenv exec python interview.py start --help
pyenv exec python interview.py resume --help
```

The transport self-check is offline and mocked. [Controller tests](test_interview.py) and [index tests](test_topic_search.py) use synthetic notes and mocked inference to check containment, citations, topic admission/refusal, multi-note progression, skip without assessment/answer retention, invalid-phase no-ops, pause/restart recovery, locking and privacy; they create fixtures in ignored `tmp/` and session fixtures in temporary storage outside the repository. They do not test whole-Vault coverage, which is planned.

To check an already-running local model with a synthetic prompt:

```sh
pyenv exec python ollama_smoke.py \
  --endpoint "http://127.0.0.1:11434" --model "qwen3.5:9b" --timeout 120
```

New sessions bound the complete serialized HTTP request body, including schema and options, to 8192 bytes in both controller and transport. The previous 3072-byte cap forced topic mode's worst-case escaping probe to choose 128-byte source chunks, reducing the context available for grounded questions. Synthetic checks confirm that ordinary new topics now keep a 512-byte chunk limit, including the worst-case source escaping reservation. Long or heavily escaped topics/models can still require smaller chunks; oversized answers are rejected without truncation or loss of saved progress.

Topic discovery's cost ceiling is 32 examined candidates and at most 32 successful admission decisions, plus one successful query expansion. Failed-call retries and answer assessments are separate calls; indexing still inventories eligible notes. Offline tests cover the candidate ceiling and durable boundary recovery, not live inference latency or Mac memory use.

The live smoke reports elapsed time and available Ollama timing/token metrics, not peak memory or interview quality. Offline checks do not establish real model usability, citation interpretation, latency, or Mac memory use; the increased request cap and its effect on question quality need local live/manual verification. Source requests are bounded; explicit-note mode still reads the complete selected note into memory. Topic indexing excludes notes above 8 MiB and reports the search as incomplete if any notes were excluded.
