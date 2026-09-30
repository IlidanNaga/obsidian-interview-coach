# Obsidian Interview Coach

Rehearse interview answers against your own Markdown notes using a local Ollama model. Questions and feedback cite supplied note text; feedback compares your answer with the notes, rather than giving an objective grade.

**Implemented:** an explicit-note CLI and reusable JSON API, sequential note chunks, validated citations, pause/resume, retry, and finish. **Planned:** natural-language topic discovery with absent-topic refusal (step 3), whole-Vault coverage (step 4), and usefulness/resource evaluation before a framework comparison (step 5). See the product contract in [BIBLE.md](BIBLE.md) and the [architecture](ARCHITECTURE.md).

## Prerequisites

- Python 3.13 (offline transport checks verified with 3.13.11); application and tests use only the standard library.
- A macOS/POSIX environment supporting `fcntl` locks and the directory/file operations used for persistence.
- Local Ollama with a downloaded model. The default is `qwen3.5:9b`; check that it fits and runs on your machine. For that model, download it once with `ollama pull qwen3.5:9b` if needed.
- A readable UTF-8 `.md` note inside a Vault. No Obsidian plugin or special note layout is required.

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

The note path may be Vault-relative or absolute, but must resolve inside the Vault. Only explicit HTTP loopback addresses with a port are accepted: `127.0.0.1` or `[::1]`; `localhost` is rejected.

Enter an answer at `>`, or use a command:

| Command | Effect |
| --- | --- |
| `:next` | After feedback, advance to the next chunk and request its question. |
| `:pause` | Save the current position/question and exit the CLI. |
| `:retry` | Retry a pending/failed question; after failed assessment, clear the block so you can re-enter the answer. |
| `:finish` | Close the session; it cannot be resumed. The saved snapshot remains. |

Copy the printed session ID to resume; this continues the saved position without automatically repeating inference:

```sh
read -r OIC_SESSION_ID
pyenv exec python interview.py resume "$OIC_SESSION_ID"
```

Paste the ID when `read` waits for input. If you used `--state-dir` at start, pass the same option on resume. Source changes block continuation; finish that session and start a new one for the changed note. After interruption, resume with the ID and use `:retry` if a question is pending; answers may need re-entry. Exhaustion leaves the session open until pause or finish.

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

Snapshots retain source paths, identity/hash, endpoint/model, limits, position, current question/feedback and quoted citations. They are sensitive derived data. Raw answers and full transcripts are not stored by this application; a check rejects feedback containing the complete answer verbatim (case-insensitively), but does not guarantee removal of answer fragments or paraphrases. The local runtime receives the selected chunk and, for assessment, the question and answer; its own retention is outside this controller.

The application has no cloud fallback and disables proxies and redirects. The loopback URL alone does not prove the Ollama runtime executes locally: keep cloud disabled in Ollama too. Keep real Vaults, state, indexes, traces, credentials and model files outside Git; [`.gitignore`](.gitignore) is only a backstop.

## Verification and limits

With the Python environment above:

```sh
pyenv exec python ollama_smoke.py --self-check
pyenv exec python -m unittest -v test_interview
pyenv exec python interview.py start --help
pyenv exec python interview.py resume --help
```

The transport self-check is offline and mocked. [Controller tests](test_interview.py) use synthetic notes and mocked inference to check containment, citations, traversal within one note, state recovery, locking and privacy; they create fixtures in ignored `tmp/` and session fixtures in temporary storage outside the repository. They do not test topic admission or whole-Vault traversal, which are planned.

To check an already-running local model with a synthetic prompt:

```sh
pyenv exec python ollama_smoke.py \
  --endpoint "http://127.0.0.1:11434" --model "qwen3.5:9b" --timeout 120
```

The live smoke reports elapsed time and available Ollama timing/token metrics, not peak memory or interview quality. Offline checks do not establish real model usability, citation interpretation, latency, or Mac memory use; those need local live/manual verification. Source requests are bounded, but loading/chunking reads the complete selected note into memory.
