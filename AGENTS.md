# Agent instructions

Read [BIBLE.md](BIBLE.md) before changing behavior. It is the product contract; this file is the working procedure. Keep changes within the requested delivery step and surface adjacent findings instead of silently fixing them.

## Planning and implementation

- Keep the initial implementation a local CLI and one controller. Reuse Python's standard library for file traversal, local indexing, and session storage where it meets the contract. Add a dependency or framework only for a demonstrated need.
- Follow the delivery order in `BIBLE.md`. Before moving to the next step, run the smallest meaningful check of the current one; do not treat a green unit test as proof of model quality or Mac resource use.
- Put model-dependent behavior behind a narrow boundary so the hand-written loop and a later framework can run the same scenarios. Do not build multiple engines or a native macOS app in the first pass.
- Public Python entrypoints take and return JSON-shaped dictionaries or lists; a thin CLI owns human-readable output.
- If a change alters an OIC invariant, update the invariant and its verification with the code. Do not weaken a check to make an unsupported claim pass.

## Vault and publication boundaries

- Treat every Vault path and note body as untrusted input. Canonicalize paths before reading, reject escape through `..` or symlinks, and keep the Vault read-only.
- Send inference requests only to an explicit loopback endpoint (`127.0.0.1` or `::1`); disable redirects and environment proxies. A model name does not prove locality. Cloud fallback is an error.
- Keep real Vaults, indexes, session state, answers, traces, model weights, `.env` files, and machine-specific paths outside Git. Synthetic Markdown fixtures are allowed. Inspect staged files and history before every public push; `.gitignore` is only a backstop.
- Use no real Vault content in tests, examples, screenshots, issues, or CI. Run any private smoke locally and report its result without copying note text into the repository.
- Preserve user progress on pause and classified failures. Finish and pause commands must work even when the model proposes another step.

## Verification and claims

- Test path containment, absent-topic refusal, citation validation, traversal coverage, and pause/resume with synthetic Vaults as their implementation lands.
- Verify the local model against the real runtime before claiming it is usable; record observed latency and peak memory without committing prompts or responses from a personal Vault.
- Distinguish a checked source span from a correct assessment of the user's answer. Keep uncertain or unsupported feedback visible as such.
