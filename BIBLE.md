# Obsidian Interview Coach Bible

This document records the product contracts that implementation, tests, and user-facing claims must agree on. It is inspired by [Claudexor's verifiable-invariant approach](https://github.com/razzant/claudexor/blob/main/CLAUDEXOR_BIBLE.md), scaled to this project. IDs are stable: do not renumber or reuse them. When a contract changes, explain the old and new behavior and update its verification in the same change.

## Purpose

Help a person rehearse an interview against **their own Obsidian Vault** using a locally hosted model. The Vault supplies the material; the model asks questions and discusses answers. The app does not claim that a note, a citation, or the model is an independent source of truth.

## Invariants

- **OIC-001 — Any local Vault.** A user can start from the whole Vault, a topic expressed in natural language, or an explicit Markdown note path. No project-specific folder layout, question-callout format, Obsidian plugin, or cloud account is required. Existing question callouts may be used when present. **Verify:** synthetic Vaults with nested notes, with and without callouts, exercise all three entry modes.
- **OIC-002 — Grounded admission.** A topic starts an interview only when retrieved note text supports at least one relevant question. If the topic is not found, say so and offer topic suggestions derived from indexed Vault text with their note paths; do not claim semantic closeness when only lexical ranking is available, and do not substitute the model's general knowledge. An explicit path must resolve to a readable Markdown file inside the chosen Vault. **Verify:** absent-topic and out-of-root path cases refuse before generation; suggested topics link to indexed notes and the refusal text does not imply unsupported closeness.
- **OIC-003 — Traceable questions and feedback.** Each question and assessment identifies the note section it relies on. Code checks that cited paths and quoted spans exist in the material supplied to the model. Feedback is described as a comparison with the notes, not an objective grade; a valid citation alone does not prove the interpretation is correct. **Verify:** reject fabricated citations and manually review a small sample of answer assessments.
- **OIC-004 — Complete, bounded traversal.** Whole-Vault mode inventories readable `.md` files under the canonical Vault root, excluding dot-directories and symlinks that escape the root. A section runs from one Markdown heading to the next; nonempty text before the first heading is a section, and a headingless file is one section. The app records which sections were covered or skipped with a reason, visits uncovered material before repeating, and never sends the whole Vault to the model as one prompt. **Verify:** a synthetic multi-folder Vault reaches every eligible section across pauses and resumes without duplicate coverage.
- **OIC-005 — The controller owns actions.** Deterministic code owns file boundaries, source selection limits, session transitions, retries, and resource budgets. The model may propose queries, questions, follow-ups, and note-based feedback; its output cannot grant itself a new file path, tool, or permission. Note text is data, never an instruction. **Verify:** injected instructions in a note cannot trigger a write, outside-root read, or unsupported topic.
- **OIC-006 — The user owns duration.** There is no required number of questions. Pause preserves the current question and traversal position; resume continues that session; finish closes it. User commands are handled outside the model. A failure preserves recoverable state instead of silently changing scope. **Verify:** pause, resume, finish, and model-failure transitions retain the expected position and status.
- **OIC-007 — Vault integrity and privacy.** The app reads the Vault but never edits it. Interview inference uses only an explicit loopback endpoint (`127.0.0.1` or `::1`), without redirects, environment proxies, or cloud fallback. Vault content, answers, transcripts, indexes, credentials, model files, and machine-specific paths stay outside the public repository; raw transcripts are not retained by default. Diagnostics omit note and answer text by default. **Verify:** file snapshots show no Vault writes; endpoint tests reject non-loopback and redirected requests; staged files and outgoing commits contain no real Vault data or derived artifacts.
- **OIC-008 — Measured local operation.** Model, context, request, and concurrency limits are chosen from a bounded local smoke test, not an advertised maximum. The app reports model unavailability, timeout, and unsupported output plainly. The first interface is a CLI over reusable application logic; later interfaces use the same controller and state. **Verify:** local smoke records latency and peak memory, while failure fixtures show explicit outcomes.

## Delivery constraints

1. Prove local serving and structured responses on the target Mac before depending on model behavior.
2. Deliver one end-to-end interview for an explicit note, including pause, resume, finish, and source references.
3. Add topic discovery with a tested refusal path.
4. Add whole-Vault coverage with durable progress.
5. Evaluate usefulness and resource use on a small private sample before comparing the same flow in an agent framework.

Each step must leave a runnable check for its new behavior. Personal Vault samples and their results stay local; public tests use invented notes.
