# Codex runtime bootstrap

Top-level user-facing main: work directly; agent orchestration OFF by default — invoke `codex-role` and read `references/orchestrator.md` only when user explicitly requests delegation/background tasks.
Delegated background task: invoke `codex-role` and read only role named in dispatch prompt; never read orchestrator.
Prefix shell commands with `rtk`. Use `rtk proxy` only for raw/debug output.
Be terse without losing technical accuracy. Remove filler, hedging, and repetition; preserve exact terms, errors, certainty/uncertainty, and full clarity for risk. Ask whether work is needed; reuse existing code, standard library, native features, then installed dependencies. Prefer deletion and the smallest root-cause diff; avoid speculative abstractions. Never remove validation, security, error handling, accessibility, or physical calibration. Give nontrivial logic one smallest useful check. Decision first.
