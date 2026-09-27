---
name: codex-assistant
description: >-
  Use this skill when you need to consult OpenAI Codex CLI, run automated code reviews (`codex review`), get a second opinion on algorithms, or perform cross-validation and pair programming using the local Codex CLI.
---

# OpenAI Codex CLI Skill

This skill allows Antigravity to interact with the local **OpenAI Codex CLI** (`/Users/azizcetin/.local/bin/codex`), providing Dual-LLM peer reviews, architectural second opinions, and automated code inspection.

## Key Capabilities

1. **Non-Interactive Execution (`codex exec`)**:
   Query Codex with structured instructions, run second-opinion evaluations, or ask for code generation.
2. **Automated Code Review (`codex review`)**:
   Perform automated code reviews on uncommitted changes (`--uncommitted`) or specific commits (`--commit <SHA>`).
3. **Sandbox Policies**:
   - `read-only` (default & safest): Codex can read files but cannot modify the workspace.
   - `workspace-write`: Allows Codex to modify workspace files if refactoring is requested.

## Bridge Helper Tool

A dedicated wrapper script is available at:
`scripts/codex_bridge.py`

### Common Workflows

#### 1. Code Review on Current Changes
To review unstaged or staged changes in the Git tree:
```bash
# Automated review of working tree:
python3 scripts/codex_bridge.py review --uncommitted --cd /Users/azizcetin/PycharmProjects/AlphaGenome-SAE

# Or with custom review focus:
python3 scripts/codex_bridge.py review --instructions "Focus on PyTorch tensor contiguousness, memory leaks, and statistical test edge cases." --cd /Users/azizcetin/PycharmProjects/AlphaGenome-SAE
```

#### 2. Review a Specific Commit
```bash
python3 scripts/codex_bridge.py review --commit dcf80f8 --cd /Users/azizcetin/PycharmProjects/AlphaGenome-SAE
```

#### 3. Second Opinion on Code or Algorithm
To query Codex for architectural advice without allowing file edits:
```bash
python3 scripts/codex_bridge.py exec "Review the following residual stream hook in AlphaGenome. Is there any risk of gradient retention or memory buildup during long 40-window loops?" --sandbox read-only --cd /Users/azizcetin/PycharmProjects/AlphaGenome-SAE
```

#### 4. Environment & Session Health Check
```bash
python3 scripts/codex_bridge.py doctor
```

## Security & Execution Guidelines

- **BypassSandbox Requirement:** Because Codex CLI communicates with OpenAI APIs and establishes local IPC sockets, running commands that invoke Codex requires `BypassSandbox: true` when executing via `run_command`.
- **Default to Read-Only:** Never use `workspace-write` unless the user explicitly requested that Codex directly edit repository files.
