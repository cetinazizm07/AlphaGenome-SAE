Scoped bridge reviews discard caller-provided constraints. Focused sparse-mapping tests and JSON validation checks passed; the full suite could not run because the project interpreter and required dependencies were unavailable.

Review comment:

- [P2] Preserve custom instructions in scoped reviews — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/scripts/codex_bridge.py:140-145
  When `--instructions` accompanies `--commit` or `--base`, this branch drops the instructions entirely and runs a generic scoped review. Requested review constraints, such as security-only scope or a required output format, are therefore ignored despite a successful exit status. Preserve the instructions through a supported invocation, or explicitly reject the combination instead of completing a different review.