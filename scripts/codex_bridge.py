#!/usr/bin/env python3
"""
OpenAI Codex CLI Bridge for Antigravity & AlphaGenome-SAE.

Provides a structured, programmatic interface to invoke OpenAI Codex CLI
for code review, second opinion, refactoring, and automated diagnostics.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


DEFAULT_CODEX_PATH = Path.home() / ".local" / "bin" / "codex"


def resolve_codex_bin() -> str:
    """Find the Codex CLI executable path."""
    if DEFAULT_CODEX_PATH.exists() and os.access(DEFAULT_CODEX_PATH, os.X_OK):
        return str(DEFAULT_CODEX_PATH)
    found = shutil.which("codex")
    if found:
        return found
    raise FileNotFoundError(
        f"Codex CLI executable not found at '{DEFAULT_CODEX_PATH}' or in PATH."
    )


def filter_codex_stderr(stderr_text: str) -> str:
    """Filter out non-critical system sandbox/PATH warnings from Codex stderr."""
    cleaned_lines = []
    for line in stderr_text.splitlines():
        if "could not create PATH aliases" in line:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def run_doctor(codex_bin: str) -> int:
    """Run `codex doctor` to verify environment and authentication."""
    cmd = [codex_bin, "doctor"]
    print(f"[*] Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, text=True)
    return proc.returncode


def run_exec(
    codex_bin: str,
    prompt: str,
    cd: Path,
    sandbox: str = "read-only",
    model: str | None = None,
    output_file: Path | None = None,
) -> int:
    """Execute instructions non-interactively via `codex exec`."""
    is_temp = False
    if output_file:
        effective_out = output_file
    else:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_out:
            effective_out = Path(tmp_out.name)
            is_temp = True

    try:
        cmd = [
            codex_bin,
            "-C",
            str(cd.resolve()),
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            sandbox,
            "-o",
            str(effective_out.resolve()),
        ]
        if model:
            cmd.extend(["-m", model])
        cmd.append(prompt)

        print(f"[*] Invoking Codex Exec in '{cd}' [sandbox: {sandbox}]...")
        proc = subprocess.run(cmd, capture_output=True, text=True)

        err = filter_codex_stderr(proc.stderr)
        if err:
            print(f"[Codex Stderr]\n{err}", file=sys.stderr)

        if effective_out.exists():
            content = effective_out.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                print("\n=== Codex Response ===")
                print(content)
                print("======================\n")
                if output_file:
                    print(f"[+] Output saved to: {output_file.resolve()}")
            else:
                if proc.stdout.strip():
                    print(f"\n[Codex Stdout]\n{proc.stdout.strip()}\n")
        else:
            if proc.stdout.strip():
                print(f"\n[Codex Stdout]\n{proc.stdout.strip()}\n")

        return proc.returncode
    finally:
        if is_temp and effective_out.exists():
            effective_out.unlink(missing_ok=True)


def run_review(
    codex_bin: str,
    instructions: str | None,
    cd: Path,
    model: str | None = None,
    uncommitted: bool = False,
    commit: str | None = None,
    base: str | None = None,
    output_file: Path | None = None,
) -> int:
    """Run an automated code review via `codex review`."""
    cmd = [codex_bin, "-C", str(cd.resolve())]
    if model:
        cmd.extend(["-m", model])
    cmd.append("review")
    if commit:
        cmd.extend(["--commit", commit])
    elif base:
        cmd.extend(["--base", base])
    elif uncommitted and not instructions:
        cmd.append("--uncommitted")
    elif uncommitted and instructions:
        # In codex-cli v0.157.1, `--uncommitted` and `[PROMPT]` are mutually exclusive in clap.
        # Passing instructions alone automatically reviews current working tree changes.
        pass

    if instructions:
        cmd.append(instructions)

    print(f"[*] Invoking Codex Review in '{cd}'...")
    proc = subprocess.run(cmd, capture_output=True, text=True)

    err = filter_codex_stderr(proc.stderr)
    if err:
        print(f"[Codex Stderr]\n{err}", file=sys.stderr)

    out = proc.stdout.strip()
    if out:
        print("\n=== Codex Review Summary ===")
        print(out)
        print("============================\n")
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(out, encoding="utf-8")
            print(f"[+] Review saved to: {output_file.resolve()}")

    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Codex CLI Bridge for Antigravity & AlphaGenome-SAE"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Doctor subcommand
    subparsers.add_parser("doctor", help="Check Codex CLI health and auth")

    # Exec subcommand
    exec_parser = subparsers.add_parser("exec", help="Run Codex non-interactively")
    exec_parser.add_argument("prompt", type=str, help="Instructions / prompt for Codex")
    exec_parser.add_argument(
        "--cd",
        type=Path,
        default=Path.cwd(),
        help="Working root directory (defaults to current dir)",
    )
    exec_parser.add_argument(
        "--sandbox",
        type=str,
        choices=["read-only", "workspace-write", "danger-full-access"],
        default="read-only",
        help="Sandbox execution policy (default: read-only)",
    )
    exec_parser.add_argument("-m", "--model", type=str, default=None, help="Model override")
    exec_parser.add_argument("-o", "--output", type=Path, default=None, help="Save response to file")

    # Review subcommand
    review_parser = subparsers.add_parser("review", help="Run code review non-interactively")
    review_parser.add_argument(
        "instructions",
        type=str,
        nargs="?",
        default=None,
        help="Custom review instructions (positional)",
    )
    review_parser.add_argument(
        "-i",
        "--instructions",
        dest="opt_instructions",
        type=str,
        default=None,
        help="Custom review instructions (flag)",
    )
    review_parser.add_argument(
        "--cd",
        type=Path,
        default=Path.cwd(),
        help="Working root directory (defaults to current dir)",
    )
    review_parser.add_argument(
        "--uncommitted",
        action="store_true",
        help="Review staged, unstaged, and untracked changes",
    )
    review_parser.add_argument("--commit", type=str, default=None, help="Review specific commit SHA")
    review_parser.add_argument("--base", type=str, default=None, help="Review against base branch")
    review_parser.add_argument("-m", "--model", type=str, default=None, help="Model override")
    review_parser.add_argument("-o", "--output", type=Path, default=None, help="Save review to file")

    args = parser.parse_args()

    try:
        codex_bin = resolve_codex_bin()
    except FileNotFoundError as e:
        print(f"[Error] {e}", file=sys.stderr)
        sys.exit(1)

    if args.command == "doctor":
        rc = run_doctor(codex_bin)
    elif args.command == "exec":
        rc = run_exec(
            codex_bin=codex_bin,
            prompt=args.prompt,
            cd=args.cd,
            sandbox=args.sandbox,
            model=args.model,
            output_file=args.output,
        )
    elif args.command == "review":
        effective_instructions = args.opt_instructions or args.instructions
        rc = run_review(
            codex_bin=codex_bin,
            instructions=effective_instructions,
            cd=args.cd,
            model=args.model,
            uncommitted=args.uncommitted,
            commit=args.commit,
            base=args.base,
            output_file=args.output,
        )
    else:
        parser.print_help()
        rc = 1

    sys.exit(rc)


if __name__ == "__main__":
    main()
