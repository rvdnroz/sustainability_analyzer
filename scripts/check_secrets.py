"""Scan local UTF-8 files without printing credential values. No network calls."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", "node_modules"}
PATTERNS = {
    "provider token": re.compile(
        r"(?:AIza[0-9A-Za-z_-]{30,}|gsk_[0-9A-Za-z]{20,}|"
        r"sk-(?:proj-|or-v1-)?[0-9A-Za-z_-]{20,}|"
        r"gh[pousr]_[0-9A-Za-z]{20,}|github_pat_[0-9A-Za-z_]{20,}|"
        r"AKIA[0-9A-Z]{16})"
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "Azure account key": re.compile(r"AccountKey=[A-Za-z0-9+/]{20,}={0,2}", re.I),
    "signed URL": re.compile(r"[?&]sig=[A-Za-z0-9%+/=_-]{16,}", re.I),
    "credential in URL": re.compile(r"https?://[^\s/:@]+:[^\s/@]{8,}@", re.I),
    "literal credential": re.compile(
        r'''\b(?:[a-z0-9_]*(?:api_key|api_token|access_token|auth_token|client_secret|password|connection_string))['"]?\s*[:=]\s*['"]([^'"\r\n]{8,})['"]''',
        re.I,
    ),
    "environment credential": re.compile(
        r"^\s*[A-Z0-9_]*(?:API_KEY|API_TOKEN|ACCESS_TOKEN|CLIENT_SECRET|PASSWORD|CONNECTION_STRING)\s*=\s*([^\s#'\"]{8,})",
    ),
    "Bearer token": re.compile(r"\bBearer\s+[A-Za-z0-9_.-]{20,}"),
}


def placeholder(value: str) -> bool:
    text = value.lower().strip()
    return (
        text.startswith(("your_", "your-", "example", "placeholder", "replace_", "<", "${", "{", "env:"))
        or text in {"changeme", "redacted", "not-a-real-key"}
    )


def scan_text(text: str) -> list[tuple[int, str]]:
    findings = []
    for number, line in enumerate(text.splitlines(), 1):
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(line):
                value = match.group(1) if match.lastindex else match.group()
                if not placeholder(value):
                    findings.append((number, label))
                    break
    return findings


def scan_directory(root: Path) -> list[tuple[Path, int, str]]:
    findings = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        try:
            content = path.read_text(encoding="utf-8-sig")
        except UnicodeError:
            continue  # Binary assets are outside this text scanner's scope.
        findings.extend((path.relative_to(root), number, label) for number, label in scan_text(content))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.path.resolve()
    if not root.is_dir():
        parser.error("path must be a directory")
    findings = scan_directory(root)
    for path, number, label in findings:
        print(f"{path}:{number}: possible {label} (value hidden)")
    print(f"Secret scan: {len(findings)} finding(s). Local text files only; Git history is not scanned.")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
