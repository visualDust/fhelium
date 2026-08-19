#!/usr/bin/env python3
"""Validate the project version and release tag used by release automation."""

from __future__ import annotations

import argparse
import re
import subprocess
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--remote")
    return parser.parse_args()


def git(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def remote_tag_commit(remote: str, tag: str) -> str:
    ref = f"refs/tags/{tag}"
    lines = git("ls-remote", remote, ref, f"{ref}^{{}}").splitlines()
    direct = [line.split()[0] for line in lines if line.split()[1] == ref]
    peeled = [
        line.split()[0] for line in lines if line.split()[1] == f"{ref}^{{}}"
    ]
    if len(direct) != 1 or len(peeled) > 1:
        raise ValueError(f"remote release tag is missing or ambiguous: {tag}")
    return peeled[0] if peeled else direct[0]


def main() -> None:
    args = parse_args()
    project = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    expected = f"v{project['version']}"
    if args.tag != expected:
        raise ValueError(
            f"release tag must equal the project version: "
            f"expected={expected!r}, actual={args.tag!r}"
        )
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    versions = re.findall(r"^version:\s*(\S+)\s*$", citation, re.MULTILINE)
    if versions != [project["version"]]:
        raise ValueError("CITATION.cff version must equal project.version")
    if args.source_commit is not None:
        if re.fullmatch(r"[0-9a-f]{40}", args.source_commit) is None:
            raise ValueError("--source-commit must be a full Git commit ID")
        head = git("rev-parse", "HEAD")
        if head != args.source_commit:
            raise ValueError(
                "checked-out commit differs from --source-commit: "
                f"expected={args.source_commit}, actual={head}"
            )
        local_tag_commit = git("rev-parse", f"{args.tag}^{{commit}}")
        if local_tag_commit != args.source_commit:
            raise ValueError(
                "release tag differs from --source-commit: "
                f"tag={local_tag_commit}, expected={args.source_commit}"
            )
        if args.remote is not None:
            remote_commit = remote_tag_commit(args.remote, args.tag)
            if remote_commit != args.source_commit:
                raise ValueError(
                    "remote release tag differs from --source-commit: "
                    f"tag={remote_commit}, expected={args.source_commit}"
                )
    print(project["version"])


if __name__ == "__main__":
    main()
