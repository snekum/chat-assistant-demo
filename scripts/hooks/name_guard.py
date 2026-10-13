"""Pre-push privacy guard: block any push whose new commits contain a real member's name.

The corpus is dossiers on real people, so a member's name in a public file -- a test fixture, a
prompt example copied from live output, a run config carrying a person_id -- re-identifies them
against this project. This guard reads the names at run time from the private, gitignored
data/parsed/persons.jsonl (a guard that listed the names would itself be the leak) and scans
exactly what is about to be pushed: every file version and commit message in the new commits.

Matches, case-insensitively:
  * canonical names, with the parts separated by spaces, hyphens, underscores or dots, or split
    across a line break with a comment marker ("Jane\\n# Rivera") -- all seen in this repo;
  * person_ids (real-name slugs such as first-last).

Fails CLOSED: if the name list cannot be loaded, the push is blocked. Bypass deliberately with
`git push --no-verify` only after checking by hand.

Usage (from the pre-push hook, which passes git's stdin):  python scripts/hooks/name_guard.py
Manual:  python scripts/hooks/name_guard.py --range origin/main..HEAD
         python scripts/hooks/name_guard.py --self-test
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True,
                           text=True).stdout.strip() or ".")
PERSONS = ROOT / "data" / "parsed" / "persons.jsonl"
ZERO = "0" * 40


def load_pattern() -> re.Pattern:
    rows = [json.loads(line) for line in PERSONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    alts = set()
    for r in rows:
        for name in {r["canonical_name"], r.get("source_doc_id") or r["canonical_name"]}:
            parts = name.split()
            if len(parts) >= 2:
                alts.add(r"\b" + r"(?:\s+(?:#\s*)?|[-_.])".join(map(re.escape, parts)) + r"\b")
        alts.add(r"\b" + re.escape(r["person_id"]) + r"\b")
    return re.compile("|".join(sorted(alts, key=len, reverse=True)), re.IGNORECASE)


def git(*args: str, data: bytes | None = None) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, input=data, capture_output=True, check=True).stdout


def ranges_from_stdin() -> list[list[str]]:
    """git feeds one line per ref: <local ref> <local sha> <remote ref> <remote sha>."""
    out = []
    for line in sys.stdin.read().splitlines():
        parts = line.split()
        if len(parts) != 4 or parts[1] == ZERO:  # deleting a remote ref pushes no content
            continue
        local, remote = parts[1], parts[3]
        out.append([local, f"^{remote}"] if remote != ZERO else [local, "--not", "--remotes"])
    return out


def scan(rev_args: list[str], pattern: re.Pattern) -> list[str]:
    hits: list[str] = []
    # commit messages
    log = git("log", "--format=%H%x00%B%x01", *rev_args).decode("utf-8", "ignore")
    for entry in filter(None, (e.strip() for e in log.split("\x01"))):
        sha, _, msg = entry.partition("\x00")
        for m in pattern.finditer(msg):
            hits.append(f"commit message {sha[:8]}: '{m.group(0)}'")
    # every file version the new commits introduce
    objs = git("rev-list", "--objects", *rev_args).decode("utf-8", "ignore").splitlines()
    paths = {o.split()[0]: (o.split(maxsplit=1) + [""])[1] for o in objs}
    if not paths:
        return hits
    kinds = git("cat-file", "--batch-check", data="\n".join(paths).encode()).decode().splitlines()
    blobs = [k.split()[0] for k in kinds if k.split()[1] == "blob"]
    if blobs:
        stream = git("cat-file", "--batch", data="\n".join(blobs).encode())
        pos = 0
        for sha in blobs:
            header_end = stream.index(b"\n", pos)
            size = int(stream[pos:header_end].split()[2])
            body = stream[header_end + 1: header_end + 1 + size].decode("utf-8", "ignore")
            pos = header_end + 1 + size + 1
            for m in pattern.finditer(body):
                line = body.count("\n", 0, m.start()) + 1
                hits.append(f"{paths[sha]}:{line}: '{m.group(0)}'")
    return hits


def main() -> int:
    try:
        pattern = load_pattern()
    except Exception as exc:  # fail closed: no list, no push
        print(f"name-guard: cannot load member names from {PERSONS} ({exc}); push blocked.\n"
              "Restore the file, or bypass deliberately with `git push --no-verify`.")
        return 1

    if "--self-test" in sys.argv:
        row = json.loads(PERSONS.read_text(encoding="utf-8").splitlines()[0])
        first, *_, last = row["canonical_name"].split()
        for probe in (f"met {first} {last} today", f"{first.lower()}-{last.lower()}",
                      f"[{first}\n# {last}]", row["person_id"]):
            assert pattern.search(probe), "self-test: a name form was not caught"
        assert not pattern.search("Jane Rivera and Dana Whitfield"), "self-test: placeholder flagged"
        print("name-guard self-test passed (4 name forms caught, placeholders pass)")
        return 0

    if "--range" in sys.argv:
        ranges = [[sys.argv[sys.argv.index("--range") + 1]]]
    else:
        ranges = ranges_from_stdin()

    hits = [h for r in ranges for h in scan(r, pattern)]
    if hits:
        print(f"name-guard: push BLOCKED -- {len(hits)} real member name(s) in the commits being pushed:")
        for h in hits[:30]:
            print("   ", h)
        print("Replace them with placeholders (Jane Rivera, Dana Whitfield) and amend or add a "
              "commit. If a name is already in an earlier unpushed commit, it must be rewritten "
              "out of that commit, not just fixed on top.")
        return 1
    print(f"name-guard: clean ({sum(1 for _ in ranges)} ref(s) checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
