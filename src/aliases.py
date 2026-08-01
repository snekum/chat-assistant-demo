"""Company aliases and non-subject names (D-034): the two resolver inputs the census asked for.

Both are DERIVED from the corpus at build time and never committed -- they are real company
names and real people's names, so the same rule as the resolver's stop-list applies: source
carries the algorithm, the repo never carries the list.

COMPANY ALIASES answer "Tell me about <company>" / "Anyone connected to <company>?" (census
Tier 1). Scoped DELIBERATELY to each member's CURRENT company, declared in the section-5 header.
Prior companies (section 6) are NOT aliased: "connected to <vendor>" usually means a supplier or
a client relationship, not a person to resolve, and aliasing every prior employer would turn a
precise index into a noisy one. Trigger to revisit: traces show company mentions failing to
resolve that a section-6 alias would have caught.

NON-SUBJECT NAMES answer "Who is <someone named inside a dossier>?" -- people listed as Key
Personnel of a member's company who have no dossier of their own. The resolver must NOT treat
them as members; the honest answer is "mentioned in X's profile as ..., not in the network".
Without this they would fall into the same bucket as a public figure who was never in the corpus
at all, and those two deserve different answers.

HEADER VARIANTS ARE ASSERTED, NOT ASSUMED. Phase 2 learned this the expensive way: one
documented corpus claim was already wrong, and a parser that silently skips documents dilutes
whatever it feeds. build_company_index() reports coverage and the caller asserts on it.
"""
from __future__ import annotations

import collections
import re

# Four observed header phrasings, with or without the section number, and any dash style.
# Measured coverage with all four: 268/268 persons. Anything less means the corpus changed --
# which is exactly what the coverage assertion is for.
_COMPANY_HEADER = re.compile(
    r"^\s*#{1,3}\s*(?:\d+\.\s*)?"
    r"(?:Current Company Overview|Current Organization Overview"
    r"|Current Role & Company|Most Recent Executive Role Overview)"
    r"\s*[–—:-]\s*(.+?)\s*$",
    re.M,
)

# Key Personnel is a labelled bullet at column 0 whose people are bullets INDENTED under it:
#
#   *   **Key Personnel:**
#       *   **A Name:** President & CEO.
#   *   **Strategic Alliances:**          <- indent returns to 0, block ends
#
# Indentation is the only reliable delimiter -- a regex that ends the block at "the next
# **Label:** bullet" ends it on the first person, because those look identical. Found by
# printing the raw lines; the first implementation silently returned zero people.
_LABEL_BULLET = re.compile(r"^\s*\*\s+\*\*(.+?):?\*\*")

# Some Key Personnel bullets name a ROLE where others name a person -- "**GM Travel:**",
# "**Sr. Director HR:**", "**Former Managing Partner:**". They are capitalised two-token
# phrases, so they are indistinguishable from names by shape alone. A role-word veto is the
# cheap discriminator; measured on this corpus it removes 6 of 17 extractions, leaving 11 real
# people.
# TUNABLE(role-word veto list; symptom wrong: a real member of staff whose surname is an
#         ordinary role word gets dropped, or a new role phrasing slips through into the
#         non-subject index -- both visible by eyeballing the extracted list after a re-ingest)
_ROLE_WORDS = frozenset(
    {"gm", "hr", "sr", "jr", "vp", "ceo", "cfo", "coo", "cto", "manager", "managing",
     "management", "partner", "director", "president", "chief", "officer", "vice", "head",
     "counsel", "former", "senior", "junior", "executive", "operations", "finance", "travel",
     "marketing", "sales", "engineering", "founder", "chair", "chairman", "board", "principal"}
)
_PERSON_BULLET = re.compile(r"^\s*\*\s+\*\*([A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){1,2})[:,]?\*\*")

# Company names that are a single ordinary word would false-match ordinary prose the same way
# surnames do, so they get the same treatment as the resolver's stop-list.
_MIN_ALIAS_TOKENS = 2


def normalize_company(name: str) -> str:
    """Fold the punctuation and suffix noise that stops '<X>, Inc.' matching '<X>'."""
    s = name.lower()
    s = re.sub(r"\(.*?\)", " ", s)                      # trailing "(d.b.a. ...)" style notes
    s = re.sub(r"[^a-z0-9&\s]", " ", s)
    s = re.sub(r"\b(inc|llc|ltd|lp|llp|corp|corporation|company|co|pty|plc|gmbh)\b", " ", s)
    return " ".join(s.split())


def build_company_index(chunks: list[dict]) -> tuple[dict[str, list[str]], int]:
    """company (normalized) -> person_ids, plus the number of persons covered.

    Returns coverage rather than asserting internally so the caller decides how strict to be --
    an eval build should refuse to proceed on a regression; an exploratory script may not care.
    """
    per_person: dict[str, set[str]] = collections.defaultdict(set)
    for row in chunks:
        for match in _COMPANY_HEADER.finditer(row["text"]):
            company = normalize_company(match.group(1))
            if company:
                per_person[row["person_id"]].add(company)

    index: dict[str, list[str]] = collections.defaultdict(list)
    for person_id, companies in per_person.items():
        for company in companies:
            index[company].append(person_id)
    return dict(index), len(per_person)


def build_non_subject_index(chunks: list[dict], member_names: set[str]) -> dict[str, list[str]]:
    """person named in a dossier's Key Personnel -> ids of the dossiers naming them.

    Members are excluded: a member listed as their own company's CEO is a member, and resolving
    them through this index instead of the canonical one would lose their person_id.
    """
    members = {n.lower() for n in member_names}
    index: dict[str, list[str]] = collections.defaultdict(list)
    for row in chunks:
        lines = row["text"].splitlines()
        for i, line in enumerate(lines):
            label = _LABEL_BULLET.match(line)
            if not label or "key personnel" not in label.group(1).lower():
                continue
            base = len(line) - len(line.lstrip())
            for follower in lines[i + 1:]:
                if not follower.strip():
                    continue
                if len(follower) - len(follower.lstrip()) <= base:
                    break  # indentation returned to the label level: block over
                match = _PERSON_BULLET.match(follower)
                if not match:
                    continue
                key = " ".join(match.group(1).split()).lower()
                if key in members:
                    continue  # a member listed at their own company resolves canonically
                if any(tok.strip(".") in _ROLE_WORDS for tok in key.split()):
                    continue  # a role label, not a person
                if row["person_id"] not in index[key]:
                    index[key].append(row["person_id"])
    return dict(index)


def alias_stop_terms(index: dict[str, list[str]], corpus_texts: list[str]) -> set[str]:
    """Single-word aliases that are also ordinary lowercase words -- same hazard, same rule as
    the resolver's name stop-list, so they need capitalisation in the query to count."""
    lowercase: collections.Counter[str] = collections.Counter()
    for text in corpus_texts:
        for word in re.findall(r"\b[a-z][a-z'\-]*\b", text):
            lowercase[word] += 1
    return {
        alias
        for alias in index
        if len(alias.split()) < _MIN_ALIAS_TOKENS and lowercase.get(alias, 0) > 0
    }
