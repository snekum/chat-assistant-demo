"""Company aliases (D-034): the one non-name index the resolver earns.

WHEN A DETERMINISTIC INDEX IS JUSTIFIED AT ALL (the rule this module exists under, D-034
amended): semantic search is good at FINDING INFORMATION. It cannot produce an ID to pre-filter
on, and it cannot detect ambiguity -- it always returns its top-k with no signal about whether
two different people could be meant (measured: score ranges for answerable and unanswerable
questions overlap almost entirely, so the score carries no confidence information). So an index
earns its place only when it produces person_ids or detects ambiguity. Anything that is merely
"find information about X" belongs to retrieval, and building a lookup for it is scope creep.

Company aliases pass that test on ONE count, and the narrowing is deliberate (D-035): they
produce person_ids. "Compare <company A> and <company B>" and "brief me on <company>" need
person-scoped retrieval, and global search fetches whichever ranks higher -- the multi-hop
failure Phase 3 exists to fix. The ambiguity leg does NOT apply here: D-035 established that a
company with several members is MULTIPLICITY, not ambiguity, so company mentions never clarify.

For a plain factual question about one company, retrieval alone is sufficient and this index
adds nothing. Nor does it serve "anyone connected to <a company>?" when that company appears
inside dossiers as a vendor, partner or client rather than as a member's own employer -- that is
free-text retrieval across chunks, no index involved. Both boundaries are honest limits, not
oversights.

SCOPED to each member's CURRENT company, from the section-5 header. Prior companies (section 6),
vendors, partners and clients are NOT aliased: "anyone connected to <a vendor named inside
dossiers>" is an aggregate question for retrieval, not a mention to resolve, and aliasing every
company that appears anywhere would trade a precise index for a noisy one. Revisit trigger:
traces show company MENTIONS failing to resolve that a section-6 alias would have caught.

DERIVED, NEVER COMMITTED -- these are real company names, so the same rule as the resolver's
stop-list applies: source carries the algorithm, the repo never carries the list.

HEADER VARIANTS ARE ASSERTED, NOT ASSUMED. Phase 2 learned this the expensive way: one
documented corpus claim was already wrong, and a parser that silently skips documents dilutes
whatever it feeds. build_company_index() reports coverage and the caller asserts on it.
"""
from __future__ import annotations

import collections
import re

# Four observed header phrasings, with or without the section number, and any dash style. None
# of them is documented anywhere; each was found by inspecting what the previous pattern missed
# (194/268 -> 267/268 -> 268/268). Anything below full coverage means the corpus changed, which
# is exactly what the coverage assertion is for.
_COMPANY_HEADER = re.compile(
    r"^\s*#{1,3}\s*(?:\d+\.\s*)?"
    r"(?:Current Company Overview|Current Organization Overview"
    r"|Current Role & Company|Most Recent Executive Role Overview)"
    r"\s*[–—:-]\s*(.+?)\s*$",
    re.M,
)

# A single-word company name would false-match ordinary prose the same way a surname does, so
# it gets the same treatment as the resolver's stop-list.
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
