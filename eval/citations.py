"""Citation validity for the synth-v1 contract (D-020 instrument, re-pointed at Step 5).

run.py's parser reads f6-v1's bare [Full Name] tags. synth-v1 (core rules 5-6) emits four kinds,
and the old parser would score every one of them FABRICATED -- "[profile: X]" is not a doc_id,
and "[self]" / "[stated]" name nobody. This module reads the new forms against the TURN'S
ENVELOPE (env.evidence), not a retrieved-list: under the tool loop, what the writer was shown is
whatever the tools returned, and the envelope is the single record of that.

Validity per kind (each a deterministic check, no model):
  [profile: Name]    valid iff Name is the doc_id of some CorpusEvidence this turn.
  [web: Pub, date]   valid iff some WebEvidence this turn has Pub in its title or url.
  [self]             valid iff the turn holds CorpusEvidence for the ASKER -- a [self] tag with
                     no asker profile retrieved is a profile fact from nowhere.
  [stated]           counted, never scored fabricated: UserStatedEvidence is recorded every turn,
                     so "was something stated" is always yes. WHETHER the sentence matches what
                     was stated is a content question -- the groundedness judge's job, which is
                     why the stated text rides on the envelope at all.
  bare [Name]        the f6-v1 form, drifted into. Scored like [profile: Name] so a format slip
                     never masquerades as a fabrication, but counted as off-contract so drift
                     is visible.
Other brackets ("[1]", "[Section 5]") are dropped as non-citations, as run.py already does.

# TUNABLE([stated] is unverifiable-by-design here, not "valid". Revisit when: the judge's
#   groundedness reasons cite a mismatched "as you mentioned" claim -- then add a
#   quote-overlap check against UserStatedEvidence.text.)
"""
from __future__ import annotations

import re

import normalize

_TAG = re.compile(r"\[([^\]]+)\]")
_SECTION_REF = re.compile(r"^section\s+\d+$", re.I)
_CITE_PREFIX = re.compile(r"^\s*cite\s*:\s*", re.I)


def _norm(s: str) -> str:
    return normalize.normalize_for_match(s)


def parse(answer: str, env) -> dict:
    """-> per-answer citation record. `env` is a contracts.Envelope (duck-typed: .asker and the
    three evidence accessors), so the eval never constructs evidence of its own."""
    corpus = env.corpus_evidence()
    profile_docs = {_norm(e.doc_id) for e in corpus}
    asker_retrieved = any(e.person_id == env.asker for e in corpus)
    web = [(_norm(w.title), _norm(w.url)) for w in env.web_evidence()]

    kinds = {"profile": 0, "web": 0, "self": 0, "stated": 0, "bare": 0}
    fabricated: list[str] = []
    cited_docs: set[str] = set()

    for tag in _TAG.findall(answer):
        body = tag.strip()
        low = body.lower()
        if low == "self":
            kinds["self"] += 1
            if not asker_retrieved:
                fabricated.append("[self] without the asker's profile")
            continue
        if low == "stated":
            kinds["stated"] += 1
            continue
        if low.startswith("profile:"):
            kinds["profile"] += 1
            name = body.split(":", 1)[1].strip()
            if _norm(name) in profile_docs:
                cited_docs.add(_norm(name))
            else:
                fabricated.append(name)
            continue
        if low.startswith("web:"):
            kinds["web"] += 1
            pub = _norm(body.split(":", 1)[1].split(",")[0])
            if not any(pub and (pub in t or pub in u) for t, u in web):
                fabricated.append(body)
            continue
        # bare / legacy forms: split plurals, drop numeric and Section-N pointers
        for raw in re.split(r"[,;]", body):
            name = _CITE_PREFIX.sub("", raw).strip()
            if not name or name.isdigit() or _SECTION_REF.match(name):
                continue
            kinds["bare"] += 1
            if _norm(name) in profile_docs:
                cited_docs.add(_norm(name))
            else:
                fabricated.append(name)

    n = sum(kinds.values())
    return {
        "n_citations": n,
        "n_fabricated": len(fabricated),
        "n_valid": n - kinds["stated"] - len(fabricated),
        "by_kind": kinds,
        "off_contract": kinds["bare"],
        "has_any_citation": n > 0,
        "has_fabricated": bool(fabricated),
        "fabricated": sorted(set(fabricated)),
        # RESIDUE (D-029(c) consequence 3): profiles the writer was shown but never cited.
        # Denominator is corpus evidence only -- user-stated evidence is excluded by
        # construction, since it is the asker's own words and is never a profile to cite.
        "uncited_profiles": sorted(profile_docs - cited_docs - (
            {_norm(e.doc_id) for e in corpus if e.person_id == env.asker})),
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from contracts import CorpusEvidence, Envelope, UserStatedEvidence

    env = Envelope(query="q", asker="p-jane")
    env.evidence += [
        CorpusEvidence(person_id="p-dana", doc_id="Dana Whitfield", chunk_id="c1", text="", score=0.9),
        CorpusEvidence(person_id="p-sam", doc_id="Sam Okafor", chunk_id="c2", text="", score=0.8),
        UserStatedEvidence(text="we met in Lagos", turn=0),
    ]
    r = parse("Dana Whitfield ran ops [profile: Dana Whitfield]. You founded a firm [self]. "
              "As you mentioned, you met in Lagos [stated]. Alex Placeholder too [profile: Alex "
              "Placeholder]. Old form [Dana Whitfield, Section 5]. Per Reuters [web: Reuters, 2026].", env)
    assert r["by_kind"] == {"profile": 2, "web": 1, "self": 1, "stated": 1, "bare": 1}, r
    assert set(r["fabricated"]) == {"Alex Placeholder", "[self] without the asker's profile",
                                    "web: Reuters, 2026"}, r["fabricated"]
    assert r["uncited_profiles"] == [normalize.normalize_for_match("Sam Okafor")], r
    assert r["n_valid"] == 2 and r["off_contract"] == 1
    env.evidence.append(CorpusEvidence(person_id="p-jane", doc_id="Jane Rivera", chunk_id="c3",
                                       text="", score=0.7))
    assert "[self] without the asker's profile" not in parse("You did [self].", env)["fabricated"]
    print("citations self-check passed:", r["by_kind"], "fabricated:", r["fabricated"])
