"""Person resolution (D-028, D-033): the ONLY thing that turns a mention into a person_id.

Deterministic lookup against the 268 canonical names, never a model's guess. The stage-2 router
LLM may NOMINATE a mention string; only this module APPOINTS an id (D-028 nominate-vs-appoint).

NEVER AUTO-RESOLVE (G-001/D-016): more than one candidate means CLARIFY, always. That is not
politeness -- it is the cardinal-failure policy. Measured on this corpus, 35% of members (94/268)
cannot be reached unambiguously by first name (the most common one maps to six people), so
clarify sits on the happy path, not in the edge cases. Surnames are far cleaner: only 5.2%
(14/268) collide.

MATCHING IS CASE-INSENSITIVE (D-033), because chat input is unreliably cased -- users type
"brief me on jane". The exception is a measured stop-list: tokens that are ALSO ordinary
English words AND resolve to exactly one person. Those need corroboration (capitalised, or part
of a full-name pair) before they count as a mention.

Why only the uniquely-resolving ones are on that list: a false spot on an AMBIGUOUS token lands
in clarify, which is self-correcting. A false spot on a unique token auto-resolves and the system
confidently discusses the wrong person. The error asymmetry decides the design -- a MISSED name
degrades the query to another flow, a FALSE name is the failure D-009 exists to prevent.

The stop-list is DERIVED, never hard-coded: it is real members' surnames and must not live in a
public source file. build_stop_tokens() computes it from the persons table plus corpus text at
index-build time; this module carries the algorithm only.
"""
from __future__ import annotations

import collections
import re

from contracts import Resolution

# A mention is a word or hyphenated/apostrophised word. Names in this corpus are exactly two
# tokens (verified: all 268), so bigrams are the longest pattern worth trying.
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")

# Asker self-reference. Kept tiny and literal: these resolve to the authenticated member's own
# person_id, which is what makes "what does my profile say?" answerable (memo section 6).
SELF_TERMS = frozenset({"me", "my", "mine", "myself", "i"})

# Words that are capitalised for grammar, not because they are names. Only used to decide
# whether an UNKNOWN capitalised token counts as an out-of-corpus person mention.
_SENTENCE_WORDS = frozenset(
    {"i", "the", "a", "an", "and", "but", "or", "if", "is", "are", "was", "were", "do", "does",
     "did", "can", "could", "should", "would", "who", "what", "when", "where", "why", "how",
     "tell", "give", "show", "brief", "compare", "any", "anyone", "someone", "my", "me"}
)


def normalize(token: str) -> str:
    return token.lower().strip("-'")


def build_stop_tokens(canonical_names: list[str], corpus_texts: list[str]) -> set[str]:
    """Tokens that need corroboration before counting as a mention.

    A token qualifies when BOTH hold:
      1. it appears in the corpus as an ordinary lowercase word (so a user could plausibly type
         it without meaning a person), and
      2. it resolves to exactly ONE member (so nothing downstream would catch a false spot).

    Derived rather than fixed, so it stays correct when the corpus changes -- and so no real
    surname is ever written into source.

    # TUNABLE(corpus lowercase usage as the proxy for QUERY usage; the corpus is business prose
    #         and queries are business prose, but they are not the same distribution. Symptom
    #         wrong: traces show a false spot on a token this list missed -> add it, and if that
    #         recurs, switch the proxy to a general English word list or real query logs.)
    """
    by_token: dict[str, set[str]] = collections.defaultdict(set)
    for name in canonical_names:
        parts = name.split()
        by_token[normalize(parts[0])].add(name)
        by_token[normalize(parts[-1])].add(name)

    lowercase_words: collections.Counter[str] = collections.Counter()
    for text in corpus_texts:
        for word in _WORD.findall(text):
            if word.islower():
                lowercase_words[word] += 1

    return {
        token
        for token, people in by_token.items()
        if lowercase_words.get(token, 0) > 0 and len(people) == 1
    }


class Resolver:
    """Built from the persons table (D-016's resolution anchor). Pure: no DB, no framework, no
    model -- so it unit-tests against a handful of fake people."""

    def __init__(self, persons: list[dict], stop_tokens: frozenset[str] | set[str] = frozenset()):
        self.stop_tokens = frozenset(stop_tokens)
        self.by_full: dict[str, str] = {}
        self.by_token: dict[str, list[str]] = collections.defaultdict(list)
        self.name_of: dict[str, str] = {}
        for p in persons:
            pid, name = p["person_id"], p["canonical_name"]
            self.name_of[pid] = name
            parts = [normalize(t) for t in name.split()]
            self.by_full[" ".join(parts)] = pid
            for token in set(parts):
                self.by_token[token].append(pid)

    def _token_hits(self, token: str, capitalized: bool) -> list[str]:
        """Candidate ids for a single token, applying the stop-list rule."""
        key = normalize(token)
        if key in self.stop_tokens and not capitalized:
            return []  # ordinary word used as an ordinary word
        return self.by_token.get(key, [])

    def resolve(self, query: str) -> Resolution:
        """Spot mentions and look them up. Longest match first: a full-name pair wins over its
        parts, so "grant miller" resolves even though bare lowercase "grant" would not.

        Takes no asker: the asker is authenticated and already on the envelope, so they are
        never resolved. Self-reference sets a flag instead -- see Resolution.self_reference.
        """
        tokens = _WORD.findall(query)
        lowered = [normalize(t) for t in tokens]

        resolved: list[str] = []
        self_reference = False
        candidates: dict[str, list[str]] = {}
        unresolved: list[str] = []
        consumed = [False] * len(tokens)

        # Pass 1 -- full-name pairs. Case-insensitive with no stop-list check: two tokens
        # matching a canonical name in order is corroboration in itself.
        for i in range(len(tokens) - 1):
            pair = f"{lowered[i]} {lowered[i + 1]}"
            pid = self.by_full.get(pair)
            if pid is not None:
                resolved.append(pid)
                consumed[i] = consumed[i + 1] = True

        # Pass 2 -- single tokens not already consumed by a full-name match.
        for i, token in enumerate(tokens):
            if consumed[i]:
                continue
            if lowered[i] in SELF_TERMS:
                self_reference = True
                consumed[i] = True
                continue
            hits = self._token_hits(token, capitalized=token[:1].isupper())
            if len(hits) == 1:
                if hits[0] not in resolved:
                    resolved.append(hits[0])
                consumed[i] = True
            elif len(hits) > 1:
                # NEVER auto-resolve. Recorded as a clarify candidate set (G-001/D-016).
                candidates[token] = hits
                consumed[i] = True

        # Pass 3 -- capitalised runs that matched nothing: people who are not members. Covers
        # both the not-in-corpus subject and the non-subject name mentioned inside a dossier;
        # which of those it is cannot be told from the query, and the distinction is made
        # downstream where the dossier text is available.
        run: list[str] = []
        for i, token in enumerate(tokens):
            unknown = (
                not consumed[i]
                and token[:1].isupper()
                and lowered[i] not in _SENTENCE_WORDS
                and not self.by_token.get(lowered[i])
            )
            if unknown:
                run.append(token)
                continue
            if len(run) >= 2:
                unresolved.append(" ".join(run))
            run = []
        if len(run) >= 2:
            unresolved.append(" ".join(run))

        # Status precedence: ambiguity outranks everything -- one unresolvable reference makes
        # the whole turn a clarify, because answering the resolvable half of a comparison while
        # silently dropping the other is the confidently-wrong shape.
        if candidates:
            status = "ambiguous"
        elif resolved:
            status = "resolved"
        elif unresolved:
            status = "not_found"
        else:
            status = "none_named"

        return Resolution(
            status=status,
            person_ids=resolved,
            candidates=candidates,
            unresolved=unresolved,
            self_reference=self_reference,
        )

    def clarify_question(self, resolution: Resolution) -> str:
        """The clarify turn, written deterministically rather than generated: it is a closed
        list of known people, so a model adds latency and hallucination risk for nothing."""
        parts = []
        for mention, ids in resolution.candidates.items():
            names = [self.name_of[i] for i in ids]
            listed = ", ".join(names[:-1]) + f" or {names[-1]}" if len(names) > 1 else names[0]
            parts.append(f'"{mention}" could be {listed}')
        return " ".join(parts) + " - which did you mean?"
