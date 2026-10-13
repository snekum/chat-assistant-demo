"""The one-call greeting (D-041): recognise a PURE greeting without a model.

Owner's constraint: a greeting -- often the first thing a member ever types -- must not cost
more than one model call. In a tool loop that is structurally impossible on the normal path
(any tool forces a second round, and the answer is written by a separate pinned call), so a
pure greeting skips the coordinator entirely and goes straight to the pinned writer, with the
asker's own profile summary and last topic preloaded by plain SQL.

FAIL-SAFE BY CONSTRUCTION. A message qualifies only if EVERY token is greeting vocabulary and
at least one is a core greeting word. Anything else -- "Hi, who knows fintech?", "hi Dana",
a greeting this list has never seen -- takes the normal path: it costs extra calls, never a
wrong answer. The list can be too short; it can never send a question down the greeting path.
"""
from __future__ import annotations

import re

_CORE = frozenset({
    "hi", "hello", "hey", "heya", "hiya", "howdy", "greetings", "yo",
    "morning", "afternoon", "evening",
})
_FILLER = frozenset({
    "good", "there", "all", "everyone", "again", "how", "are", "you", "doing", "is", "it",
    "going", "whats", "up", "sup", "hope", "well", "nice", "to", "see", "back", "and", "ya",
})

# TUNABLE(<=6 tokens, every token greeting vocabulary, >=1 core word; symptom wrong: traces show
#   pure greetings ("hey hey, good morning to you all!") taking the multi-call path -> add the
#   words or raise the cap; a QUESTION on this path is impossible unless a content word is
#   added to the vocabulary, so review additions for that, never the cap.)
MAX_TOKENS = 6


def is_pure_greeting(text: str) -> bool:
    tokens = re.findall(r"[a-z]+", text.lower().replace("'", "").replace("’", ""))
    return (
        0 < len(tokens) <= MAX_TOKENS
        and any(t in _CORE for t in tokens)
        and all(t in _CORE or t in _FILLER for t in tokens)
    )


if __name__ == "__main__":
    yes = ["Hi", "hello!", "Hey there", "Good morning", "hi, how are you?", "Hey, what's up",
           "Good evening all", "hello again"]
    no = ["Hi, who knows fintech?", "hi Dana", "Is there anyone in pharma?", "", "thanks",
          "hey can you help me find a CFO", "How are you", "good"]
    for t in yes:
        assert is_pure_greeting(t), t
    for t in no:
        assert not is_pure_greeting(t), t
    print(f"greeting self-check passed ({len(yes)} greetings, {len(no)} non-greetings)")
