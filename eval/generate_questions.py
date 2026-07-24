"""Draft single-hop questions from sampled reports (ROADMAP 1c / DECISIONS D-012).

The supervised-expansion half of the eval ladder: single-hop fact retrieval is the PLATEAUED
type (the owner has learned it), so it's LLM-drafted then human-approved -- never authored
unattended (D-012). Pipeline per candidate:

  sample report -> LLM drafts {question, verbatim quote, answer, fact_type}
    -> DETERMINISTIC gate: quote must appear normalize-then-exact in the doc (normalize.py,
       D-011) -- this is what stops a hallucinated gold span silently laundering a miss
    -> auto-tag depth/section (depth.py) -- captured now, cheap, cannot be backfilled; the
       split-by-difficulty reporting is DEFERRED to the chunking phase (2026-07-24 decision:
       under whole-doc embedding, position is not the difficulty axis, so don't slice yet)
    -> review queue (eval/review_queue.jsonl) for human approve/reject.

Approved rows get merged into questions.jsonl with author="llm-assisted" + their tags.

Model = claude-sonnet-5 for DRAFTING quality. This is an authoring AID (every span is
human-verified), NOT the system-under-test -- the retriever + Haiku answerer are what the
eval measures, so using a stronger model here doesn't contaminate anything.

Usage:  ./.venv/Scripts/python.exe eval/generate_questions.py [n_docs]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "eval")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from depth import depth_of, load_corpus  # noqa: E402
from normalize import quote_hits  # noqa: E402

MODEL = "claude-sonnet-5"  # drafting aid, human-reviewed; not the system-under-test
QUESTIONS = Path("eval/questions.jsonl")
QUEUE = Path("eval/review_queue.jsonl")
SEED = 42  # reproducible sampling; we exclude already-covered docs so batches don't repeat

# TUNABLE(2 drafts/doc keeps each report's questions distinct facts, not rephrasings; revisit
# if the same fact shows up twice per doc -> drop to 1, or raise if reports are fact-dense.)
DRAFTS_PER_DOC = 2
# TUNABLE(6 docs -> ~12 candidates/batch = the roadmap supervision batch size; small enough
# to actually review, big enough to read a reject-rate signal. Revisit once reject-rate is
# stable and low -> larger batches earn less review.)
DEFAULT_BATCH_DOCS = 6

MAX_TOKENS = 1024

SYSTEM = (
    "You are helping build a RETRIEVAL eval set from a professional dossier on a business "
    "leader. Given ONE report, write factual single-hop questions whose answer is a short, "
    "specific span found in THAT report.\n"
    "Hard rules:\n"
    "1. `quote` MUST be copied CHARACTER-FOR-CHARACTER from the report (a short span, ideally "
    "<=8 words) -- it is validated by exact containment and rejected if it isn't verbatim. Do "
    "NOT paraphrase, summarize, or fix punctuation in the quote.\n"
    "2. The question must be answerable from this report ALONE, and the `quote` must be the "
    "evidence that answers it.\n"
    "3. Vary specificity ACROSS your questions: some about central facts (role, industry, "
    "company), some about narrow peripheral details (a number, a place, a date, a named tool). "
    "Do not make them all the easy summary-line facts.\n"
    "4. Do not ask about the person's own name, and avoid yes/no questions.\n"
    "5. `fact_type` labels what the answer span IS."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "quote": {"type": "string"},
                    "answer": {"type": "string"},
                    "fact_type": {
                        "type": "string",
                        "enum": ["number", "date", "place", "name", "org", "title", "other"],
                    },
                },
                "required": ["question", "quote", "answer", "fact_type"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}


def covered_doc_ids() -> set[str]:
    """Docs already used by ANY existing question -- excluded so batches cover new ground."""
    out: set[str] = set()
    for l in QUESTIONS.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        for e in json.loads(l).get("evidence", []):
            out.add(e["doc_id"])
    return out


def draft(client, doc_id: str, text: str) -> list[dict]:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        thinking={"type": "disabled"},
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content":
                   f"Report on {doc_id}:\n\n{text}\n\nWrite {DRAFTS_PER_DOC} questions."}],
    )
    payload = json.loads(next(b.text for b in resp.content if b.type == "text"))
    return payload["questions"]


def main() -> None:
    import anthropic

    n_docs = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BATCH_DOCS
    corpus = load_corpus()
    pool = sorted(set(corpus) - covered_doc_ids())
    rng = random.Random(SEED)
    sample = rng.sample(pool, min(n_docs, len(pool)))
    print(f"corpus={len(corpus)}  uncovered={len(pool)}  sampling {len(sample)} docs\n")

    client = anthropic.Anthropic()
    accepted: list[dict] = []
    n_drafted = 0
    n_bad_quote = 0
    for doc_id in sample:
        for c in draft(client, doc_id, corpus[doc_id]):
            n_drafted += 1
            quote = c["quote"]
            if not quote_hits(quote, corpus[doc_id]):
                n_bad_quote += 1
                print(f"  REJECT (quote not verbatim) [{doc_id}] {quote[:50]!r}")
                continue
            tags = depth_of(quote, corpus[doc_id]) or {}
            tags["fact_type"] = c["fact_type"]
            accepted.append({
                "type": "single-hop",
                "question": c["question"],
                "evidence": [{"doc_id": doc_id, "quote": quote}],
                "gold_answer": c["answer"],
                "author": "llm-assisted",
                "tags": tags,
            })
            print(f"  ok     [{doc_id:<16}] §{tags.get('section')} {tags.get('bucket'):<6} "
                  f"{c['fact_type']:<6} {c['question'][:52]}")

    QUEUE.write_text("\n".join(json.dumps(a, ensure_ascii=False) for a in accepted) + "\n",
                     encoding="utf-8")
    auto_reject = n_bad_quote / n_drafted if n_drafted else 0
    print(f"\ndrafted={n_drafted}  auto-rejected(bad quote)={n_bad_quote} ({auto_reject:.0%})  "
          f"queued for review={len(accepted)}")
    print(f"-> {QUEUE}  (review, then merge approved rows into {QUESTIONS})")


if __name__ == "__main__":
    main()
