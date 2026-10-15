"""LLM reranking of member search (experiment arm; NOT wired into find_members).

The web-search cross-encoders (src/rerank.py) judge "does this passage discuss the query"; the
product needs "which person is best for this member". Here a language model -- Haiku 4.5, the
agent's own model; never Haiku 5.5, which grades the answer key -- reads hybrid search's top POOL
candidates (a profile summary + the sections find_members would show first) and returns them in
order. Two variants:

  rules  the answer key's relevance rubric, recast as a ranking instruction (the product's own
         definition of "relevant" -- so agreement with the key is partly built in; flagged)
  plain  no rubric: "most useful to recommend" (diagnostic: how much of any gain is the rubric)

Candidates are shown SHUFFLED (seeded per question) and labelled A..: the model cannot just copy
hybrid's order, and position bias does not favour hybrid's top. People past POOL keep hybrid order.

  python eval/llm_rerank.py smoke              one direct call per variant (latency; prints no names)
  python eval/llm_rerank.py submit             both variants, every scored question (Batch API)
  python eval/llm_rerank.py collect <batch_id> -> analysis/llm_rerank_<variant>_<ts>.jsonl (write-once)

Outputs stay under analysis/ (gitignored): they map candidates to person_ids.
"""
from __future__ import annotations

import json
import random
import string
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]
load_dotenv(ROOT / ".env")
import anthropic  # noqa: E402

MODEL = "claude-haiku-4-5"            # = agent.AGENT_MODEL
PRICE = (1.0, 5.0)                    # $/MTok in, out (standard rate, 2026-10); batch halves both
# TUNABLE(15 candidates: a perfect reorder of hybrid's top 10 / 20 would score 0.772 / 0.876 vs
#   0.565 today, at ~5k input tokens per call. Symptom wrong: clearly-useful people sit at hybrid
#   ranks 16-20 -> widen; per-call cost or latency too high for the shipping form -> narrow.)
POOL = 15
# TUNABLE(excerpt sizes: summary 450 + two sections 380 chars each, ~290 tokens per candidate
#   (trimmed from 500/450 before any run to fit the approved spend).
#   Symptom wrong: the model ranks on the summary alone because the sections are cut mid-fact ->
#   raise the section length.)
SUMMARY_CHARS, SECTION_CHARS, SECTIONS = 450, 380, 2
LETTERS = list(string.ascii_uppercase[:POOL])

BASE = ("You help a private network of business leaders find the right people to talk to. A "
        "member asked the question below. Rank ALL the candidates from most to least useful to "
        "recommend for that question, judging only from the profile excerpts shown.")
END = "Call rank_candidates with every candidate letter exactly once, best first."
TOOL = {"name": "rank_candidates", "description": "Submit the ranking, best candidate first.",
        "strict": True,
        "input_schema": {"type": "object", "additionalProperties": False, "required": ["order"],
                         "properties": {"order": {"type": "array",
                                                  "items": {"type": "string", "enum": LETTERS}}}}}


def system_for(variant: str) -> str:
    if variant == "plain":
        return f"{BASE}\n\n{END}"
    from grade_recommend import RUBRIC   # the answer key's rubric v3, verbatim
    return (f"{BASE}\n\nHow to judge usefulness:\n\n{RUBRIC}\n\nRank every candidate you would "
            f"grade 2 above every 1, and every 1 above every 0; within a grade, the stronger fit "
            f"first.\n\n{END}")


def _clip(text: str, n: int) -> str:
    from rerank import _clean
    text = _clean(text)
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " ..."


def candidate_pools() -> list[dict]:
    """Per scored question: the search text the product sends (D-052), hybrid's full order (top 50
    kept), and the shuffled, lettered top POOL with their excerpts. Deterministic."""
    import store
    from embedder import NomicLocal
    from hybrid import rank_people
    from recommend_eval import load_key, query_of
    from tools import SCHEME

    conn, emb = store.connect(), NomicLocal()
    pools = []
    for q in load_key()[1]:
        text = query_of(q["question"])
        order, sections = rank_people(conn, emb.embed_query(text), text, SCHEME, top=POOL,
                                      per_person=3)
        pool = order[:POOL]
        shown = pool[:]
        random.Random(q["id"]).shuffle(shown)
        blocks = []
        for letter, pid in zip(LETTERS, shown):
            summary = store.fetch_person_chunks(conn, pid, scheme=SCHEME)[0]["text"]
            secs = [_clip(r["text"], SECTION_CHARS) for r in sections[pid][:SECTIONS]]
            blocks.append(f"[{letter}]\nSummary: {_clip(summary, SUMMARY_CHARS)}\n"
                          + "\n".join(f"Relevant: {s}" for s in secs))
        pools.append({"qid": q["id"], "question": q["question"], "search_text": text,
                      "hybrid_top50": order[:50], "letters": dict(zip(LETTERS, shown)),
                      "user": f"Question: {q['question']}\n\nCandidates:\n\n" + "\n\n".join(blocks)})
    return pools


def _params(variant: str, user: str) -> dict:
    return {"model": MODEL, "max_tokens": 300, "system": system_for(variant), "tools": [TOOL],
            "tool_choice": {"type": "tool", "name": "rank_candidates"},
            "messages": [{"role": "user", "content": user}]}


def to_ranking(order: list[str], pool: dict) -> tuple[list[str], int]:
    """Model letters -> person_ids, hybrid's rest appended. Invalid output is repaired, not
    trusted: unknown or repeated letters dropped, missing ones appended in hybrid order."""
    seen, ids = set(), []
    for letter in order:
        pid = pool["letters"].get(letter)
        if pid and pid not in seen:
            seen.add(pid)
            ids.append(pid)
    missing = [p for p in pool["hybrid_top50"][:POOL] if p not in seen]
    return ids + missing + pool["hybrid_top50"][POOL:], len(missing) + (len(order) - len(ids))


def smoke() -> None:
    client, pool = anthropic.Anthropic(), candidate_pools()[0]
    for variant in ("rules", "plain"):
        t0 = time.perf_counter()
        msg = client.messages.create(**_params(variant, pool["user"]))
        took = time.perf_counter() - t0
        order = next(b.input["order"] for b in msg.content if b.type == "tool_use")
        _, repaired = to_ranking(order, pool)
        cost = (msg.usage.input_tokens * PRICE[0] + msg.usage.output_tokens * PRICE[1]) / 1e6
        print(f"{variant}: {took:.2f} s, {msg.usage.input_tokens} in / {msg.usage.output_tokens} out "
              f"tokens, ${cost:.4f}, {len(order)} letters returned, {repaired} repaired")


def submit() -> None:
    pools = candidate_pools()
    reqs = [{"custom_id": f"{p['qid']}__{v}", "params": _params(v, p["user"])}
            for p in pools for v in ("rules", "plain")]
    batch = anthropic.Anthropic().messages.batches.create(requests=reqs)
    plan = ROOT / "analysis" / f"llm_rerank_plan_{batch.id}.json"
    plan.write_text(json.dumps({"batch_id": batch.id, "model": MODEL, "pool": POOL,
                                "pools": {p["qid"]: {k: p[k] for k in ("question", "search_text",
                                          "hybrid_top50", "letters")} for p in pools}},
                               ensure_ascii=False), encoding="utf-8")
    print(f"submitted batch {batch.id}: {len(reqs)} requests (plan -> {plan.relative_to(ROOT)})")


def collect(batch_id: str) -> None:
    client = anthropic.Anthropic()
    while client.messages.batches.retrieve(batch_id).processing_status != "ended":
        time.sleep(30)
    plan = json.loads((ROOT / "analysis" / f"llm_rerank_plan_{batch_id}.json").read_text(
        encoding="utf-8"))
    rows: dict[str, list] = {"rules": [], "plain": []}
    spent, bad = 0.0, []
    for r in client.messages.batches.results(batch_id):
        qid, variant = r.custom_id.split("__")
        use = ([b.input for b in r.result.message.content if b.type == "tool_use"]
               if r.result.type == "succeeded" else [])
        if not use:
            bad.append(r.custom_id)
            continue
        msg = r.result.message
        spent += (msg.usage.input_tokens * PRICE[0] + msg.usage.output_tokens * PRICE[1]) * 0.5 / 1e6
        ranked, repaired = to_ranking(use[0].get("order", []), plan["pools"][qid])
        rows[variant].append({"qid": qid, "question": plan["pools"][qid]["question"],
                              "ranked": ranked[:50], "repaired": repaired,
                              "batch_id": batch_id, "model": MODEL, "pool": plan["pool"]})
    stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    for variant, vrows in rows.items():
        out = ROOT / "analysis" / f"llm_rerank_{variant}_{stamp}.jsonl"
        out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in vrows) + "\n",
                       encoding="utf-8")
        print(f"{variant}: {len(vrows)} questions, {sum(x['repaired'] for x in vrows)} letters "
              f"repaired -> {out.relative_to(ROOT)}")
    print(f"batch {batch_id} ended: {len(bad)} unusable {bad}, ${spent:.4f}")


if __name__ == "__main__":
    {"smoke": smoke, "submit": submit, "collect": lambda: collect(sys.argv[2])}[sys.argv[1]]()
