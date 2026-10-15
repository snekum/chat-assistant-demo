"""What the chat model would put in a find_members call, for every recommendation question
(experiment arm 3, needs split). Two calls per question, both Haiku 4.5 -- the agent's own model --
with the agent's system prompt and find_members forced:

  live   today's tool (criterion only): the search phrase the product actually sends. The
         recommendation eval has always searched with the raw question instead.
  split  an extended tool: criterion + needs (what the person must help with, one phrase each)
         + exclude_topics + asker_context (the asker's own situation, kept out of needs)

Never Haiku 5.5 here: it grades the answer key these arms are scored against.

  python eval/split_needs.py smoke <n>          direct calls, first n questions (+ latency)
  python eval/split_needs.py submit             Batch API, every scored question (half price)
  python eval/split_needs.py collect <batch_id> -> analysis/needs_haiku45_<ts>.jsonl (write-once)

The file carries the questions verbatim, so it stays under analysis/ (gitignored). Prints
progress and cost only.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")
import anthropic  # noqa: E402

from contracts import REFUSAL_STRING  # noqa: E402

MODEL = "claude-haiku-4-5"            # = agent.AGENT_MODEL
PRICE = (1.0, 5.0)                    # $/MTok in, out (standard rate, 2026-10)
QUESTIONS = ROOT / "eval" / "recommend_questions.jsonl"
SYSTEM = (ROOT / "prompts" / "agent" / "core-v1.md").read_text(encoding="utf-8") \
    .replace("\r\n", "\n").strip().format(refusal_string=REFUSAL_STRING)

_DESCRIPTION = ("Discover members of the network who match a criterion (an industry, an "
                "experience, a company, a deal). The member you are speaking with is never "
                "included.")
LIVE_TOOL = {   # mirrors agent.find_members as the model sees it today
    "name": "find_members", "description": _DESCRIPTION, "strict": True,
    "input_schema": {"type": "object", "additionalProperties": False,
                     "required": ["criterion"],
                     "properties": {"criterion": {"type": "string",
                                                  "description": "what to look for, in plain words."}}}}
SPLIT_TOOL = {
    "name": "find_members", "description": _DESCRIPTION, "strict": True,
    "input_schema": {
        "type": "object", "additionalProperties": False,
        "required": ["criterion", "needs", "exclude_topics", "asker_context"],
        "properties": {
            "criterion": {"type": "string", "description": "what to look for, in plain words."},
            "needs": {"type": "array", "items": {"type": "string"}, "description": (
                "Each thing the person must be able to help with, as its own short phrase. "
                "Several items only when the member wants someone who covers several things at "
                "once ('both X and Y', 'X with experience in Y'); one item otherwise. Only what "
                "the person must help with -- never facts about the asker's own situation.")},
            "exclude_topics": {"type": "array", "items": {"type": "string"}, "description": (
                "Areas the member says they do NOT want ('rather than fundraising'). Empty if "
                "none.")},
            "asker_context": {"type": "string", "description": (
                "The asker's own situation if they describe it (industry, stage, place), in a "
                "few words. Empty if none.")},
        }}}


def _params(tool: dict, question: str) -> dict:
    return {"model": MODEL, "max_tokens": 1024, "system": SYSTEM, "tools": [tool],
            "tool_choice": {"type": "tool", "name": "find_members"},
            "messages": [{"role": "user", "content": question}]}


def _questions(n: int | None = None) -> list[dict]:
    sys.path.insert(0, str(ROOT / "eval"))
    from recommend_eval import load_key       # the 34 scored questions, same drop rule
    return load_key()[1][:n]


def _stamp() -> dict:
    return {"model": MODEL, "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest()[:16],
            "split_tool_sha256": hashlib.sha256(json.dumps(SPLIT_TOOL, sort_keys=True)
                                                .encode()).hexdigest()[:16]}


def _out() -> Path:
    return ROOT / "analysis" / f"needs_haiku45_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.jsonl"


def smoke(n: int) -> None:
    """Direct calls (full price) for the first n questions -- also measures call latency."""
    client, rows, spent = anthropic.Anthropic(), [], 0.0
    for q in _questions(n):
        row = {"qid": q["id"], "author": q.get("author"), "question": q["question"],
               "seconds": [], "usage": [], **_stamp()}
        for kind, tool in (("live", LIVE_TOOL), ("split", SPLIT_TOOL)):
            t0 = time.perf_counter()
            msg = client.messages.create(**_params(tool, q["question"]))
            row["seconds"].append(round(time.perf_counter() - t0, 2))
            row[kind] = next(b.input for b in msg.content if b.type == "tool_use")
            row["usage"].append({"input_tokens": msg.usage.input_tokens,
                                 "output_tokens": msg.usage.output_tokens})
            spent += (msg.usage.input_tokens * PRICE[0] + msg.usage.output_tokens * PRICE[1]) / 1e6
        rows.append(row)
    out = _out()
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                   encoding="utf-8")
    print(f"-> {out.relative_to(ROOT)}  ({len(rows)} questions, ${spent:.4f})")


def submit() -> None:
    """Both calls for every scored question via the Batch API (half price)."""
    reqs = [{"custom_id": f"{q['id']}__{kind}", "params": _params(tool, q["question"])}
            for q in _questions() for kind, tool in (("live", LIVE_TOOL), ("split", SPLIT_TOOL))]
    batch = anthropic.Anthropic().messages.batches.create(requests=reqs)
    print(f"submitted batch {batch.id}: {len(reqs)} requests")


def collect(batch_id: str) -> None:
    client = anthropic.Anthropic()
    while client.messages.batches.retrieve(batch_id).processing_status != "ended":
        time.sleep(30)
    qs = {q["id"]: q for q in _questions()}
    rows = {qid: {"qid": qid, "author": q.get("author"), "question": q["question"], "usage": {},
                  "batch_id": batch_id, **_stamp()} for qid, q in qs.items()}
    spent, bad = 0.0, []
    for r in client.messages.batches.results(batch_id):
        qid, kind = r.custom_id.split("__")
        if r.result.type != "succeeded":
            bad.append(r.custom_id)
            continue
        msg = r.result.message
        use = [b.input for b in msg.content if b.type == "tool_use"]
        if not use:
            bad.append(r.custom_id)
            continue
        rows[qid][kind] = use[0]
        rows[qid]["usage"][kind] = {"input_tokens": msg.usage.input_tokens,
                                    "output_tokens": msg.usage.output_tokens}
        spent += (msg.usage.input_tokens * PRICE[0] + msg.usage.output_tokens * PRICE[1])             * 0.5 / 1e6
    out = _out()
    out.write_text("\n".join(json.dumps(rows[q], ensure_ascii=False) for q in qs) + "\n",
                   encoding="utf-8")
    print(f"batch {batch_id} ended: {len(qs)} questions, {len(bad)} unusable {bad}, "
          f"${spent:.4f} -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    {"smoke": lambda: smoke(int(sys.argv[2])), "submit": submit,
     "collect": lambda: collect(sys.argv[2])}[sys.argv[1]]()
