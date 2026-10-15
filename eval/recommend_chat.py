"""Recommendation quality at the ANSWER level: the 34 scored recommendation questions through the
real chat agent (src/agent.py), scoring the members its reply actually recommends -- in the order
it names them -- against the answer key. recommend_eval.py scores the search's ranked list; this
scores what the member reads.

Two configurations, same questions, same fixed asker:
  today   tools.MODEL_PICKS off: find_members hands the model hybrid's top 5, best first
  picks   on (D-053): 15 unranked candidates + the recommendation rules; the model chooses

The recommended people are the members named in the reply, by full name, in order of first
mention (citation tags name the same person, so they do not reorder). Metrics are
recommend_eval's ndcg@5 / useful@5 / best@5 over that list; recall@k does not apply to a reply.

Usage:  python eval/recommend_chat.py today picks [--real]   (paid: ~$0.023/question/config;
                                                              --real = the real-user questions)
        EVAL_LIMIT=1 python eval/recommend_chat.py picks   (smoke)
Writes runs/<ts>-<sha>-rec-chat-<config>/ (config + summary committable; results.jsonl, which
holds the replies, gitignored). Compare with recommend_eval.py compare.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import agent  # noqa: E402
import cost  # noqa: E402
import store  # noqa: E402
import tools  # noqa: E402
from chat_run import FIXED_CLOCK, usage_of  # noqa: E402
from recommend_eval import QUESTIONS, load_key, score  # noqa: E402

LIMIT = int(os.environ.get("EVAL_LIMIT", "0")) or None
REAL_ONLY = "--real" in sys.argv     # only questions real members asked (author != assistant)
# TUNABLE($1 for both configurations, the amount approved for this check; a run that would cross
#   it stops and reports. Symptom wrong: a normal run aborts near the end -> re-estimate.)
SPEND_CEILING_USD = float(os.environ.get("EVAL_SPEND_CEILING", "1.0"))
METRICS = ("ndcg@5", "useful@5", "best@5")


def pick_asker(grades: dict, kept: list[dict], name_of: dict) -> str:
    """A fixed asker the key rates 0 on every scored question: find_members always excludes the
    asker, so an asker who is a right answer would cost every configuration a slot."""
    for pid in sorted(name_of):
        if all(grades[q["id"]].get(pid, 0) == 0 for q in kept):
            return pid
    raise RuntimeError("no asker graded 0 everywhere")


def named_in_order(reply: str, name_of: dict, asker: str) -> list[str]:
    """Members named in the reply by full name, in order of first mention."""
    first: dict[str, int] = {}
    for pid, name in name_of.items():
        if pid == asker:
            continue
        m = re.search(rf"(?<!\w){re.escape(name)}(?!\w)", reply, re.IGNORECASE)
        if m:
            first[pid] = m.start()
    return sorted(first, key=first.get)


def run_config(config: str, conn, emb, resolver, grades, kept, asker, spent_before: float):
    from langgraph.checkpoint.memory import InMemorySaver

    picks = config == "picks"
    chat = agent.build_agent(emb, conn, resolver, InMemorySaver(), web_enabled=False,
                             model_picks=picks)
    rows, per_q, spent = [], {}, spent_before
    for q in kept:
        cfg = {"configurable": {"thread_id": f"rec-{config}-{q['id']}"}}
        t0 = time.perf_counter()
        error, msgs = None, []
        try:
            state = chat.invoke({"messages": [{"role": "user", "content": q["question"]}]},
                                config=cfg, context=asker)
            msgs = state["messages"]
        except Exception as exc:  # a guard firing is a result, not a crash
            error = f"{type(exc).__name__}: {exc}"
        took = time.perf_counter() - t0
        reply = agent._text(msgs[-1]) if msgs else ""
        named = named_in_order(reply, resolver.name_of, asker.person_id)
        s = score(named, grades[q["id"]])
        per_q[q["id"]] = {m: round(s[m], 3) for m in METRICS}
        ai = [m for m in msgs if m.type == "ai"]
        calls = [u for u in (usage_of(m) for m in ai) if u]
        usage = {k: sum(c[k] for c in calls) for k in calls[0]} if calls else {}
        usd = cost.cost_usd({"model": agent.AGENT_MODEL, **usage}) if usage else 0.0
        spent += usd or 0.0
        rows.append({"qid": q["id"], "error": error, "named": named, "n_named": len(named),
                     "reply": reply, "turn_s": round(took, 2), "rounds": len(ai),
                     "tool_calls": [{"name": c["name"], "args": c["args"]}
                                    for m in ai for c in (m.tool_calls or [])],
                     "usage": usage, "cost_usd": usd})
        print(f"  [{config} {q['id']}] named={len(named)} ndcg@5={s['ndcg@5']:.2f} "
              f"tools={[c['name'] for m in ai for c in (m.tool_calls or [])]} {took:.1f}s "
              f"${usd or 0:.4f}" + (f" !! {error}" if error else ""))
        if spent > SPEND_CEILING_USD:
            print(f"!! spend ceiling ${SPEND_CEILING_USD} crossed at ${spent:.3f}; stopping.")
            break
    return rows, per_q, spent


def write_run(config: str, rows, per_q, kept, asker) -> str:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         cwd=ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                                cwd=ROOT).stdout.strip())
    tag = f"chat-{config}" + ("-real" if REAL_ONLY else "")
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{sha}-rec-{tag}"
    out = ROOT / "runs" / run_id
    out.mkdir(parents=True)
    picks = config == "picks"
    base = agent.base_prompt(False, picks).encode()
    (out / "config.json").write_text(json.dumps({
        "run_id": run_id, "runner": "eval/recommend_chat.py", "config": config,
        "git": {"sha": sha, "dirty": dirty}, "query_input": {"mode": "chat"},
        "agent": {"model": agent.AGENT_MODEL, "model_picks": picks,
                  "prompt_sha256": hashlib.sha256(base).hexdigest(), "web": "off",
                  "candidates": tools.CANDIDATES_FOR_MODEL if picks else tools.TOP_N_MEMBERS,
                  "passages": tools.CANDIDATE_PASSAGES if picks else tools.SNIPPETS_PER_MEMBER,
                  "hybrid_search": tools.HYBRID_SEARCH, "clock": "fixed"},
        "asker_sha256_12": hashlib.sha256(asker.person_id.encode()).hexdigest()[:12],
        "answer_key": "answer_key_v3_haiku55.jsonl", "questions": str(QUESTIONS.name),
        "questions_kept": [q["id"] for q in kept], "limit": LIMIT,
        "subset": "real" if REAL_ONLY else "all"}, indent=1), encoding="utf-8")
    turn = sorted(r["turn_s"] for r in rows)
    (out / "summary.json").write_text(json.dumps({
        "n_questions": len(per_q),
        "overall": {m: round(sum(v[m] for v in per_q.values()) / len(per_q), 3) for m in METRICS},
        "named_mean": round(sum(r["n_named"] for r in rows) / len(rows), 2),
        "no_find_members": sum(1 for r in rows if not any(c["name"] == "find_members"
                                                          for c in r["tool_calls"])),
        "errors": sum(1 for r in rows if r["error"]),
        "turn_s_p50": turn[len(turn) // 2], "turn_s_p95": turn[min(len(turn) - 1,
                                                                    math.ceil(0.95 * len(turn)) - 1)],
        "cost_usd": round(sum(r["cost_usd"] or 0 for r in rows), 4),
        "input_tokens_mean": round(sum(r["usage"].get("input_tokens", 0)
                                       + r["usage"].get("cache_read_tokens", 0)
                                       + r["usage"].get("cache_creation_tokens", 0)
                                       for r in rows) / len(rows)),
        "per_question": per_q}, indent=1), encoding="utf-8")
    (out / "results.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
                                       + "\n", encoding="utf-8")
    return run_id


def main(configs: list[str]) -> None:
    from embedder import NomicLocal

    grades, kept = load_key()
    if REAL_ONLY:
        kept = [q for q in kept if q.get("author") != "assistant"]
    kept = kept[:LIMIT] if LIMIT else kept
    conn = store.connect()
    resolver = tools.build_resolver(conn)
    emb = NomicLocal()
    emb.warm()
    agent._clock = lambda: FIXED_CLOCK
    asker = agent.load_asker(conn, resolver, pick_asker(grades, load_key()[1], resolver.name_of))
    spent = 0.0
    for config in configs:
        rows, per_q, spent = run_config(config, conn, emb, resolver, grades, kept, asker, spent)
        run_id = write_run(config, rows, per_q, kept, asker)
        s = json.loads((ROOT / "runs" / run_id / "summary.json").read_text())
        print(f"{run_id}: {s['overall']} | named {s['named_mean']}/reply | turn p50 "
              f"{s['turn_s_p50']}s | ${s['cost_usd']}")
        if spent > SPEND_CEILING_USD:
            break


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if a in ("today", "picks")])
