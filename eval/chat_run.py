"""Product-path eval: the gold set through the CHAT AGENT members actually use (src/agent.py,
D-042) -- LangChain create_agent, its prompt, its tools, its guards.

Why a third runner: eval/agent_run.py drives the retired coordinator (baselines.coordinator)
and cut the agent baselines of record, so it stays as it is -- re-pointing it would make those
runs unreproducible from HEAD (the same reason run.py was left alone). Scoring pieces are shared
(judge, Wilson bands, cost, refusal rule, citation parser, evidence recall), so the three runners
measure the same things the same way where they overlap.

What differs from agent_run.py, and why:
  * The agent declares no outcome (D-042 dropped declared outcomes). Refusal is still the exact
    sentence (D-019 prefix rule); clarify rows are scored by the correctness judge against their
    gold text instead of by a declaration, and clarify over-triggering on answerable questions
    is not measurable here (no declaration to count).
  * Evidence = what the tools returned this turn, recorded from the Toolbox itself (the model is
    shown a slimmer view of the same passages, A25), plus the asker's profile summary, which the
    agent's system prompt carries -- without it, correct use of the asker's own profile would
    score ungrounded.
  * The eval owns the clock (EVAL NOTES 2026-10-14 b): the agent's pause-triggered trimming
    (D-046) can never fire by accident during a rate-limit wait. Every gold row is one turn, so
    trimming could not fire anyway; the fixed clock keeps it that way for multi-turn gold later.
  * Web search is OFF by default (CHAT_EVAL_WEB=off): the gold set is corpus-only and replay mode
    with no recordings makes the tool dead weight (13/59 turns wasted, 2026-10-12). Whether the
    measured config should carry web is a product call to make before the first paid run.

Usage:  python eval/chat_run.py                  (full gold set; paid)
        EVAL_LIMIT=5 python eval/chat_run.py     (first N rows)
        CHAT_EVAL_WEB=replay python eval/chat_run.py   (web tool on, recorded fixtures only)
Needs Postgres up and ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

for stream in (sys.stdout, sys.stderr):  # cp1252 console vs corpus em dashes (chat_smoke.py)
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import agent  # noqa: E402
import citations  # noqa: E402
import cost  # noqa: E402
import hitrate  # noqa: E402
import normalize  # noqa: E402
import store  # noqa: E402
import tools  # noqa: E402
from agent_run import binom, pick_asker  # noqa: E402
from baselines.generate import format_context  # noqa: E402
from contracts import REFUSAL_STRING  # noqa: E402
from run import ANSWERABLE, QUESTIONS, RUNS_DIR, git_info, sha256_file, warn_if_dirty  # noqa: E402

LIMIT = int(os.environ.get("EVAL_LIMIT", "0")) or None
WEB = os.environ.get("CHAT_EVAL_WEB", "off")
# TUNABLE($5 per run = the working agreement's BIG-spend line; a run that would cross it stops
#   and reports. Estimated full 62-row run ~$2-3 (agent ~$0.012/turn measured 2026-10-13, judge
#   ~$0.03/q). Symptom wrong: a normal run aborts near the end -> re-estimate with sign-off.)
SPEND_CEILING_USD = float(os.environ.get("EVAL_SPEND_CEILING", "5.0"))
FIXED_CLOCK = 1_800_000_000.0     # any constant: no turn ever looks like it follows a pause


def is_refusal(answer: str) -> bool:
    """D-019's prefix rule, anchored on the shared refusal sentence."""
    na, nr = normalize.normalize_for_match(answer), normalize.normalize_for_match(REFUSAL_STRING)
    return bool(nr) and (na == nr or na.startswith(nr + " "))


class Recorder:
    """Wraps the Toolbox's three tools so each turn's raw results are kept: the corpus and web
    evidence the eval scores, and the call list for the trajectory. The product is not edited --
    the wrapper is installed on the class for this process only."""

    def __init__(self):
        self.corpus: list = []
        self.web: list = []
        self.calls: list[dict] = []
        for name in ("find_members", "get_person_evidence", "web_search"):
            original = getattr(tools.Toolbox, name)
            setattr(tools.Toolbox, name, self._wrap(name, original))

    def _wrap(self, name, original):
        rec = self

        def wrapped(box, *args, **kwargs):
            result = original(box, *args, **kwargs)
            ok = bool(result.get("ok"))
            rec.calls.append({"tool": name, "ok": ok,
                              **({} if ok else {"error": result.get("error")})})
            for e in result.get("evidence", []) if ok else []:
                (rec.web if name == "web_search" else rec.corpus).append(e)
            return result
        return wrapped

    def reset(self):
        self.corpus, self.web, self.calls = [], [], []


class TurnEvidence:
    """The duck-typed envelope eval/citations.py reads (.asker + evidence accessors)."""

    def __init__(self, asker: str, corpus: list, web: list):
        self.asker, self._corpus, self._web = asker, corpus, web

    def corpus_evidence(self):
        return self._corpus

    def web_evidence(self):
        return self._web


def usage_of(msg) -> dict | None:
    """One model call's usage. response_metadata["usage"] is authoritative for invoke() (the
    normalised usage_metadata files cache WRITES under a different key -- MEASUREMENT
    2026-10-13); fall back to usage_metadata, which is all a streamed turn carries."""
    raw = (getattr(msg, "response_metadata", None) or {}).get("usage")
    if raw:
        return {"input_tokens": raw.get("input_tokens", 0),
                "output_tokens": raw.get("output_tokens", 0),
                "cache_read_tokens": raw.get("cache_read_input_tokens") or 0,
                "cache_creation_tokens": raw.get("cache_creation_input_tokens") or 0}
    um = getattr(msg, "usage_metadata", None)
    if not um:
        return None
    det = um.get("input_token_details", {}) or {}
    written = (det.get("cache_creation") or 0) + (det.get("ephemeral_5m_input_tokens") or 0) \
        + (det.get("ephemeral_1h_input_tokens") or 0)
    read = det.get("cache_read") or 0
    return {"input_tokens": um.get("input_tokens", 0) - read - written,
            "output_tokens": um.get("output_tokens", 0),
            "cache_read_tokens": read, "cache_creation_tokens": written}


def run_row(chat_agent, recorder: Recorder, judge, row: dict, asker: agent.Asker) -> dict:
    """One gold row = one fresh single-turn thread. Exceptions are results, not crashes."""
    recorder.reset()
    rec: dict = {"id": row["id"], "type": row["type"], "question": row["question"],
                 "error": None, "answer": None, "tools": [], "rounds": None,
                 "evidence_recall": None, "evidence_hit": None, "refusal_exact": None,
                 "grounded": None, "is_refusal_judge": None, "correct": None,
                 "judge_reason": None, "citations": None, "cost": None}
    cfg = {"configurable": {"thread_id": f"eval-{row['id']}"}}
    t0 = time.perf_counter()
    msgs = []
    try:
        state = chat_agent.invoke({"messages": [{"role": "user", "content": row["question"]}]},
                                  config=cfg, context=asker)
        msgs = state["messages"]
        rec["answer"] = agent._text(msgs[-1])
    except Exception as exc:  # a guard or limit firing is a finding, not a crash
        rec.update(error=f"{type(exc).__name__}: {exc}", correct=False)
    rec["turn_latency_s"] = round(time.perf_counter() - t0, 3)

    ai = [m for m in msgs if m.type == "ai"]
    rec["rounds"] = len(ai)
    rec["tools"] = [c["tool"] + ("" if c["ok"] else f"!{c.get('error')}") for c in recorder.calls]
    rec["tool_calls"] = [{"name": c["name"], "args": c["args"]}
                         for m in ai for c in (m.tool_calls or [])]
    stages: dict = {}
    calls = [u for u in (usage_of(m) for m in ai) if u]
    if calls:
        stages["agent"] = {"model": agent.AGENT_MODEL, "calls": len(calls),
                           **{k: sum(c[k] for c in calls) for k in calls[0]}}
        stages["agent"]["cost_usd"] = cost.cost_usd(stages["agent"])

    if rec["answer"] is not None:
        answer = rec["answer"]
        ev_dicts = [{"doc_id": e.doc_id, "text": e.text} for e in recorder.corpus]
        if row["type"] in ANSWERABLE:
            found = hitrate.spans_found_at_k(row["evidence"], ev_dicts, len(ev_dicts) or 1)
            rec["evidence_recall"] = found / len(row["evidence"])
            rec["evidence_hit"] = found == len(row["evidence"])
        rec["refusal_exact"] = is_refusal(answer)
        rec["citations"] = citations.parse(answer, TurnEvidence(asker.person_id, recorder.corpus,
                                                                recorder.web))
        shown = ev_dicts + ([{"doc_id": asker.name, "text": asker.profile_summary}]
                            if asker.profile_summary else [])
        shown += [{"doc_id": f"web: {w.url}", "text": f"{w.title}\n{w.snippet}"}
                  for w in recorder.web]
        g, g_meta = judge.groundedness(row["question"], format_context(shown), answer)
        c, c_meta = judge.correctness(row["question"], row["gold_answer"], answer)
        rec.update(grounded=g["grounded"], is_refusal_judge=g["is_refusal"], correct=c["correct"],
                   judge_reason={"groundedness": g["reason"], "correctness": c["reason"]})
        for st, meta in (("groundedness", g_meta), ("correctness", c_meta)):
            if meta:
                stages[st] = {**meta, "cost_usd": cost.cost_usd(meta)}
    rec["cost"] = {"stages": stages,
                   "total_usd": round(sum(s.get("cost_usd") or 0 for s in stages.values()), 6)}
    return rec


def summarize(results: list[dict]) -> dict:
    answerable = [x for x in results if x["type"] in ANSWERABLE]
    abst = [x for x in results if x["type"] == "abstention"]
    non_refusal = [x for x in results if x["refusal_exact"] is False]
    cited = [x for x in non_refusal if x["citations"]]
    lat = sorted(x["turn_latency_s"] for x in results)

    def pct(p):
        return lat[min(len(lat) - 1, int(p * len(lat)))] if lat else None

    correctness = {t: binom([x["correct"] for x in results if x["type"] == t])
                   for t in sorted({x["type"] for x in results})}
    correctness["answerable_overall"] = binom([x["correct"] for x in answerable])
    correctness["overall"] = binom([x["correct"] for x in results])
    priced = [x["cost"] for x in results if x.get("cost")]
    by_stage: dict = {}
    for c in priced:
        for st, m in c["stages"].items():
            b = by_stage.setdefault(st, {"input_tokens": 0, "output_tokens": 0,
                                         "cache_read_tokens": 0, "cache_creation_tokens": 0,
                                         "total_usd": 0.0})
            for k in ("input_tokens", "output_tokens", "cache_read_tokens",
                      "cache_creation_tokens"):
                b[k] += m.get(k, 0)
            b["total_usd"] = round(b["total_usd"] + (m.get("cost_usd") or 0), 4)
    return {
        "n": len(results),
        "errors": {"n": sum(1 for x in results if x["error"]),
                   "ids": [x["id"] for x in results if x["error"]]},
        "evidence_hit": binom([x["evidence_hit"] for x in answerable]),
        "groundedness": binom([x["grounded"] for x in non_refusal]),
        "correctness": correctness,
        "abstention_accuracy": binom([x["refusal_exact"] for x in abst]),
        "false_refusal_rate": binom([x["refusal_exact"] for x in answerable]),
        "refusal_judge_divergence": [x["id"] for x in results if x["refusal_exact"] is not None
                                     and x["is_refusal_judge"] is not None
                                     and x["refusal_exact"] != x["is_refusal_judge"]],
        "citations": {
            "n_answers": len(cited),
            "fabricated_rate": binom([x["citations"]["has_fabricated"] for x in cited]),
            "zero_citation_rate": binom([not x["citations"]["has_any_citation"] for x in cited]),
            "off_contract_answers": sum(1 for x in cited if x["citations"]["off_contract"]),
        },
        "trajectory": {"rounds_mean": round(sum(x["rounds"] or 0 for x in results) / len(results), 2),
                       "rounds_max": max(x["rounds"] or 0 for x in results)},
        "latency_s": {"p50": pct(0.5), "p95": pct(0.95)},
        "cost": {"total_usd": round(sum(c["total_usd"] for c in priced), 4),
                 "mean_usd_per_question": round(sum(c["total_usd"] for c in priced) / len(priced), 4)
                 if priced else None,
                 "by_stage": by_stage},
    }


def config_of(run_id: str, rows: list[dict], asker: agent.Asker, judge_cfg: dict, emb) -> dict:
    core = agent.PROMPT_FILE.read_bytes().replace(b"\r\n", b"\n")
    web_prompt = agent.WEB_PROMPT_FILE.read_bytes().replace(b"\r\n", b"\n")
    return {
        "run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
        "runner": "eval/chat_run.py", "git": git_info(),
        # person_ids are real-name slugs and config.json is committed: a hash pins the asker.
        "asker_sha256_12": hashlib.sha256(asker.person_id.encode()).hexdigest()[:12],
        "agent": {"model": agent.AGENT_MODEL,
                  "prompt_core_sha256": hashlib.sha256(core).hexdigest(),
                  "prompt_web_sha256": hashlib.sha256(web_prompt).hexdigest() if WEB != "off"
                  else None,
                  "web": WEB, "hybrid_search": tools.HYBRID_SEARCH, "keep_turns": agent.KEEP_TURNS,
                  "clock": "fixed"},
        "stack": {p: version(p) for p in ("langchain", "langgraph", "langchain-anthropic",
                                          "anthropic")},
        "judge": judge_cfg, "embedder": {"id": emb.model_id}, "chunk_scheme": tools.SCHEME,
        "normalizer_version": normalize.NORMALIZER_VERSION,
        "question_set": {"path": str(QUESTIONS), "n": len(rows), "limit": LIMIT,
                         "sha256": sha256_file(QUESTIONS)},
        "prices_dated": cost.PRICES_DATED, "spend_ceiling_usd": SPEND_CEILING_USD,
    }


def main() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    from embedder import NomicLocal
    from judge import JUDGE_MODEL, RUBRIC_VERSION, Judge

    warn_if_dirty()
    if WEB not in ("off", "replay"):
        sys.exit("CHAT_EVAL_WEB must be off or replay -- an eval never calls the live web")
    if WEB == "replay":
        os.environ["WEB_MODE"] = "replay"
    rows = [json.loads(ln) for ln in QUESTIONS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows = rows[:LIMIT] if LIMIT else rows

    conn = store.connect()
    resolver = tools.build_resolver(conn)
    emb = NomicLocal()
    emb.warm()
    agent._clock = lambda: FIXED_CLOCK
    recorder = Recorder()
    chat_agent = agent.build_agent(emb, conn, resolver, InMemorySaver(), web_enabled=WEB != "off")
    asker = agent.load_asker(conn, resolver, pick_asker(conn, resolver, rows))
    judge = Judge()

    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{git_info()['sha']}-chat"
    out_dir = RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    results, spent = [], 0.0
    for r in rows:
        rec = run_row(chat_agent, recorder, judge, r, asker)
        results.append(rec)
        spent += rec["cost"]["total_usd"]
        print(f"  [{r['id']:<8}] {r['type']:<11} rounds={rec['rounds']} tools={rec['tools']} "
              f"recall={rec['evidence_recall']} correct={rec['correct']} "
              f"grounded={rec['grounded']} ${rec['cost']['total_usd']:.4f}"
              + (f"  !! {rec['error']}" if rec["error"] else ""))
        if spent > SPEND_CEILING_USD:
            print(f"!! spend ceiling ${SPEND_CEILING_USD} crossed at ${spent:.2f}; stopping.")
            break

    summary = summarize(results)
    cfg = config_of(run_id, rows, asker, {"id": JUDGE_MODEL, "rubric_version": RUBRIC_VERSION}, emb)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== chat run {run_id}: {json.dumps({k: summary[k] for k in ('n', 'errors')})}")
    print(json.dumps({k: summary[k] for k in ("evidence_hit", "groundedness", "abstention_accuracy",
                                              "false_refusal_rate", "latency_s", "cost")}, indent=1))


if __name__ == "__main__":
    main()
