"""Agent-path eval: the gold set through the REAL coordinator loop (D-036) and the pinned synth-v1
answer call (D-032). The Step-5 re-baseline runs here.

Why a second runner instead of editing run.py: run.py IS the f6-v1 pipeline's instrument, and
every Phase 1-2 baseline-of-record was cut by it. Re-pointing it in place would make those runs
unreproducible from HEAD. This file reuses run.py's scoring pieces (judge, Wilson bands, cost,
refusal normalizer) so the two runners measure the same things the same way where they overlap.

What changes vs run.py, and why:
  * The unit is a TURN, not a retrieve+generate call. Each gold question is a fresh one-turn
    thread for a fixed asker. Retrieval is whatever the coordinator chose, so hit@k/MRR (which
    assume one ranked list) are replaced by EVIDENCE RECALL: are the gold spans anywhere in what
    the tools returned this turn.
  * The groundedness judge reads env.evidence -- the same record the writer was shown.
  * Refusal anchors on synthesize.REFUSAL_STRING (re-anchored at D-019/G-003), not f6-v1's.
  * Harness-law exceptions (BypassViolation, BudgetExhausted, contract errors) are RESULTS, not
    crashes: the row records the error, scores incorrect, and the run continues.
  * Trajectory + overhead: rounds, tool sequence, per-stage tokens and latency -- the defense
    pack's overhead row reads these.

Usage:  python eval/agent_run.py                 (full gold set)
        EVAL_LIMIT=5 python eval/agent_run.py    (first N rows, smoke)
Needs Postgres up and ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

for stream in (sys.stdout, sys.stderr):  # cp1252 console vs corpus em dashes (chat_smoke.py)
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import citations  # noqa: E402
import cost  # noqa: E402
import hitrate  # noqa: E402
import normalize  # noqa: E402
import store  # noqa: E402
from baselines.coordinator import (  # noqa: E402
    MAX_ROUNDS,
    Coordinator,
    contract_versions as coord_versions,
)
from embedder import NomicLocal  # noqa: E402
from baselines.generate import format_context  # noqa: E402
from baselines.provider import AnthropicProvider  # noqa: E402
from run import QUESTIONS, RUNS_DIR, ANSWERABLE, git_info, sha256_file, warn_if_dirty  # noqa: E402
from baselines.session import Thread  # noqa: E402
from stats import wilson_interval  # noqa: E402
from baselines.synthesize import REFUSAL_STRING, answer_contract_versions, make_answer_writer  # noqa: E402
from tools import SCHEME, Toolbox, build_resolver  # noqa: E402
from baselines.tracing import init_tracing, shutdown  # noqa: E402

LIMIT = int(os.environ.get("EVAL_LIMIT", "0")) or None

# TUNABLE($5 spend ceiling per run = the BIG-decision threshold in the working agreement; a run
#   that would cross it stops and reports rather than spending. Estimated full run ~$2-4
#   (coordinator+synth ~$0.012/turn measured, judge ~$0.04/q). Symptom wrong: a full run aborts
#   near the end on a normal-looking per-question cost -> re-estimate, raise with explicit sign-off.)
SPEND_CEILING_USD = float(os.environ.get("EVAL_SPEND_CEILING", "5.0"))


class CountingProvider:
    """Wraps the real provider and records every call's usage, so the run can price the
    coordinator rounds and the synth call separately (the overhead row needs the split)."""

    def __init__(self, inner):
        self.inner = inner
        self.model, self.synth_model = inner.model, inner.synth_model
        self.calls: list[dict] = []

    def _meta(self, stage, model, turn, t0) -> dict:
        return {"stage": stage, "model": model, "latency_s": round(time.perf_counter() - t0, 3),
                "input_tokens": turn.input_tokens, "output_tokens": turn.output_tokens,
                "cache_read_tokens": turn.cache_read_tokens,
                "cache_creation_tokens": turn.cache_creation_tokens}

    def complete(self, system, messages, tools, force_tool, max_tokens):
        t0 = time.perf_counter()
        turn = self.inner.complete(system, messages, tools=tools, force_tool=force_tool,
                                   max_tokens=max_tokens)
        self.calls.append(self._meta("coordinator", self.model, turn, t0))
        return turn

    def stream_text(self, system, messages, tools=None, max_tokens=1024, on_delta=None):
        t0 = time.perf_counter()
        turn = self.inner.stream_text(system, messages, tools=tools, max_tokens=max_tokens,
                                      on_delta=on_delta)
        self.calls.append(self._meta("synth", self.synth_model, turn, t0))
        return turn


class NullStore:
    """Eval threads are throwaway; nothing is persisted to the threads table."""

    def save(self, thread):
        pass


def is_refusal(answer: str) -> bool:
    """D-019's prefix rule (run.is_exact_refusal), anchored on the synth-v1 string."""
    na, nr = normalize.normalize_for_match(answer), normalize.normalize_for_match(REFUSAL_STRING)
    return bool(nr) and (na == nr or na.startswith(nr + " "))


def pick_asker(conn, resolver, rows: list[dict]) -> str:
    """A fixed asker who is neither a gold doc nor named in any question. find_members always
    excludes the asker, and the writer cites the asker's profile as [self] -- so an asker who
    IS a gold subject would change what the question means. First such person_id in sorted
    order: deterministic, and recorded in config."""
    gold_docs = {e["doc_id"] for r in rows for e in r["evidence"]}
    named: set[str] = set()
    for r in rows:
        res = resolver.resolve(r["question"])
        named |= set(res.person_ids)
        for c in res.candidates.values():
            named |= set(c)
    with conn.cursor() as cur:
        cur.execute("SELECT person_id, canonical_name FROM persons ORDER BY person_id")
        for pid, name in cur.fetchall():
            if name not in gold_docs and pid not in named:
                return pid
    raise RuntimeError("no eligible asker")


def binom(vals: list) -> dict:
    vals = [v for v in vals if v is not None]
    n, k = len(vals), sum(1 for v in vals if v)
    lo, hi = wilson_interval(k, n)
    return {"n": n, "successes": k, "rate": k / n if n else None, "ci_lo": lo, "ci_hi": hi}


def main() -> None:
    warn_if_dirty()
    rows = [json.loads(ln) for ln in QUESTIONS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows = rows[:LIMIT] if LIMIT else rows

    conn = store.connect()
    resolver = build_resolver(conn)
    emb = NomicLocal()
    emb.warm()  # load cost belongs to startup, not question 1 (Step-5 finding)
    provider = CountingProvider(AnthropicProvider())
    from judge import JUDGE_MODEL, RUBRIC_VERSION, Judge  # noqa: E402

    judge = Judge()
    asker = pick_asker(conn, resolver, rows)
    coord = Coordinator(provider, toolbox_factory=lambda a: Toolbox(emb, conn, resolver, asker=a),
                        resolver=resolver, store=NullStore(),
                        synthesize_fn=make_answer_writer(provider, resolver.name_of))

    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{git_info()['sha']}-agent"
    out_dir = RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    init_tracing(str(out_dir / "trace.jsonl"))

    results: list[dict] = []
    spent = 0.0
    for r in rows:
        provider.calls = []
        thread = Thread(thread_id=f"eval-{r['id']}", asker=asker)
        t0 = time.perf_counter()
        rec: dict = {"id": r["id"], "type": r["type"], "question": r["question"],
                     "error": None, "outcome": None, "answer": None, "tools": [],
                     "rounds": None, "evidence_recall": None, "evidence_hit": None,
                     "refusal_exact": None, "grounded": None, "is_refusal_judge": None,
                     "correct": None, "judge_reason": None, "citations": None, "cost": None}
        env = None
        try:
            env = coord.run_turn(thread, r["question"])
            rec.update(outcome=env.response_mode, answer=env.response,
                       clarify_reason=env.clarify_reason)
        except Exception as exc:  # harness law firing is a finding, not a crash
            rec.update(error=f"{type(exc).__name__}: {exc}", correct=False)
        rec["turn_latency_s"] = round(time.perf_counter() - t0, 3)
        rec["rounds"] = sum(1 for c in provider.calls if c["stage"] == "coordinator")

        stages = {}
        if env is not None:
            rec["tools"] = [c.tool + ("" if c.ok else f"!{c.error}") for c in env.tool_calls]
            # Full calls WITH arguments: the trajectory is undebuggable without what the model
            # actually asked for (first agent run: retrieval misses on chunks that rank top-3
            # for the user's own question could not be explained from tool names alone).
            rec["tool_calls"] = [c.to_dict() for c in env.tool_calls]
            rec["resolved_query"] = env.resolved_query
            ev_dicts = [{"doc_id": e.doc_id, "text": e.text} for e in env.corpus_evidence()]
            if r["type"] in ANSWERABLE:
                found = hitrate.spans_found_at_k(r["evidence"], ev_dicts, len(ev_dicts) or 1)
                rec["evidence_recall"] = found / len(r["evidence"])
                rec["evidence_hit"] = found == len(r["evidence"])
            rec["refusal_exact"] = is_refusal(env.response)
            rec["citations"] = citations.parse(env.response, env)
            context = format_context(ev_dicts) or "(no evidence retrieved)"
            g, g_meta = judge.groundedness(r["question"], context, env.response)
            if r["type"] == "clarify":
                # D-040: the right outcome is DECLARED, so it is scored on the declaration --
                # no judge. Groundedness still runs: the clarify text must assert nothing.
                c, c_meta = {"correct": env.response_mode == "clarify",
                             "reason": f"declared {env.response_mode}"}, None
            else:
                c, c_meta = judge.correctness(r["question"], r["gold_answer"], env.response)
            rec.update(grounded=g["grounded"], is_refusal_judge=g["is_refusal"],
                       correct=c["correct"],
                       judge_reason={"groundedness": g["reason"], "correctness": c["reason"]})
            stages["groundedness"] = g_meta
            if c_meta:
                stages["correctness"] = c_meta

        for st in ("coordinator", "synth"):
            calls = [c for c in provider.calls if c["stage"] == st]
            if calls:
                stages[st] = {"model": calls[0]["model"], "calls": len(calls),
                              "input_tokens": sum(c["input_tokens"] for c in calls),
                              "output_tokens": sum(c["output_tokens"] for c in calls),
                              "cache_read_tokens": sum(c["cache_read_tokens"] for c in calls),
                              "cache_creation_tokens": sum(c["cache_creation_tokens"]
                                                           for c in calls),
                              "latency_s": round(sum(c["latency_s"] for c in calls), 3),
                              "cost_usd": round(sum(cost.cost_usd(c) or 0 for c in calls), 6)}
        for st in ("groundedness", "correctness"):
            if st in stages:
                stages[st]["cost_usd"] = cost.cost_usd(stages[st])
        total = round(sum(s.get("cost_usd") or 0 for s in stages.values()), 6)
        rec["cost"] = {"stages": stages, "total_usd": total}
        spent += total
        results.append(rec)

        print(f"  [{r['id']:<8}] {r['type']:<11} {rec['outcome'] or 'ERROR':<8} "
              f"rounds={rec['rounds']} tools={rec['tools']} recall={rec['evidence_recall']} "
              f"correct={rec['correct']} grounded={rec['grounded']} ${total:.4f}"
              + (f"  !! {rec['error']}" if rec["error"] else ""))
        if spent > SPEND_CEILING_USD:
            print(f"!! spend ceiling ${SPEND_CEILING_USD} crossed at ${spent:.2f}; stopping.")
            break

    summary = summarize(results)
    config = {
        "run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
        "runner": "eval/agent_run.py", "git": git_info(),
        # person_ids are real-name slugs and config.json is committed: record a hash that
        # pins WHICH asker without naming them (same selection rule reproduces it).
        "asker_sha256_12": hashlib.sha256(asker.encode()).hexdigest()[:12],
        "coordinator": {**coord_versions(), "model": provider.model, "max_rounds": MAX_ROUNDS},
        "answer": {**answer_contract_versions(), "model": provider.synth_model},
        "judge": {"id": JUDGE_MODEL, "rubric_version": RUBRIC_VERSION},
        "embedder": {"id": emb.model_id}, "chunk_scheme": SCHEME,
        "normalizer_version": normalize.NORMALIZER_VERSION,
        "question_set": {"path": str(QUESTIONS), "n": len(rows), "limit": LIMIT,
                         "sha256": sha256_file(QUESTIONS)},
        "prices_dated": cost.PRICES_DATED, "spend_ceiling_usd": SPEND_CEILING_USD,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    shutdown()
    report(run_id, summary)


def summarize(results: list[dict]) -> dict:
    answerable = [x for x in results if x["type"] in ANSWERABLE]
    abst = [x for x in results if x["type"] == "abstention"]
    non_refusal = [x for x in results if x["refusal_exact"] is False]
    cited = [x for x in non_refusal if x["citations"]]
    priced = [x["cost"] for x in results if x.get("cost")]
    lat = sorted(x["turn_latency_s"] for x in results)

    def pct(p):
        return lat[min(len(lat) - 1, int(p * len(lat)))] if lat else None

    correctness = {t: binom([x["correct"] for x in results if x["type"] == t])
                   for t in sorted({x["type"] for x in results})}
    correctness["answerable_overall"] = binom([x["correct"] for x in answerable])
    correctness["overall"] = binom([x["correct"] for x in results])
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
        "outcomes": {o: sum(1 for x in results if (x["outcome"] or "error") == o)
                     for o in sorted({x["outcome"] or "error" for x in results})},
        "evidence_hit": binom([x["evidence_hit"] for x in answerable]),
        "groundedness": binom([x["grounded"] for x in non_refusal]),
        "correctness": correctness,
        "abstention_accuracy": binom([x["refusal_exact"] for x in abst]),
        "false_refusal_rate": binom([x["refusal_exact"] for x in answerable]),
        # Clarify over-triggering: asking "what do you mean?" on a question it should have
        # searched is a false refusal by another name (the clarify decision, 2026-10-12).
        "clarify_on_answerable": binom([x["outcome"] == "clarify" for x in answerable]),
        "refusal_judge_divergence": [x["id"] for x in results if x["refusal_exact"] is not None
                                     and x["is_refusal_judge"] is not None
                                     and x["refusal_exact"] != x["is_refusal_judge"]],
        "citations": {
            "n_answers": len(cited),
            "fabricated_rate": binom([x["citations"]["has_fabricated"] for x in cited]),
            "zero_citation_rate": binom([not x["citations"]["has_any_citation"] for x in cited]),
            "off_contract_answers": sum(1 for x in cited if x["citations"]["off_contract"]),
            "answers_with_uncited_profiles": sum(1 for x in cited
                                                 if x["citations"]["uncited_profiles"]),
        },
        "trajectory": {
            "rounds_mean": round(sum(x["rounds"] for x in results) / len(results), 2),
            "rounds_max": max(x["rounds"] for x in results),
        },
        "latency_s": {"p50": pct(0.5), "p95": pct(0.95)},
        "cost": {"total_usd": round(sum(c["total_usd"] for c in priced), 4),
                 "mean_usd_per_question": round(sum(c["total_usd"] for c in priced) / len(priced), 4)
                 if priced else None,
                 "by_stage": by_stage},
    }


def report(run_id: str, s: dict) -> None:
    def f(m):
        return "n/a" if m["rate"] is None else f"{m['rate']:.2f} [{m['ci_lo']:.2f}, {m['ci_hi']:.2f}] (n={m['n']})"

    print(f"\n=== agent run {run_id} ===")
    print(f"outcomes: {s['outcomes']}   errors: {s['errors']['n']} {s['errors']['ids']}")
    print(f"evidence hit (all gold spans in turn evidence): {f(s['evidence_hit'])}")
    print(f"groundedness (non-refusal): {f(s['groundedness'])}")
    for t, m in s["correctness"].items():
        print(f"  correctness {t:<18}: {f(m)}")
    print(f"abstention accuracy: {f(s['abstention_accuracy'])} | false-refusal: {f(s['false_refusal_rate'])}")
    print(f"clarify on answerable (over-trigger): {f(s['clarify_on_answerable'])}")
    print(f"refusal judge divergence: {s['refusal_judge_divergence']}")
    ct = s["citations"]
    print(f"citations over {ct['n_answers']}: fabricated {f(ct['fabricated_rate'])} | zero "
          f"{f(ct['zero_citation_rate'])} | off-contract {ct['off_contract_answers']} | "
          f"uncited-profile answers {ct['answers_with_uncited_profiles']}")
    print(f"rounds mean {s['trajectory']['rounds_mean']} max {s['trajectory']['rounds_max']} | "
          f"latency p50 {s['latency_s']['p50']}s p95 {s['latency_s']['p95']}s")
    print(f"cost ${s['cost']['total_usd']} (${s['cost']['mean_usd_per_question']}/q) "
          f"{ {k: v['total_usd'] for k, v in s['cost']['by_stage'].items()} }")
    for st in ("coordinator", "synth"):
        b = s["cost"]["by_stage"].get(st)
        if b:
            print(f"  cache {st}: uncached-in {b['input_tokens']} | read {b['cache_read_tokens']}"
                  f" | written {b['cache_creation_tokens']}")


if __name__ == "__main__":
    main()
