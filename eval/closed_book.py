"""Closed-book contamination control (D-013, roadmap item 14).

Runs the generator with EMPTY context on the ANSWERABLE questions and judges CORRECTNESS only
(there is no context, so groundedness is N/A). Needs the API but NOT the DB -- empty context
means no retrieval.

WHY (D-013): corpus subjects may appear in the generator's pretraining data, so it could answer
correctly WITHOUT retrieval -- inflating correctness and masking a broken RAG.
Closed-book correctness = the share of correctness that is retrieval-INDEPENDENT (pure memory).
open-book correctness (the baseline) minus closed-book = retrieval's actual lift. HIGH closed-book
correctness => correctness is grading the model's memory of the subjects, not the RAG -> fire the
D-013 names-only anonymizer.

OPTION A (chosen over B): keep the f6 contract in force -- rule 1 "do not use
prior knowledge, even if you recognize the person" + rule 2 "refuse if the fact isn't in the
reports". With EMPTY reports a contract-obeying bot refuses everything (correctness -> 0), and a
contaminated bot answers from memory anyway (correctness > 0). This measures OPERATIONAL
contamination -- memory leaking PAST the guardrail -- which is the SAME decision the bot faces
open-book when retrieval misses. A plain "just answer" prompt (B) would measure raw memory
CAPACITY, an upper bound that never occurs operationally because the guardrail is always present.

Interpretation anchor from the 1d baseline: open-book single-hop correctness was correct-IFF-
retrieved (HIT 29/29, MISS 0/12) -- zero correct-on-miss -- so this control is expected to
CONFIRM ~0 closed-book correctness. A non-zero result would CONTRADICT that and flag a leak.

Usage: ./.venv/Scripts/python.exe eval/closed_book.py   (needs ANTHROPIC_API_KEY; no DB)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "eval")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import generate  # noqa: E402
from run import git_info, is_exact_refusal, sha256_file  # noqa: E402 -- reuse D-019/D-021 helpers
from stats import wilson_interval  # noqa: E402

QUESTIONS = Path(os.environ.get("EVAL_QUESTIONS", "eval/questions.jsonl"))
RUNS_DIR = Path("runs")
ANSWERABLE = {"single-hop", "multi-hop"}


def binom(pred, subset) -> dict:
    vals = [pred(x) for x in subset if pred(x) is not None]
    n = len(vals)
    k = sum(1 for v in vals if v)
    lo, hi = wilson_interval(k, n)
    return {"n": n, "successes": k, "rate": (k / n if n else None), "ci_lo": lo, "ci_hi": hi}


def main() -> None:
    if not generate.has_credentials():
        sys.exit("closed-book needs ANTHROPIC_API_KEY (it calls generate + the correctness judge).")

    rows = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    answerable = [r for r in rows if r["type"] in ANSWERABLE]

    gen = generate.Generator()
    from judge import Judge  # noqa: E402

    judge = Judge()

    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{git_info()['sha']}-closedbook"
    out_dir = RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=False)  # write-once, like run.py

    results: list[dict] = []
    for r in answerable:
        answer, _ = gen.answer(r["question"], [])  # EMPTY context (Option A) -- no retrieval
        c, _ = judge.correctness(r["question"], r["gold_answer"], answer)
        rec = {
            "id": r["id"], "type": r["type"], "question": r["question"],
            "answer": answer,
            "refusal_exact": is_exact_refusal(answer),  # did the guardrail hold (refuse w/ no context)?
            "correct": c["correct"],                    # correct DESPITE empty context = memory
            "judge_reason": c["reason"],
        }
        results.append(rec)
        print(f"  [{r['id']:<8}] {r['type']:<11} refuse={rec['refusal_exact']}  correct={rec['correct']}")

    types = sorted({r["type"] for r in answerable})
    summary = {
        "run_id": run_id,
        "control": "closed-book (D-013 item 14, Option A: contract + empty context)",
        "closed_book_correctness": {
            **{t: binom(lambda x: x["correct"], [r for r in results if r["type"] == t]) for t in types},
            "answerable_overall": binom(lambda x: x["correct"], results),
        },
        "guardrail_hold_rate": binom(lambda x: x["refusal_exact"], results),  # refused w/ empty context
        "note": "compare closed_book_correctness to the open-book baseline's correctness; "
                "open - closed = retrieval's lift. High closed-book => memory contamination (D-013).",
    }
    config = {
        "run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_info(),
        "control": "closed-book / no-context (D-013 item 14)",
        "no_context": True,
        "generator": {"id": generate.GENERATOR_MODEL,
                      "prompt_contract_version": generate.PROMPT_CONTRACT_VERSION},
        "judge": {"id": __import__("judge").JUDGE_MODEL, "rubric_version": __import__("judge").RUBRIC_VERSION},
        "question_set": {"path": str(QUESTIONS), "n_answerable": len(answerable),
                         "sha256": sha256_file(QUESTIONS)},
        "scored": "correctness only (no context -> groundedness N/A)",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def fmt(m: dict) -> str:
        return "n/a" if m["rate"] is None else f"{m['rate']:.2f} [{m['ci_lo']:.2f}, {m['ci_hi']:.2f}] (n={m['n']})"

    cb = summary["closed_book_correctness"]
    print(f"\n=== closed-book control {run_id} -> {out_dir} ===")
    print("closed-book correctness (correct DESPITE empty context = memory leak):")
    for t in types:
        print(f"  {t:<18}: {fmt(cb[t])}")
    print(f"  {'answerable_overall':<18}: {fmt(cb['answerable_overall'])}")
    print(f"guardrail-hold rate (refused with empty context): {fmt(summary['guardrail_hold_rate'])}")
    print("\nRead: closed-book ~0 CONFIRMS no contamination (correctness is retrieval-driven). "
          "High closed-book => fire the D-013 names-only swap.")


if __name__ == "__main__":
    main()
