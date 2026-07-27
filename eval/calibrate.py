"""Judge calibration harness (DECISIONS D-022; GAPS G-002; Phase 1e, Tier-2 items 9+10).

Checks the ruler, in both directions, and measures whether it reads the same twice:

  1. NATURAL slice -- does the Sonnet judge agree with a human on the REAL 1d verdicts?
     The run's groundedness pool is 33/33 grounded=true (the bot refuses instead of
     hallucinating, D-013), so this only tests the FALSE-ALARM direction: does the judge
     wrongly FAIL a genuinely-grounded answer? report-only, no gate -- a false "ungrounded"
     just makes the bot look worse than it is (the safe direction).
  2. INJECTED slice -- the run cannot produce an ungrounded answer, so we plant them
     (eval/calibration_injected.jsonl: a real question + its real retrieved context + an
     answer with ONE planted unsupported claim). This tests the DANGEROUS direction: does the
     judge CATCH a lie? ZERO-TOLERANCE (D-022) -- one trip-bucket item the judge passes as
     grounded trips the alarm (=> bump RUBRIC_VERSION, re-score all runs). Borderline-bucket
     items (manufactured superlative / elaboration drift) are report-only, NEVER trip -- they
     are the gray zone where careful humans themselves disagree (owner's own refinement).
  3. FLIP-RATE (item 10) -- Sonnet-5 rejects temperature (D-010 amendment), so stability is
     MEASURED: judge each injected + borderline item FLIP_RUNS times; flip-rate = fraction
     non-unanimous. Measured on the hard cases only -- a slam-dunk grounded answer never
     wobbles, so it would give a falsely reassuring 0%.

The human labels BLIND: injected and natural items are shuffled together into one file with
the judge's verdicts stripped, so the labeler cannot tell which is which (or what the judge
said) before labeling. Cohen's kappa is DEFERRED until >=50 cumulative mixed labels (D-022);
at n=33 all-true the human labels have no variance and kappa is undefined -- we report raw
agreement now.

Subcommands (all take --run <run_id>, default = the 1d baseline of record):
  build          reconstruct context, sample the natural slice, emit blind files + manifest
  judge-injected run the Sonnet judge on the injected items FLIP_RUNS x each (needs API $)
  score          join human labels + judge verdicts + manifest -> report.json + printout

BLIND-PROTOCOL RULE (learned the hard way 2026-07-27): the labeler must not see the judge's
verdicts OR the set's composition (how many plants / grounded / gray-zone) before labeling --
either leaks anchor the human and contaminate the independence being measured. So: LABEL FIRST,
judge is silent (prints only a progress counter), and nobody narrates results until `score`.

Usage (ORDER MATTERS -- label before you see anything):
  ./.venv/Scripts/python.exe eval/calibrate.py build
  # ... LABEL calibration/<run>/groundedness_labels.csv (+ correctness_labels.csv) BLIND ...
  ./.venv/Scripts/python.exe eval/calibrate.py judge-injected   # prints no verdicts
  ./.venv/Scripts/python.exe eval/calibrate.py score
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "eval")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()  # read .env into os.environ (mirrors eval/run.py) so the judge sees the API key

# The 1d baseline of record (D-013 close). Override with --run.
DEFAULT_RUN = "20260726T091259Z-2158c98"
RUNS_DIR = Path("runs")
CALIB_DIR = Path("calibration")
QUESTIONS = Path("eval/questions.jsonl")
INJECTED = Path("eval/calibration_injected.jsonl")

# item 10: 3 = cheapest odd number that yields a majority; 5 barely moves the estimate at this n.
# TUNABLE(3 runs, majority-of-3 above ~5-10% non-unanimous; symptom: A/B groundedness deltas you
#   act on are SMALLER than the measured flip-rate => the ruler's wobble dominates the signal.)
FLIP_RUNS = 3
FLIP_ALARM = 0.10  # >10% non-unanimous => switch official runs to majority-of-3 (D-022)

# Deterministic natural slice (D-022 ~8/8/... budget). Fixed id lists, NOT random, so `build` is
# reproducible and reviewable. NONE of these may be an injected base_qid, or the human would see
# two near-identical answers and spot the plant (blindness break). Bases used by the injected set:
# sh-027 sh-022 sh-020 sh-015 sh-001 sh-030 sh-004 sh-011.
NATURAL_GROUNDED = [  # all grounded=true in the 1d run -> tests the false-alarm direction only
    "mh-002",  # multi-hop, grounded=true (but correct=false -- reused in correctness slice)
    "sh-013",  # relays the report's quoted "world's largest LED supplier" -> judge must NOT flag
    "sh-024",  # "became a staple in the BMW track community" -> grounded-elaboration false-alarm test
    "sh-002",  # list-shaped answer (industries)
    "sh-035",  # a range ("8-10 weeks")
    "sh-038",  # a date
    "sh-006",  # a year
    "sh-025",  # a migration fact (QuickBooks -> NetSuite)
]
NATURAL_CORRECT = [  # correctness lane -- HAS natural negatives (unlike groundedness)
    "mh-002",  # grounded=true BUT correct=false -- the ONE fluent-but-wrong case (must include)
    "sh-007", "sh-016", "mh-001", "sh-034",  # false-refusals (answerable, refused) -> correct=false
    "sh-005", "sh-021", "sh-028",  # grounded + correct answers -> correct=true
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def blind_order(item_id: str) -> str:
    """Stable pseudo-random sort key so the blind file's order leaks nothing about source
    (natural vs injected) or id. Deterministic (md5), so `build` and `score` agree."""
    return hashlib.md5(item_id.encode()).hexdigest()


def reconstruct_contexts(questions_by_qid: dict[str, str], expected_docs: dict[str, list[str]]):
    """Re-run the REAL retrieval path (embed_query -> pgvector top-k) to rebuild the exact context
    string the judge originally scored. Retrieval is deterministic (cached nomic embeddings + exact
    seqscan), so reconstruction == the original context. Loads NomicLocal once (~17s). Warns if a
    reconstructed doc set diverges from the run's recorded retrieved doc_ids (would signal the DB or
    embedder changed under us)."""
    import generate
    from retrieve import Retriever

    r = Retriever()
    contexts: dict[str, str] = {}
    for qid, question in questions_by_qid.items():
        retrieved = r.retrieve(question)
        got = [h["doc_id"] for h in retrieved]
        exp = expected_docs.get(qid)
        if exp is not None and got != exp:
            print(f"  !! WARN {qid}: reconstructed docs {got} != recorded {exp} "
                  f"(retrieval drifted -- context may not match the scored run)")
        contexts[qid] = generate.format_context(retrieved)
    return contexts


def cmd_build(run_id: str) -> None:
    run_dir = RUNS_DIR / run_id
    results = {r["id"]: r for r in load_jsonl(run_dir / "results.jsonl")}
    gold = {r["id"]: r for r in load_jsonl(QUESTIONS)}
    injected = load_jsonl(INJECTED)
    out = CALIB_DIR / run_id
    out.mkdir(parents=True, exist_ok=True)

    # sanity: the natural picks must all be present and NOT be injected bases
    inj_bases = {inj["base_qid"] for inj in injected}
    for qid in set(NATURAL_GROUNDED) | set(NATURAL_CORRECT):
        assert qid in results, f"{qid} not in run {run_id}"
    clash = (set(NATURAL_GROUNDED) & inj_bases)
    assert not clash, f"natural-grounded picks clash with injected bases (blindness break): {clash}"

    # --- reconstruct contexts for everything the groundedness lane needs ---
    need_qids = {qid: results[qid]["question"] for qid in NATURAL_GROUNDED}
    for inj in injected:
        b = inj["base_qid"]
        need_qids[b] = results[b]["question"]  # injected answer reuses its base question + context
    expected_docs = {qid: [h["doc_id"] for h in results[qid]["retrieved"]] for qid in need_qids}
    print(f"reconstructing {len(need_qids)} contexts (loads the embedder once ~17s)...")
    contexts = reconstruct_contexts(need_qids, expected_docs)

    # --- assemble the blind GROUNDEDNESS set: natural-grounded + injected, shuffled ---
    g_items: list[dict] = []
    for qid in NATURAL_GROUNDED:
        rr = results[qid]
        g_items.append({"source": "natural", "real_id": qid, "kind": "natural-grounded",
                        "bucket": "natural", "question": rr["question"],
                        "context": contexts[qid], "answer": rr["answer"],
                        "judge_grounded": rr["grounded"], "expected_grounded": None})
    for inj in injected:
        g_items.append({"source": "injected", "real_id": inj["id"], "kind": inj["kind"],
                        "bucket": inj["bucket"], "question": results[inj["base_qid"]]["question"],
                        "context": contexts[inj["base_qid"]], "answer": inj["answer"],
                        "judge_grounded": None, "expected_grounded": inj["expected_grounded"],
                        "planted": inj["planted"]})
    g_items.sort(key=lambda x: blind_order(x["real_id"]))
    for i, it in enumerate(g_items, 1):
        it["item"] = f"G{i:02d}"

    # --- assemble the CORRECTNESS set (natural only; judge sees gold + answer, NO context) ---
    c_items: list[dict] = []
    for qid in NATURAL_CORRECT:
        rr = results[qid]
        c_items.append({"source": "natural", "real_id": qid, "question": rr["question"],
                        "gold_answer": gold[qid]["gold_answer"], "answer": rr["answer"],
                        "judge_correct": rr["correct"]})
    c_items.sort(key=lambda x: blind_order("c-" + x["real_id"]))
    for i, it in enumerate(c_items, 1):
        it["item"] = f"C{i:02d}"

    # --- manifest (HIDDEN key: source, real_id, judge verdict, expected) ---
    g_keys = ("item", "source", "real_id", "kind", "bucket", "judge_grounded", "expected_grounded")
    g_manifest = []
    for it in g_items:
        m = {k: it[k] for k in g_keys}
        if it["source"] == "injected":  # keep the plant note for post-hoc reading (labeler never sees it)
            m["planted"] = it["planted"]
        g_manifest.append(m)
    manifest = {
        "run_id": run_id,
        "groundedness": g_manifest,
        "correctness": [{k: it[k] for k in ("item", "source", "real_id", "judge_correct")}
                        for it in c_items],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # --- blind, human-readable MD (contexts are long; MD reads better than a CSV cell) ---
    _write_groundedness_md(out / "groundedness_TO_READ.md", g_items)
    _write_labels_csv(out / "groundedness_labels.csv", g_items,
                      "your_verdict", "grounded | ungrounded")
    _write_correctness_md(out / "correctness_TO_READ.md", c_items)
    _write_labels_csv(out / "correctness_labels.csv", c_items,
                      "your_verdict", "correct | incorrect")

    print(f"\nbuilt -> {out}")
    print(f"  groundedness: {len(g_items)} items "
          f"({sum(1 for x in g_items if x['source']=='injected')} injected, "
          f"{sum(1 for x in g_items if x['source']=='natural')} natural)")
    print(f"  correctness : {len(c_items)} items")
    print("NEXT: run `judge-injected`, then label the two *_labels.csv files, then `score`.")


def _write_groundedness_md(path: Path, items: list[dict]) -> None:
    lines = ["# Groundedness labeling (BLIND)", "",
             "For each item, decide: is EVERY factual claim in the ANSWER supported by the",
             "REPORTS shown -- and ONLY those reports? A claim that is true in the real world but",
             "not in the reports is UNGROUNDED. A refusal makes no claims -> grounded.",
             "Write `grounded` or `ungrounded` in groundedness_labels.csv next to each item.", ""]
    for it in items:
        lines += [f"## {it['item']}", "", f"**Question:** {it['question']}", "",
                  "**Reports shown:**", "", "```", it["context"], "```", "",
                  f"**Answer:** {it['answer']}", "",
                  f"YOUR VERDICT ({it['item']}): ______  (grounded / ungrounded)", "", "---", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_correctness_md(path: Path, items: list[dict]) -> None:
    lines = ["# Correctness labeling (BLIND)", "",
             "Does the ANSWER convey the same facts as the GOLD answer? Ignore wording/citations.",
             "If gold is a refusal, the answer is correct only if it also declines.",
             "Write `correct` or `incorrect` in correctness_labels.csv next to each item.", ""]
    for it in items:
        lines += [f"## {it['item']}", "", f"**Question:** {it['question']}", "",
                  f"**Gold answer:** {it['gold_answer']}", "",
                  f"**Answer:** {it['answer']}", "",
                  f"YOUR VERDICT ({it['item']}): ______  (correct / incorrect)", "", "---", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_labels_csv(path: Path, items: list[dict], col: str, hint: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", col, f"# fill with: {hint}"])
        for it in items:
            w.writerow([it["item"], "", ""])


def cmd_judge_injected(run_id: str) -> None:
    """Run the Sonnet groundedness judge on each injected item FLIP_RUNS x (trip check + flip-rate).
    Also re-judges the natural multi-hop item mh-002 (a borderline real case) FLIP_RUNS x, so the
    flip-set includes a real hard case, not only synthetic ones."""
    import generate
    from judge import Judge

    if not generate.has_credentials():
        sys.exit("ANTHROPIC_API_KEY not set -- judge-injected needs the API.")
    run_dir = RUNS_DIR / run_id
    results = {r["id"]: r for r in load_jsonl(run_dir / "results.jsonl")}
    injected = load_jsonl(INJECTED)
    out = CALIB_DIR / run_id

    # contexts for the injected bases + mh-002
    need = {inj["base_qid"]: results[inj["base_qid"]]["question"] for inj in injected}
    need["mh-002"] = results["mh-002"]["question"]
    expected_docs = {q: [h["doc_id"] for h in results[q]["retrieved"]] for q in need}
    print(f"reconstructing {len(need)} contexts...")
    contexts = reconstruct_contexts(need, expected_docs)

    judge = Judge()
    flip_targets = [{"id": inj["id"], "question": results[inj["base_qid"]]["question"],
                     "context": contexts[inj["base_qid"]], "answer": inj["answer"],
                     "bucket": inj["bucket"], "expected_grounded": inj["expected_grounded"]}
                    for inj in injected]
    flip_targets.append({"id": "mh-002", "question": results["mh-002"]["question"],
                         "context": contexts["mh-002"], "answer": results["mh-002"]["answer"],
                         "bucket": "natural-borderline", "expected_grounded": None})

    # BLINDNESS: print NO verdicts, ids, or kinds -- only an opaque progress counter. Revealing
    # what the judge said (or even the set's composition) before the human labels would anchor the
    # labeler and contaminate the independence the calibration depends on (learned 2026-07-27).
    records = []
    for i, t in enumerate(flip_targets, 1):
        verdicts = []
        for _run_i in range(FLIP_RUNS):
            g, _meta = judge.groundedness(t["question"], t["context"], t["answer"])
            verdicts.append({"grounded": g["grounded"], "is_refusal": g["is_refusal"],
                             "reason": g["reason"]})
        records.append({"id": t["id"], "bucket": t["bucket"],
                        "expected_grounded": t["expected_grounded"], "verdicts": verdicts})
        print(f"  judged {i}/{len(flip_targets)} items")
    (out / "judge_injected.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\ndone -> {len(records)} items x {FLIP_RUNS} runs written (verdicts hidden until `score`).")


_VERDICT_TOKENS = {"grounded", "ungrounded", "correct", "incorrect"}


def _read_labels(path: Path) -> dict[str, str]:
    """Map item -> verdict token. Robust to WHICH column the label was typed in: scan the row for
    a recognized verdict word (the fillable CSV has a hint column that invited stray entries)."""
    labels = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip() or row[0].strip() == "item":
                continue
            found = [c.strip().lower() for c in row[1:] if c.strip().lower() in _VERDICT_TOKENS]
            labels[row[0].strip()] = found[0] if found else ""
    return labels


def _majority_grounded(verdicts: list[dict]) -> bool:
    return sum(1 for v in verdicts if v["grounded"]) > len(verdicts) / 2


def cmd_score(run_id: str) -> None:
    out = CALIB_DIR / run_id
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    g_labels = _read_labels(out / "groundedness_labels.csv")
    c_labels = _read_labels(out / "correctness_labels.csv")
    judged = {r["id"]: r for r in json.loads((out / "judge_injected.json").read_text(encoding="utf-8"))}

    missing_g = [m["item"] for m in manifest["groundedness"] if not g_labels.get(m["item"])]
    if missing_g:
        sys.exit(f"unlabeled groundedness items (fill the CSV first): {missing_g}")
    # correctness is OPTIONAL (secondary, mechanical lane): score it only if FULLY labeled;
    # skip cleanly if untouched; refuse a half-done file (a partial rate would mislead).
    labeled_c = [m["item"] for m in manifest["correctness"] if c_labels.get(m["item"])]
    score_correctness = len(labeled_c) == len(manifest["correctness"])
    if labeled_c and not score_correctness:
        sys.exit(f"correctness partially labeled ({len(labeled_c)}/{len(manifest['correctness'])}) "
                 f"-- finish it or clear it; a partial rate misleads.")

    # --- 1. natural groundedness: human vs judge (report-only, false-alarm direction) ---
    natural = [m for m in manifest["groundedness"] if m["source"] == "natural"]
    nat_agree = sum(1 for m in natural
                    if (g_labels[m["item"]] == "grounded") == bool(m["judge_grounded"]))
    nat_false_alarm = [m["real_id"] for m in natural
                       if g_labels[m["item"]] == "ungrounded" and m["judge_grounded"]]

    # --- 2. injected: judge (majority-of-FLIP_RUNS) vs planted truth; zero-tolerance on TRIP ---
    trip_hits, borderline, plant_bad = [], [], []
    for m in manifest["groundedness"]:
        if m["source"] != "injected":
            continue
        rid, human = m["real_id"], g_labels[m["item"]]
        jverd = _majority_grounded(judged[rid]["verdicts"])  # True = judge said grounded
        if m["bucket"] == "trip":
            if human != "ungrounded":  # plant failed: a careful human didn't see it as ungrounded
                plant_bad.append(rid)
            elif jverd:  # human agrees it's ungrounded, but the judge PASSED it -> alarm
                trip_hits.append(rid)
        else:  # borderline -- report human vs judge, NEVER trips
            borderline.append({"id": rid, "kind": m["kind"], "human": human,
                               "judge_grounded": jverd})

    # --- 3. correctness: human vs judge (report-only; OPTIONAL lane) ---
    if score_correctness:
        corr_agree = sum(1 for m in manifest["correctness"]
                         if (c_labels[m["item"]] == "correct") == bool(m["judge_correct"]))
        correctness_block = {
            "n": len(manifest["correctness"]),
            "agreement": round(corr_agree / len(manifest["correctness"]), 3),
            "disagreements": [{"id": m["real_id"], "human": c_labels[m["item"]],
                               "judge_correct": m["judge_correct"]}
                              for m in manifest["correctness"]
                              if (c_labels[m["item"]] == "correct") != bool(m["judge_correct"])],
        }
    else:
        correctness_block = {"n": 0, "skipped": "correctness lane not labeled"}

    # --- 4. flip-rate over the hard set (injected + mh-002) ---
    flips = []
    for rid, rec in judged.items():
        gs = [v["grounded"] for v in rec["verdicts"]]
        if len(set(gs)) > 1:
            flips.append({"id": rid, "verdicts": gs})
    flip_rate = len(flips) / len(judged) if judged else None

    tripped = bool(trip_hits)
    report = {
        "run_id": run_id,
        "trip_rule": "asymmetric zero-tolerance on injected trip-bucket (D-022)",
        "ALARM_TRIPPED": tripped,
        "natural_groundedness": {
            "n": len(natural), "agreement": round(nat_agree / len(natural), 3),
            "false_alarms_judge_said_grounded_human_ungrounded": nat_false_alarm,
            "note": "report-only; false-alarm direction is safe (bot looks worse, not wrong)",
        },
        "injected_trip": {
            "n_trip": sum(1 for m in manifest["groundedness"]
                          if m.get("bucket") == "trip"),
            "judge_missed_a_planted_lie": trip_hits,  # non-empty => TRIP
            "plants_human_did_not_confirm_ungrounded": plant_bad,  # authoring issue, not judge
        },
        "injected_borderline": borderline,  # gray zone: report human vs judge, never trips
        "correctness": correctness_block,
        "flip_rate": {
            "n_items": len(judged), "runs_each": FLIP_RUNS, "rate": flip_rate,
            "non_unanimous": flips, "alarm_threshold": FLIP_ALARM,
            "use_majority_of_n": (flip_rate or 0) > FLIP_ALARM,
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_report(report)


def _print_report(r: dict) -> None:
    ng, it, corr, fr = (r["natural_groundedness"], r["injected_trip"],
                        r["correctness"], r["flip_rate"])
    print(f"\n=== calibration report: {r['run_id']} ===")
    print(f"\n[1] natural groundedness (false-alarm direction, report-only)")
    print(f"    human vs judge agreement: {ng['agreement']:.0%}  (n={ng['n']})")
    if ng["false_alarms_judge_said_grounded_human_ungrounded"]:
        print(f"    !! human called these ungrounded but the run scored them grounded: "
              f"{ng['false_alarms_judge_said_grounded_human_ungrounded']}")
    print(f"\n[2] injected trip test (dangerous direction, ZERO-TOLERANCE)")
    print(f"    trip-bucket items: {it['n_trip']}")
    if it["plants_human_did_not_confirm_ungrounded"]:
        print(f"    (note: plants a human did NOT confirm ungrounded -- authoring issue, excluded "
              f"from the trip test: {it['plants_human_did_not_confirm_ungrounded']})")
    if it["judge_missed_a_planted_lie"]:
        print(f"    *** ALARM: judge PASSED planted lie(s) as grounded: "
              f"{it['judge_missed_a_planted_lie']} ***")
        print(f"    *** D-022 trip: bump RUBRIC_VERSION and re-score ALL runs. ***")
    else:
        print(f"    judge caught every confirmed planted lie. No trip.")
    print(f"\n    borderline (gray zone, report-only):")
    for b in r["injected_borderline"]:
        print(f"      {b['id']:<12} ({b['kind']}): human={b['human']:<10} "
              f"judge_grounded={b['judge_grounded']}")
    if corr.get("skipped"):
        print(f"\n[3] correctness (report-only): SKIPPED -- {corr['skipped']}")
    else:
        print(f"\n[3] correctness (report-only): agreement {corr['agreement']:.0%} (n={corr['n']})")
        for d in corr["disagreements"]:
            print(f"      disagree {d['id']}: human={d['human']} judge_correct={d['judge_correct']}")
    print(f"\n[4] flip-rate (hard cases only): {fr['rate']:.0%} non-unanimous "
          f"({len(fr['non_unanimous'])}/{fr['n_items']} items, {fr['runs_each']} runs each)")
    for fl in fr["non_unanimous"]:
        print(f"      wobbled {fl['id']}: {fl['verdicts']}")
    if fr["use_majority_of_n"]:
        print(f"    >{fr['alarm_threshold']:.0%} => use majority-of-{fr['runs_each']} for "
              f"comparison runs (D-022).")
    else:
        print(f"    <= {fr['alarm_threshold']:.0%} => single-call judge is stable enough.")
    print(f"\nALARM_TRIPPED = {r['ALARM_TRIPPED']}")


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] not in {"build", "judge-injected", "score"}:
        sys.exit(__doc__)
    cmd = args[0]
    run_id = DEFAULT_RUN
    if "--run" in args:
        run_id = args[args.index("--run") + 1]
    {"build": cmd_build, "judge-injected": cmd_judge_injected, "score": cmd_score}[cmd](run_id)


if __name__ == "__main__":
    main()
