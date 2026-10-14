"""Recommendation-quality eval (D-048..D-050): score a member-ranking method against the answer key
(analysis/answer_key_v3_haiku55.jsonl: every member x every question, graded 0/1/2 by Haiku 5.5 on
rubric v3). Local and free for embedding-only methods; every method is scored the same way.

Metrics, per question, then averaged:
  ndcg@5      are the 5 people shown the best ones, best first? (gain 3 for a 2, 1 for a 1;
              1.0 = the ideal 5 in the ideal order). The headline: the product shows 5 people.
  useful@5    share of the 5 shown that the key rates useful (1 or 2)
  best@5      share of the possible "clearly useful" (2) people that made the 5
              (denominator min(5, number of 2s))
  recall@k    share of ALL clearly useful (2) people ranked within the top k (10/20/50): did the
              search at least FIND them, even if not on screen? Low recall@50 = a retrieval
              problem (re-represent, hybrid, decompose); good recall@50 but low ndcg@5 = a
              ranking problem (rerank).

Questions dropped (D-050 follow-up, assistant rule): no member graded 2 (cannot test "found the
best"), or more than half the network graded 2 (every method looks perfect).

Usage:  python eval/recommend_eval.py <method>   (dense = today's search; see METHODS)
        python eval/recommend_eval.py compare <run_a> <run_b> [metric]
Writes runs/<ts>-<sha>-rec-<method>/ : config.json + summary.json (metrics only, committable) and
results.jsonl (ranked person_ids per question -- gitignored by the runs/** rule).
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

KEY = ROOT / "analysis" / "answer_key_v3_haiku55.jsonl"
QUESTIONS = ROOT / "eval" / "recommend_questions.jsonl"
SHOWN = 5                    # the product's member cap (tools.TOP_N_MEMBERS)
RECALL_KS = (10, 20, 50)
GAIN = {0: 0, 1: 1, 2: 3}
# TUNABLE(drop a question when more than half the network is "clearly useful": every method
#   scores ~perfect on it. Symptom wrong: a kept question shows ndcg@5 ~1.0 for every method.)
MAX_SHARE_OF_2S = 0.5


def load_key() -> tuple[dict[str, dict[str, int]], list[dict]]:
    grades: dict[str, dict[str, int]] = defaultdict(dict)
    for line in KEY.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        grades[r["qid"]][r["person_id"]] = r["grade"]
    qs = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l]
    n = len(next(iter(grades.values())))
    kept = [q for q in qs if 0 < sum(g == 2 for g in grades[q["id"]].values()) <= MAX_SHARE_OF_2S * n]
    return grades, kept


def score(ranked: list[str], g: dict[str, int]) -> dict:
    twos = {p for p, v in g.items() if v == 2}
    dcg = sum(GAIN[g.get(p, 0)] / math.log2(i + 2) for i, p in enumerate(ranked[:SHOWN]))
    ideal = sorted((GAIN[v] for v in g.values()), reverse=True)[:SHOWN]
    idcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
    out = {"ndcg@5": dcg / idcg,
           "useful@5": sum(g.get(p, 0) > 0 for p in ranked[:SHOWN]) / SHOWN,
           "best@5": len(twos & set(ranked[:SHOWN])) / min(SHOWN, len(twos))}
    for k in RECALL_KS:
        out[f"recall@{k}"] = len(twos & set(ranked[:k])) / len(twos)
    return out


# --- methods: question text -> every member ranked, best first ---------------------------------

# How a person's section similarities become one score. "max" is today's find_members (a person
# = their single best-matching section); the others reward people who match in several sections
# (the multi-need hypothesis). Rows arrive best-first, so s[:n] are the person's top n.
AGG = {"max": lambda s: s[0], "sum2": lambda s: sum(s[:2]), "sum3": lambda s: sum(s[:3]),
       "mean3": lambda s: sum(s[:3]) / len(s[:3])}
# TUNABLE(the standard reciprocal-rank-fusion constant; it damps the top ranks so one list cannot
#   dominate. Symptom wrong: hybrid ~= whichever single list it merges -> try 20 or 100.)
RRF_K = 60
# BM25 (Okapi) standard constants. TUNABLE(symptom wrong: long sections dominate keyword hits ->
#   raise b; repeated words over-reward -> lower k1.)
BM25_K1, BM25_B = 1.2, 0.75
_STOP = set("""a an and are as at be but by can could do does for from has have he her his how i
if in into is it its me my of on or our she should so than that the their them they this to was we
what which who whom why will with would you your here there network members member people person
someone anyone help""".split())


def _tokens(text: str) -> list[str]:
    import re
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 1]


class BM25:
    """Okapi BM25 over the section chunks, in memory (4,006 sections: build <1 s, query ~ms).
    Twenty lines instead of a dependency; Postgres full-text ranking is not BM25."""

    def __init__(self, docs: list[str]):
        from collections import Counter
        self.tf = [Counter(_tokens(d)) for d in docs]
        self.lens = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.lens) / len(self.lens)
        df = Counter(t for c in self.tf for t in c)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - k + 0.5) / (k + 0.5)) for t, k in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = set(_tokens(query))
        out = []
        for c, length in zip(self.tf, self.lens):
            norm = BM25_K1 * (1 - BM25_B + BM25_B * length / self.avg)
            out.append(sum(self.idf[t] * c[t] * (BM25_K1 + 1) / (c[t] + norm)
                           for t in terms if t in c))
        return out


def _dense_people(conn, emb, question: str) -> dict[str, list[float]]:
    import store
    from tools import SCHEME
    per: dict[str, list[float]] = defaultdict(list)
    for row in store.search(conn, emb.embed_query(question), k=100_000, scheme=SCHEME):
        per[row["person_id"]].append(row["score"])
    return per


def _rank(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=lambda p: -scores[p])


def _rrf(*rankings: list[str]) -> list[str]:
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for i, p in enumerate(ranking):
            fused[p] += 1 / (RRF_K + i + 1)
    return _rank(fused)


def method_dense(conn, emb, agg: str = "max"):
    """Meaning-based search; agg "max" is today's member search (find_members' ordering,
    extended past its 5-person cap so recall@k can look deeper)."""
    def rank(question: str) -> list[str]:
        return _rank({p: AGG[agg](s) for p, s in _dense_people(conn, emb, question).items()})
    return rank


def method_bm25(conn, emb):
    """Keyword search alone (diagnostic): a person = their best-matching section by BM25."""
    from tools import SCHEME
    with conn.cursor() as cur:
        cur.execute("SELECT person_id, text FROM chunks WHERE chunk_scheme = %s", (SCHEME,))
        rows = cur.fetchall()
    index, section_people = BM25([r[1] for r in rows]), [r[0] for r in rows]
    everyone = sorted(set(section_people))

    def rank(question: str) -> list[str]:
        best: dict[str, float] = {p: 0.0 for p in everyone}
        for p, s in zip(section_people, index.scores(question)):
            best[p] = max(best[p], s)
        return _rank(best)
    return rank


def method_hybrid(conn, emb, agg: str = "max"):
    """Keyword + meaning, merged by reciprocal rank fusion."""
    dense, keyword = method_dense(conn, emb, agg), method_bm25(conn, emb)
    return lambda question: _rrf(dense(question), keyword(question))


METHODS = {"dense": method_dense, "bm25": method_bm25,
           **{f"dense_{a}": (lambda a: lambda c, e: method_dense(c, e, a))(a)
              for a in ("sum2", "sum3", "mean3")},
           "hybrid": method_hybrid,
           **{f"hybrid_{a}": (lambda a: lambda c, e: method_hybrid(c, e, a))(a)
              for a in ("sum2", "sum3", "mean3")}}


def compare(run_a: str, run_b: str, metric: str = "ndcg@5") -> None:
    """B vs A question by question: wins/losses and a paired bootstrap 95% CI on the mean
    difference. 34 questions is small: a mean gain whose CI crosses 0 is not yet a gain."""
    import random
    pa = json.loads((ROOT / "runs" / run_a / "summary.json").read_text())["per_question"]
    pb = json.loads((ROOT / "runs" / run_b / "summary.json").read_text())["per_question"]
    diffs = [pb[q][metric] - pa[q][metric] for q in pa if q in pb]
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(10_000))
    print(f"{run_b.split('-rec-')[-1]} vs {run_a.split('-rec-')[-1]} on {metric}: "
          f"mean {sum(diffs) / len(diffs):+.3f} [95% CI {boots[250]:+.3f}, {boots[9750]:+.3f}] | "
          f"wins {sum(d > 0.005 for d in diffs)}, losses {sum(d < -0.005 for d in diffs)}, "
          f"ties {sum(abs(d) <= 0.005 for d in diffs)}")


def main(method: str) -> None:
    import store
    from embedder import NomicLocal

    grades, kept = load_key()
    conn, emb = store.connect(), NomicLocal()
    rank = METHODS[method](conn, emb)
    rank("warm-up")                         # model + index load, kept out of the timing
    per_q, results, elapsed = {}, [], 0.0
    for q in kept:
        t0 = time.perf_counter()
        ranked = rank(q["question"])
        elapsed += time.perf_counter() - t0
        per_q[q["id"]] = score(ranked, grades[q["id"]])
        results.append({"qid": q["id"], "ranked": ranked[:max(RECALL_KS)]})
    metrics = list(next(iter(per_q.values())))
    overall = {m: round(sum(s[m] for s in per_q.values()) / len(per_q), 3) for m in metrics}
    by_tag = defaultdict(list)
    for q in kept:
        for t in q["tags"]:
            by_tag[t].append(per_q[q["id"]]["ndcg@5"])

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         cwd=ROOT).stdout.strip()
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{sha}-rec-{method}"
    out = ROOT / "runs" / run_id
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({
        "run_id": run_id, "runner": "eval/recommend_eval.py", "method": method,
        "answer_key": KEY.name, "rubric": "v3", "grader": "claude-haiku-5-5",
        "questions_kept": [q["id"] for q in kept], "shown": SHOWN, "gain": GAIN,
        "max_share_of_2s": MAX_SHARE_OF_2S}, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({
        "n_questions": len(kept), "overall": overall,
        "ms_per_question": round(1000 * elapsed / len(kept)),
        "ndcg@5_by_tag": {t: round(sum(v) / len(v), 3) for t, v in sorted(by_tag.items())},
        "per_question": {q: {m: round(v, 3) for m, v in s.items()} for q, s in per_q.items()}},
        indent=1), encoding="utf-8")
    (out / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n",
                                       encoding="utf-8")
    print(f"{run_id}: {len(kept)} questions, {1000 * elapsed / len(kept):.0f} ms per question")
    for m, v in overall.items():
        print(f"  {m:10} {v:.3f}")
    print("  ndcg@5 by question type: " + ", ".join(
        f"{t} {sum(v) / len(v):.2f} (n={len(v)})" for t, v in sorted(by_tag.items())))


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(*sys.argv[2:])
    else:
        main(sys.argv[1])
