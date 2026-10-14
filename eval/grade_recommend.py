"""Recommendation-quality answer key (D-048): Claude grades how useful a member would be for a
networking question, 0/1/2, via the Message Batches API (half price; grading is not latency-bound).

  bakeoff <label_sheet_KEY.json>   submit: every candidate grader x every blind-labelled pair, on
                                   the SAME excerpt the human reviewer graded (judgement on equal information)
  collect <batch_id> <out.jsonl>   poll until the batch ends; write one row per request; print
                                   counts and cost only -- never grades (the reviewer labels blind)
  full <grader> <out_prefix>       submit the answer key: every member x every question, one
                                   request per member with the whole dossier
  collect_full <index.json> <out>  poll; one row per (member, question); completeness + cost
  agree <labels.json> <results>    each grader vs the reviewer's labels (run after labelling)

Grades never print here: the reviewer's labels come first, agreement is computed afterwards.
Outputs go under analysis/ (gitignored): they carry member names and dossier excerpts.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
import anthropic  # noqa: E402

# Candidate graders and how to switch each one's extended thinking OFF (per the current API
# reference: Haiku 4.5 has none by default; Haiku 5.5 accepts "disabled" at effort <= high;
# Sonnet 5.5 rejects "disabled" and uses "between_tools"). Thinking is billed as output and the
# bake-off compares judgement, so all three grade without it.
GRADERS = {
    "haiku-4-5": {"model": "claude-haiku-4-5"},
    "haiku-5-5": {"model": "claude-haiku-5-5", "thinking": {"type": "disabled"}},
    "sonnet-5-5": {"model": "claude-sonnet-5-5", "thinking": {"type": "between_tools"}},
}
# $/MTok (input, output), standard API rates as of 2026-10; the batch discount halves both.
PRICES = {"claude-haiku-4-5": (1.0, 5.0), "claude-haiku-5-5": (0.10, 0.50),
          "claude-sonnet-5-5": (2.0, 10.0)}
BATCH_DISCOUNT = 0.5
POLL_S = 30  # batches usually finish within an hour; 30 s keeps the poll cheap and responsive

# Rubric v3 (D-049 + D-050). v1 (bare 0/1/2 scale) agreed with the human reviewer at kappa 0.17-0.25 on
# teaching questions; v2 added the reviewer's core-vs-context rule and still scored at chance on the
# blind test (Q8-10), whose disagreements surfaced three more reviewer rules (role words, absolute
# exclusions, perspective questions) plus a leniency gap (graders promoted many of the reviewer's
# 0s). v3 = all of it. Not validated on unseen questions (a deliberate trade-off, D-050); the safeguard is
# the first method comparison, where the reviewer's blind list preferences check the key.
RUBRIC_VERSION = "v3"
RUBRIC = """Grade how useful this person would be to recommend for THIS question.

First split the question into:
- the CORE ASK: what the recommended person must be able to help with ("help me enter Japan, Australia or New Zealand", "practical turnaround advice", "selling to governments");
- the CONTEXT: who is asking and their situation ("my technology business", "a B2B startup", "a deep-tech founder").

Then grade:
2 = meets the core ask clearly AND fits the context (if the question gives no context, meeting the core ask clearly is enough)
1 = meets the core ask but not the context, OR meets the core ask only partly (knows Asia but not the named countries; has one of two required experiences)
0 = does not meet the core ask, even if the context fits perfectly

Rules:
- Role words are part of the core ask. "Which CEOs have done M&A?" needs a CEO: a finance executive with M&A experience scores 0.
- Exclusions are absolute. If the asker rules something out ("rather than fundraising advice", "not investors or consultants", "rather than five similar technology CEOs"), a person who mainly fits the excluded description scores 0, with no partial credit.
- For questions asking for different or varied perspectives (a roundtable, a panel, a board), the core ask is the perspective the person would add, not expertise in the topic. A leader from a traditional industry can be a 2 for "how traditional industries can adopt AI" without being an AI expert.
- Most members are not relevant to any given question. Give 1 or 2 only for direct, demonstrated experience in the profile, not for adjacent or general business experience. Sharing words with the question is not relevance.

Example. "I'm expanding my technology business into Japan, Australia or New Zealand. Who could help me enter?" A technology CEO with no Asia-Pacific experience = 0 (misses the core ask). Someone who built a retail business in Australia = 1 (core met, context not). Someone who took a software company into Japan = 2."""

SYSTEM = ("You grade candidate recommendations for a private network of business leaders. A member "
          "asked the question below; you see an excerpt from another member's profile. "
          + RUBRIC + "\n\nJudge from the excerpt alone. Give the grade and one short sentence "
          "of reason.")

SCHEMA = {"type": "object",
          "properties": {"grade": {"type": "integer", "enum": [0, 1, 2]},
                         "reason": {"type": "string"}},
          "required": ["grade", "reason"], "additionalProperties": False}


def _params(grader: str, question: str, excerpt: str) -> dict:
    g = GRADERS[grader]
    params = {
        "model": g["model"],
        "max_tokens": 300,
        "system": SYSTEM,
        "messages": [{"role": "user",
                      "content": f"Question:\n{question}\n\nProfile excerpt:\n{excerpt}"}],
        "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
    }
    if "thinking" in g:
        params["thinking"] = g["thinking"]
    return params


def bakeoff(key_path: str) -> None:
    pairs = json.loads(Path(key_path).read_text(encoding="utf-8"))["pairs"]
    requests = [{"custom_id": f"{grader}__{p['label']}__{RUBRIC_VERSION}",
                 "params": _params(grader, p["question"], p["excerpt"])}
                for grader in GRADERS for p in pairs]
    batch = anthropic.Anthropic().messages.batches.create(requests=requests)
    print(f"submitted batch {batch.id} (rubric {RUBRIC_VERSION}): {len(requests)} requests "
          f"({len(GRADERS)} graders x {len(pairs)} pairs)")


def collect(batch_id: str, out_path: str) -> None:
    client = anthropic.Anthropic()
    while (b := client.messages.batches.retrieve(batch_id)).processing_status != "ended":
        time.sleep(POLL_S)
    rows, cost, failed = [], 0.0, 0
    for r in client.messages.batches.results(batch_id):
        grader, label, *rubric = r.custom_id.split("__")
        row = {"grader": grader, "label": label, "rubric": rubric[0] if rubric else "v1",
               "status": r.result.type}
        if r.result.type == "succeeded":
            msg = r.result.message
            pin, pout = PRICES[msg.model] if msg.model in PRICES else PRICES[GRADERS[grader]["model"]]
            cost += (msg.usage.input_tokens * pin + msg.usage.output_tokens * pout) \
                * BATCH_DISCOUNT / 1e6
            if msg.stop_reason == "refusal":
                row["status"] = "refusal"
            else:
                text = next((b.text for b in msg.content if b.type == "text"), "")
                try:
                    row.update(json.loads(text))
                except ValueError:
                    row["status"] = "unparseable"
        failed += row["status"] != "succeeded"
        rows.append(row)
    Path(out_path).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n",
                              encoding="utf-8")
    print(f"batch {batch_id} ended: {len(rows)} results, {failed} not usable, "
          f"cost ${cost:.4f} -> {out_path}")


QUESTIONS = ROOT / "eval" / "recommend_questions.jsonl"
FULL_SYSTEM = ("You grade candidate recommendations for a private network of business leaders. You "
               "receive ONE member's full profile and a list of questions other members asked. "
               "Grade this member for EVERY question, independently. " + RUBRIC
               + "\n\nJudge from the profile alone. For each question give the grade; give a short "
               "reason only for grades 1 and 2 (an empty string for 0).")


def _full_schema(qids: list[str]) -> dict:
    item = {"type": "object",
            "properties": {"qid": {"type": "string", "enum": qids},
                           "grade": {"type": "integer", "enum": [0, 1, 2]},
                           "reason": {"type": "string"}},
            "required": ["qid", "grade", "reason"], "additionalProperties": False}
    return {"type": "object", "properties": {"grades": {"type": "array", "items": item}},
            "required": ["grades"], "additionalProperties": False}


def full(grader: str, out_prefix: str) -> None:
    """Submit: every member x every question, one request per member (the dossier is read once,
    not once per question). custom_ids are row numbers, not person_ids: ids are name slugs."""
    sys.path.insert(0, str(ROOT / "src"))
    import store

    qs = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l]
    qids = [q["id"] for q in qs]
    qtext = "Questions:\n" + "\n".join(f"[{q['id']}] {q['question']}" for q in qs)
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT person_id FROM persons ORDER BY person_id")
        pids = [r[0] for r in cur.fetchall()]
    g = GRADERS[grader]
    requests, index = [], {}
    for i, pid in enumerate(pids):
        dossier = store.fetch_person_chunks(conn, pid, scheme="whole_doc")[0]["text"]
        params = {"model": g["model"], "max_tokens": 4000, "system": FULL_SYSTEM,
                  "messages": [{"role": "user", "content": f"{qtext}\n\nProfile:\n{dossier}"}],
                  "output_config": {"format": {"type": "json_schema", "schema": _full_schema(qids)}}}
        if "thinking" in g:
            params["thinking"] = g["thinking"]
        cid = f"full__{grader}__{i:03d}__{RUBRIC_VERSION}"
        requests.append({"custom_id": cid, "params": params})
        index[cid] = pid
    batch = anthropic.Anthropic().messages.batches.create(requests=requests)
    Path(f"{out_prefix}_index.json").write_text(
        json.dumps({"batch": batch.id, "qids": qids, "index": index}, indent=1), encoding="utf-8")
    print(f"submitted batch {batch.id} (rubric {RUBRIC_VERSION}, {grader}): {len(requests)} "
          f"members x {len(qids)} questions")


def collect_full(index_path: str, out_path: str) -> None:
    """Poll, then write one row per (member, question). Reports completeness, never grades."""
    meta = json.loads(Path(index_path).read_text(encoding="utf-8"))
    client = anthropic.Anthropic()
    while client.messages.batches.retrieve(meta["batch"]).processing_status != "ended":
        time.sleep(POLL_S)
    rows, cost, incomplete = [], 0.0, []
    for r in client.messages.batches.results(meta["batch"]):
        pid = meta["index"][r.custom_id]
        if r.result.type != "succeeded":
            incomplete.append(pid)
            continue
        msg = r.result.message
        pin, pout = PRICES[GRADERS[r.custom_id.split("__")[1]]["model"]]
        cost += (msg.usage.input_tokens * pin + msg.usage.output_tokens * pout) * BATCH_DISCOUNT / 1e6
        try:
            got = {x["qid"]: x for x in json.loads(
                next(b.text for b in msg.content if b.type == "text"))["grades"]}
        except (ValueError, StopIteration, KeyError):
            got = {}
        if set(got) != set(meta["qids"]):
            incomplete.append(pid)
        rows += [{"person_id": pid, "qid": q, "grade": x["grade"], "reason": x["reason"]}
                 for q, x in got.items()]
    Path(out_path).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n",
                              encoding="utf-8")
    print(f"batch {meta['batch']} ended: {len(rows)} grades, {len(incomplete)} members incomplete "
          f"(re-run those), cost ${cost:.4f} -> {out_path}")


def agree(labels_path: str, results_path: str) -> None:
    """Each grader vs the reviewer's blind labels. Run only AFTER the reviewer has labelled.
    Reported plainly: exact matches, and the two mistakes that matter differently for a
    recommender -- missing someone the reviewer rated useful, or promoting someone they rated 0."""
    human = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in Path(results_path).read_text(encoding="utf-8").splitlines() if l]
    print(f"{'grader':11} {'exact':>9} {'useful-or-not':>14} {'missed useful':>14} "
          f"{'promoted a 0':>13} {'kappa':>6}")
    for grader in GRADERS:
        got = {r["label"]: r["grade"] for r in rows
               if r["grader"] == grader and r["status"] == "succeeded"}
        both = [(human[l], got[l]) for l in human if l in got]
        n = len(both)
        exact = sum(a == b for a, b in both)
        binary = sum((a > 0) == (b > 0) for a, b in both)
        missed = sum(a > 0 and b == 0 for a, b in both)
        promoted = sum(a == 0 and b > 0 for a, b in both)
        # Cohen's kappa (3 grades): agreement beyond what two raters with these grade mixes
        # would reach by chance -- 1 = perfect, 0 = chance level.
        pe = sum((sum(a == k for a, _ in both) / n) * (sum(b == k for _, b in both) / n)
                 for k in (0, 1, 2))
        kappa = (exact / n - pe) / (1 - pe) if pe < 1 else float("nan")
        print(f"{grader:11} {exact:>4}/{n:<4} {binary:>9}/{n:<4} {missed:>14} {promoted:>13} "
              f"{kappa:>6.2f}")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"bakeoff": bakeoff, "collect": collect, "agree": agree, "full": full,
     "collect_full": collect_full}[cmd](*args)
