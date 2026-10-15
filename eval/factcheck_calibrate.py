"""Calibrate the draft fact-check (D-057) before it is trusted -- the D-022 discipline applied to
the new guard: test it in the direction that matters (does it catch an untrue claim?) on drafts
whose truth is known.

Items are real drafts the agent wrote in the intro runs (read from those runs' gitignored
results), checked against the same whole dossiers the product uses:
  positive   a draft whose unsupported claims a human read found; each named KEY must be flagged
  negative   a draft the human read found clean; any flag is a false alarm (a needless rewrite)
  planted    a clean draft with one realistic untrue detail swapped or added; its KEY must be flagged
  gray       a borderline draft; reported only, never decides
A key counts as flagged in a run when a returned quote contains it.

PRE-REGISTERED RULE (written before any run): PASS iff every positive and planted key is flagged
in at least 2 of 3 runs and none in 0 of 3. False alarms are reported (rate over the negatives,
majority of runs); the flip rate is the share of items whose outcome differs across runs.

Usage:  python eval/factcheck_calibrate.py --dry   (items + token estimate; no calls)
        python eval/factcheck_calibrate.py         (paid, second provider; 3 runs per item)
Writes analysis/factcheck_calibration_<ts>.json (gitignored: it holds draft text) and prints a
name-free summary.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import factcheck  # noqa: E402
import store  # noqa: E402
import tools  # noqa: E402

RUNS = 3
RUN_A, RUN_B = "20261015T125955Z-ecef064-conv", "20261015T132933Z-c7b89bc-conv"
# (kind, run, conversation, turn, keys, plant) -- plant = (old, new) swapped into the draft, or
# ("+after:<anchor>", sentence) inserted after the paragraph containing <anchor>.
ITEMS = [
    ("positive", RUN_A, "cv-011", 2, ["following"], None),
    ("positive", RUN_A, "cv-010", 2, ["watching"], None),
    ("positive", RUN_B, "cv-011", 3, ["internationally", "working on taking"], None),
    ("negative", RUN_B, "cv-012", 3, [], None),
    ("negative", RUN_B, "cv-012", 4, [], None),
    ("negative", RUN_B, "cv-013", 3, [], None),
    ("negative", RUN_A, "cv-013", 3, [], None),
    ("gray", RUN_A, "cv-009", 2, [], None),       # "Series A round": member said "our fundraise"
    ("gray", RUN_A, "cv-012", 3, [], None),       # "valuable synergies" (filler, not a fact)
    ("gray", RUN_B, "cv-010", 2, [], None),       # "I came across your work"
    ("planted", RUN_B, "cv-013", 3, ["14 locations"], ("11 locations", "14 locations")),
    ("planted", RUN_B, "cv-012", 4, ["UiPath"], ("Automation Anywhere", "UiPath")),
    ("planted", RUN_B, "cv-012", 4, ["Buenos Aires"], ("Bogota", "Buenos Aires")),
    ("planted", RUN_A, "cv-013", 3, ["met briefly"],
     ("+after:Paladin", "We met briefly at a Phoenix healthcare summit last spring.")),
    ("planted", RUN_B, "cv-012", 3, ["Siemens"], ("at Emerson and Nortek",
                                                  "at Emerson, Nortek and Siemens")),
    ("planted", RUN_B, "cv-013", 3, ["seed round"],
     ("Would you have", "I'm also raising a seed round and would value your take on investors. "
                        "Would you have")),
]


def plant(draft: str, change) -> str:
    old, new = change
    if old.startswith("+after:"):
        paras = draft.split("\n\n")
        i = next(k for k, para in enumerate(paras) if old[7:] in para)
        return "\n\n".join(paras[:i + 1] + [new] + paras[i + 1:])
    assert old in draft, f"plant anchor {old!r} not in the draft"
    return draft.replace(old, new, 1)


def build_items():
    conn = store.connect()
    resolver = tools.build_resolver(conn)
    asker = sorted(resolver.name_of)[0]       # the runs' fixed asker (conversation_run.setup)

    def doc(pid):
        return "\n".join(c["text"] for c in store.fetch_person_chunks(conn, pid, scheme="whole_doc"))

    rows = {}
    for run in (RUN_A, RUN_B):
        for line in (ROOT / "runs" / run / "results.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            rows[(run, r["cid"], r["turn"])] = r
    items = []
    for kind, run, cid, turn, keys, change in ITEMS:
        r = rows[(run, cid, turn)]
        ref = next(c for c in r["tool_calls"] if c["name"] == "send_intro")["args"]["recipient"]
        rid = ref if ref in resolver.name_of else resolver.resolve(ref).person_ids[0]
        words = [rows[(run, cid, k)]["user"] for k in range(1, turn + 1)
                 if (run, cid, k) in rows and rows[(run, cid, k)].get("user")]
        draft = r["draft"].strip()
        items.append({"kind": kind, "item": f"{run[:15]}/{cid}/t{turn}", "keys": keys,
                      "draft": plant(draft, change) if change else draft,
                      "recipient_doc": doc(rid), "sender_doc": doc(asker), "words": words})
    return items


def flagged(key: str, unsupported: list[dict]) -> bool:
    return any(key.lower() in u["quote"].lower() for u in unsupported)


def main(dry: bool) -> None:
    items = build_items()
    chars = sum(len(factcheck.prompt(i["draft"], i["recipient_doc"], i["sender_doc"], i["words"]))
                + len(factcheck.SYSTEM) for i in items)
    est_in = chars / 3.5 * RUNS                   # rough: no free token counter for this provider
    est = est_in * factcheck.PRICE_IN / 1e6 + len(items) * RUNS * 500 * factcheck.PRICE_OUT / 1e6
    mix = ", ".join(f"{k} {sum(i['kind'] == k for i in items)}"
                    for k in ("positive", "negative", "planted", "gray"))
    print(f"{len(items)} items ({mix}) x {RUNS} runs; "
          f"~{est_in / (len(items) * RUNS):,.0f} input tokens per call; est. ${est:.2f}")
    if dry:
        return
    results = []
    for i in items:
        runs = [factcheck.check(i["draft"], i["recipient_doc"], i["sender_doc"], i["words"],
                                use_cache=False) for _ in range(RUNS)]
        results.append({**{k: i[k] for k in ("kind", "item", "keys", "draft")},
                        "runs": [{"ok": f.ok, "error": f.error, "unsupported": f.unsupported,
                                  "seconds": round(f.seconds, 2), "cost_usd": f.cost_usd}
                                 for f in runs]})
        print(f"  {i['kind']:<8} {i['item']}: " + " | ".join(
            "error" if f.ok is None else f"{len(f.unsupported)} flagged" for f in runs))
    keyed = [(r["item"], k, sum(flagged(k, x["unsupported"]) for x in r["runs"]))
             for r in results if r["kind"] in ("positive", "planted") for k in r["keys"]]
    missed = [(item, k) for item, k, n in keyed if n == 0]
    weak = [(item, k, n) for item, k, n in keyed if 0 < n < 2]
    false_alarms = [r["item"] for r in results if r["kind"] == "negative"
                    and sum(bool(x["unsupported"]) for x in r["runs"]) >= 2]
    flips = sum(len({bool(x["unsupported"]) for x in r["runs"]}) > 1 for r in results)
    secs = sorted(x["seconds"] for r in results for x in r["runs"] if x["ok"] is not None)
    spend = sum(x["cost_usd"] for r in results for x in r["runs"])
    verdict = "PASS" if not missed and not weak else "FAIL"
    summary = {"verdict": verdict, "model": factcheck.MODEL, "version": factcheck.VERSION,
               "reasoning_effort": factcheck.REASONING_EFFORT, "runs_per_item": RUNS,
               "keys": len(keyed), "keys_caught_3of3": sum(n == RUNS for _, _, n in keyed),
               "keys_missed": len(missed), "keys_weak": len(weak),
               "false_alarms": f"{len(false_alarms)}/{sum(r['kind'] == 'negative' for r in results)}",
               "flip_rate": round(flips / len(results), 2),
               "errors": sum(x["ok"] is None for r in results for x in r["runs"]),
               "seconds_p50": secs[len(secs) // 2] if secs else None,
               "seconds_max": secs[-1] if secs else None, "cost_usd": round(spend, 4)}
    out = ROOT / "analysis" / f"factcheck_calibration_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    out.write_text(json.dumps({"summary": summary, "missed": missed, "weak": weak,
                               "false_alarms": false_alarms, "results": results},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"details (holds draft text, gitignored): {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main("--dry" in sys.argv)
