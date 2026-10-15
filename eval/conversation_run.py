"""List tracking across turns (D-054): short multi-turn conversations through the real chat agent
(src/agent.py), checking mechanically -- no judge -- that when the member points back at people
("the second one", "Maybe Dana?", "anyone else?", "among those..."), the agent acts on the right
people.

The conversations (eval/conversations.jsonl, private like every gold file) are name-free
templates. A turn marked "list": "L1" records L1 = the members the REPLY named, in the order the
member read them (numbered or bulleted lines first, then first mention) -- "the third one" refers
to what was on screen, not to the tool's order. Later user text may say {L1.3.first}, filled at
run time from that list, so the file never carries a real name.

Checks (on the turn that carries "expect"):
  lookup      the person the agent fetched with get_person_evidence this turn is the referred
              person, and only them; if it fetched nobody, the first member its reply names
  lookup_set  same, for a set ("compare the first two")
  exclude     "anyone else?": the people this turn's search found AND the reply named share nobody
              with the earlier list(s), and there is at least one
  within      "among those": every find_members call this turn narrows with within_person_ids,
              and the reply names nobody outside the list(s)
  intro       a message for the member (D-055): the turn ends waiting for approval with NOTHING
              sent yet, the draft is to the referred person and names nobody else; then the
              runner decides as the member would ("decide": approve / edit / reject / none) and
              the outbox must hold exactly what was approved (the edited text for an edit,
              nothing for a reject). "shorter": the redraft after "make it shorter" is shorter.
              A message typed while a draft waits declines it with those words, as in the app.
  asks        the member asked for a message without saying why: no draft, a question back.
Every draft is fact-checked by a second provider before the card (D-057); each check is recorded
(time, cost, what it flagged), with the rewrites it caused and any claim still unconfirmed at
the card. The scripted self-check uses a stub checker -- it never calls the second provider.
A reference beyond a list's length (the agent showed 3, the member asks for the fourth) scores
setup_failed -- the conversation could not be set up, so it is neither a pass nor a fail.

Turns run through agent.stream_turn, the chat app's own path, so every turn also records the time
to first words on screen (D-047). The eval owns the clock: +30 s per turn, plus any "pause_s"
(a pause past the cache lifetime makes D-046 drop old search results, as in the live chat).

Usage:  python eval/conversation_run.py               (paid; all conversations)
        EVAL_LIMIT=1 python eval/conversation_run.py  (first N conversations)
        EVAL_ONLY=cv-002,cv-003 python eval/conversation_run.py   (named conversations)
        python eval/conversation_run.py --selfcheck   (offline: scripted model, real DB + tools)
Writes runs/<ts>-<sha>-conv/ (config + summary committable and name-free -- people appear as list
positions like "L1.3"; results.jsonl, which holds the replies, gitignored).
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
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
# LangGraph warns when it serialises the asker context, and the warning prints the start of a
# real person_id to the console. Local only, but nothing to learn from it here.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

import agent  # noqa: E402
import cost  # noqa: E402
import store  # noqa: E402
import tools  # noqa: E402
from chat_run import usage_of  # noqa: E402
import factcheck  # noqa: E402
from outbox import MemoryOutbox  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

CONVERSATIONS = ROOT / "eval" / "conversations.jsonl"
LIMIT = int(os.environ.get("EVAL_LIMIT", "0")) or None
ONLY = [c for c in os.environ.get("EVAL_ONLY", "").split(",") if c]   # conversation ids
# TUNABLE($0.75 ceiling for one full run of the 8 conversations, ~24 turns; a run that would cross
#   it stops and reports. Symptom wrong: a normal run aborts near the end -> re-quote.)
SPEND_CEILING_USD = float(os.environ.get("EVAL_SPEND_CEILING", "0.75"))
# TUNABLE(30 s between turns: well inside the 300 s cache lifetime, so nothing is dropped unless a
#   turn asks for a pause. Symptom wrong: never -- it only has to stay under COLD_CACHE_S.)
TURN_GAP_S = 30
START_CLOCK = 1_800_000_000.0
FACTCHECKS: list[dict] = []   # every fact-check this process made, in order (see recording_checker)
EDIT_APPEND = "Happy to work around your schedule."   # what the "member" adds on an edit turn
TEMPLATE = re.compile(r"\{(L\d+)\.(\d+|last)\.(first|name)\}")
ITEM_LINE = re.compile(r"^\s*(\d+[.)]|[-*•])\s")


# --- reading what the member saw -----------------------------------------------------------

def named_on_screen(reply: str, name_of: dict, asker: str) -> list[str]:
    """Members the reply names, in the order a reader takes them as a list: people on numbered
    or bulleted lines first (by line), then anyone named only in prose (by first mention)."""
    lines = reply.splitlines()
    item_rows = [i for i, line in enumerate(lines) if ITEM_LINE.match(line)]
    keys: dict[str, tuple] = {}
    for pid, name in name_of.items():
        if pid == asker:
            continue
        pat = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE)
        m = pat.search(reply)
        if not m:
            continue
        row = next((i for i in item_rows if pat.search(lines[i])), None)
        keys[pid] = (0, row, m.start()) if row is not None else (1, 0, m.start())
    return sorted(keys, key=keys.get)


def resolve_ref(ref: str, lists: dict[str, list[str]]) -> str | None:
    """'L1.3' / 'L1.last' -> person_id, or None when the list is missing or too short."""
    label, pos = ref.split(".")
    people = lists.get(label) or []
    if pos == "last":
        return people[-1] if people else None
    i = int(pos) - 1
    return people[i] if 0 <= i < len(people) else None


def fill(text: str, lists: dict, name_of: dict) -> str | None:
    def sub(m: re.Match) -> str:
        pid = resolve_ref(f"{m[1]}.{m[2]}", lists)
        if pid is None:
            raise LookupError(m[0])
        name = name_of[pid]
        return name.split()[0] if m[3] == "first" else name
    try:
        return TEMPLATE.sub(sub, text)
    except LookupError:
        return None


def label_of(pid: str, lists: dict, prefer: str | None = None) -> str:
    """Name-free label for the committed summary: the person's position in the preferred list
    (the one the check refers to), else their first list position."""
    order = sorted(lists, key=lambda label: label != prefer)
    for label in order:
        people = lists[label]
        if pid in people:
            return f"{label}.{people.index(pid) + 1}"
    return "outside"


def recording_checker(check):
    """Wrap a fact-checker so every call is logged (the turn rows read their slice)."""
    def wrapped(draft, recipient_doc, sender_doc, words):
        fc = check(draft, recipient_doc, sender_doc, words)
        # The filter, the card text and the tool each ask about the same draft and get the one
        # cached result: log it once, or calls and cost read ~3x (they did on 2026-10-15).
        if getattr(fc, "_recorded", False):
            return fc
        fc._recorded = True
        FACTCHECKS.append({"ok": fc.ok, "n_unsupported": len(fc.unsupported), "error": fc.error,
                           "seconds": round(fc.seconds, 2), "cost_usd": fc.cost_usd,
                           "unsupported": fc.unsupported})
        return fc
    return wrapped


def turn_evidence(new_msgs: list, asker: str) -> dict:
    """What the agent did this turn: who get_person_evidence fetched, each find_members call
    (its arguments and who it returned), and the final reply."""
    calls = {c["id"]: c for m in new_msgs if isinstance(m, AIMessage) for c in (m.tool_calls or [])}
    fetched, searches, sent_back = [], [], 0
    for m in new_msgs:
        if not isinstance(m, ToolMessage):
            continue
        if m.name == "send_intro" and agent.UNSUPPORTED in agent._text(m):
            sent_back += 1
        try:
            out = json.loads(agent._text(m))
        except json.JSONDecodeError:
            continue
        args = calls.get(m.tool_call_id, {}).get("args", {})
        if m.name == "get_person_evidence" and out.get("ok"):
            fetched += [p["person_id"] for p in out["persons"]
                        if p["person_id"] != asker and p["person_id"] not in fetched]
        elif m.name == "find_members":
            searches.append({"args": args, "ok": bool(out.get("ok")),
                             "found": [p["person_id"] for p in out.get("members", [])]})
    final = new_msgs[-1] if new_msgs else None
    reply = agent._text(final) if isinstance(final, AIMessage) and not final.tool_calls else ""
    return {"fetched": fetched, "searches": searches, "reply": reply, "sent_back": sent_back,
            "tool_calls": [{"name": c["name"], "args": c["args"]} for c in calls.values()]}


def check_intro(expect: dict, lists: dict, ev: dict, intro: dict) -> dict:
    want = resolve_ref(expect["target"], lists)
    if want is None:
        return {"result": "setup_failed", "detail": f"list too short for {expect['target']}"}
    waiting = intro["approval"]
    if waiting is None:
        tried = any(c["name"] == "send_intro" for c in ev["tool_calls"])
        return {"result": "fail", "detail": "no approval requested ("
                + ("its send was refused" if tried else "no send_intro call") + ")"}
    how, sent = expect.get("decide", "approve"), intro["sent"]
    draft = waiting["message"].strip()
    words = len(draft.split())
    problems = []
    if intro["sent_at_approval"]:
        problems.append(f"{intro['sent_at_approval']} sent BEFORE approval")
    if waiting["recipient_id"] != want:
        problems.append(f"draft to {label_of(waiting['recipient_id'], lists)}")
    if intro["others_named"]:
        problems.append(f"draft names {intro['others_named']} other member(s)")
    if how in ("approve", "edit"):
        text = (intro["edit_text"] or "").strip() if how == "edit" else draft
        if [(s.to_person_id, s.message, s.edited) for s in sent] != [(want, text, how == "edit")]:
            problems.append(f"outbox after {how}: {len(sent)} sent, not the approved text")
    elif sent:
        problems.append(f"{len(sent)} sent after {how}")
    if expect.get("shorter") and not (intro["previous_draft"]
                                      and words < len(intro["previous_draft"].split())):
        problems.append("not shorter than the previous draft")
    detail = (f"draft to {label_of(waiting['recipient_id'], lists, expect['target'].split('.')[0])}"
              f" ({words} words{', over 150' if words > 150 else ''}), {how} -> {len(sent)} sent"
              + (f", {ev['sent_back']} rewrite(s) after the fact-check" if ev["sent_back"] else "")
              + (", reached the card UNCONFIRMED" if "Still not confirmed" in
                 waiting.get("description", "") else "")
              + (", with a poor-fit heads-up" if waiting.get("note") else ""))
    return {"result": "fail" if problems else "pass",
            "detail": detail + ("; " + "; ".join(problems) if problems else "")}


def check(expect: dict, lists: dict, ev: dict, named: list[str], intro: dict | None = None) -> dict:
    kind = expect["kind"]
    if kind == "intro":
        return check_intro(expect, lists, ev, intro or {})
    if kind == "asks":
        drafted = (intro or {}).get("approval") is not None or any(
            c["name"] == "send_intro" for c in ev["tool_calls"])
        ok = not drafted and "?" in ev["reply"]
        return {"result": "pass" if ok else "fail",
                "detail": "drafted a message" if drafted else
                          ("asked" if ok else "no draft, but no question either")}
    asked = not ev["tool_calls"] and "?" in ev["reply"]
    if kind in ("lookup", "lookup_set"):
        refs = [expect["target"]] if kind == "lookup" else expect["targets"]
        want = [resolve_ref(r, lists) for r in refs]
        if None in want:
            return {"result": "setup_failed", "detail": f"list too short for {refs}"}
        if ev["fetched"]:
            got, basis = ev["fetched"], "fetched"
            ok = set(got) == set(want)
        else:
            got, basis = named[:len(want)], "reply"
            ok = set(got) == set(want) if kind == "lookup_set" else bool(got) and got[0] == want[0]
        prefer = refs[0].split(".")[0]
        detail = f"{basis} {[label_of(p, lists, prefer) for p in got]} wanted {refs}"
        return {"result": "pass" if ok else "fail",
                "detail": detail + (" (asked a question, no lookup)" if asked else "")}
    pool = {p for label in expect["lists"] for p in lists.get(label, [])}
    if kind == "exclude":
        found = {p for s in ev["searches"] for p in s["found"]}
        new = [p for p in named if p in found]
        repeats = [p for p in new if p in pool]
        excl = [s["args"].get("exclude_person_ids") for s in ev["searches"]]
        ok = bool(ev["searches"]) and bool(new) and not repeats
        return {"result": "pass" if ok else "fail",
                "detail": f"{len(new)} new named, repeats {[label_of(p, lists) for p in repeats]}, "
                          f"exclusions passed {[len(e or []) for e in excl]}"
                          + ("" if ev["searches"] else ", no search")}
    if kind == "within":
        unnarrowed = [s for s in ev["searches"] if not s["args"].get("within_person_ids")]
        outside = [p for p in named if p not in pool]
        ok = not unnarrowed and not outside
        return {"result": "pass" if ok else "fail",
                "detail": f"{len(unnarrowed)} network-wide search(es), {len(outside)} named "
                          f"outside {expect['lists']}"}
    raise ValueError(kind)


# --- running ---------------------------------------------------------------------------------

def _consume(events, t0: float):
    """Drain one stream of screen events -> (reply first words, pre-tool text time, approval).
    A drafted message is not streamed (D-057): on a draft turn the time that matters is when the
    checked draft reaches the card, recorded on the approval as card_s."""
    first = flash = approval = None
    shown = ""
    for kind, part in events:
        if kind == "approval":
            approval = {**part, "card_s": round(time.perf_counter() - t0, 2)}
        if kind != "partial":
            continue
        # Partials only grow within one model call. Text that is not a continuation
        # came from a NEW call: what was on screen before it was text the model wrote
        # ahead of a tool call (seen 2026-10-15) -- first words = the reply's own.
        now = time.perf_counter() - t0
        if first is None or not part.startswith(shown):
            flash = flash if first is None else (flash or first)
            first = now
        shown = part
    return first, flash, approval


def run_conversation(chat, resolver, asker, conv: dict, tick: dict,
                     on_turn: Callable | None = None, outbox=None) -> list[dict]:
    thread = f"conv-{conv['id']}-{tick['t']:.0f}"
    cfg = {"configurable": {"thread_id": thread}}
    lists: dict[str, list[str]] = {}
    rows, stopped, previous_draft = [], None, None
    for n, spec in enumerate(conv["turns"], 1):
        row = {"cid": conv["id"], "turn": n, "pattern": conv["pattern"], "pause_s":
               spec.get("pause_s", 0), "expect": spec.get("expect"), "check": None}
        text = None if stopped else fill(spec["user"], lists, resolver.name_of)
        if text is None:
            stopped = stopped or f"turn {n} needs a list position that was never shown"
            row.update(error=stopped, check={"result": "setup_failed", "detail": stopped}
                       if spec.get("expect") else None)
            rows.append(row)
            continue
        tick["t"] += TURN_GAP_S + spec.get("pause_s", 0)
        if on_turn:
            on_turn(spec, lists, conv)
        before = len(chat.get_state(cfg).values.get("messages", []))
        sent_before = len(outbox.sent) if outbox is not None else 0
        checks_before = len(FACTCHECKS)
        exp = spec.get("expect") or {}
        t0, first, flash, error = time.perf_counter(), None, None, None
        approval, decided, edit_text, sent_at_approval = None, None, None, None
        try:
            waiting = agent.pending_approval(chat, resolver, thread, asker)
            events = (agent.resume_turn(chat, resolver, thread, asker,
                                        agent.feedback_decision(text)) if waiting
                      else agent.stream_turn(chat, resolver, thread, asker, text))
            first, flash, approval = _consume(events, t0)
            if approval is not None:
                sent_at_approval = len(outbox.sent) - sent_before if outbox is not None else 0
                how = exp.get("decide", "approve") if exp.get("kind") == "intro" else "none"
                if how != "none":
                    if how == "edit":
                        edit_text = f"{approval['message'].rstrip()} {EDIT_APPEND}"
                    decided = agent.member_decision(approval, how != "reject", edit_text)
                    _consume(agent.resume_turn(chat, resolver, thread, asker, decided),
                             time.perf_counter())
        except Exception as exc:  # a guard or tool failure is a result, not a crash
            error = f"{type(exc).__name__}: {exc}"
        took = time.perf_counter() - t0
        state = chat.get_state(cfg).values
        new = state.get("messages", [])[before:]
        ev = turn_evidence(new, asker.person_id)
        named = named_on_screen(ev["reply"], resolver.name_of, asker.person_id)
        draft = (approval or {}).get("message", "")
        intro = {"approval": approval, "sent_at_approval": sent_at_approval,
                 "sent": outbox.sent[sent_before:] if outbox is not None else [],
                 "edit_text": edit_text, "previous_draft": previous_draft,
                 "others_named": len(set(resolver.full_names(draft))
                                     - {(approval or {}).get("recipient_id"), asker.person_id})}
        if spec.get("expect"):
            row["check"] = check(spec["expect"], lists, ev, named, intro)
        previous_draft = draft or previous_draft
        if spec.get("list"):
            earlier = {p for people in lists.values() for p in people}
            lists[spec["list"]] = [p for p in named
                                   if not (spec.get("new_only") and p in earlier)]
        ai = [m for m in new if isinstance(m, AIMessage)]
        calls = [u for u in (usage_of(m) for m in ai) if u]
        usage = {k: sum(c[k] for c in calls) for k in calls[0]} if calls else {}
        usd = cost.cost_usd({"model": agent.AGENT_MODEL, **usage}) if usage else 0.0
        checks = FACTCHECKS[checks_before:]
        usd = (usd or 0.0) + sum(c["cost_usd"] for c in checks)
        row.update(user=text, reply=ev["reply"], named=[label_of(p, lists) for p in named],
                   named_ids=named, fetched=ev["fetched"], searches=ev["searches"],
                   tool_calls=ev["tool_calls"], rounds=len(ai), error=error,
                   first_words_s=round(first, 2) if first is not None else None,
                   pre_tool_text_s=round(flash, 2) if flash is not None else None,
                   turn_s=round(took, 2), usage=usage, cost_usd=usd,
                   dropped_through=state.get("search_results_dropped_through", 0),
                   lists={k: len(v) for k, v in lists.items()},
                   draft=draft or None, draft_words=len(draft.split()) if draft else None,
                   decision=(decided or {}).get("type"), sent=len(intro["sent"]),
                   factchecks=checks, sent_back=ev["sent_back"],
                   note=(approval or {}).get("note") or None,
                   card_s=(approval or {}).get("card_s"),
                   unconfirmed_at_card="Still not confirmed" in (approval or {}).get(
                       "description", ""))
        rows.append(row)
        c = row["check"]
        print(f"  [{conv['id']} t{n}] tools={[t['name'] for t in ev['tool_calls']]} "
              f"first={row['first_words_s']}s turn={took:.1f}s ${usd or 0:.4f}"
              + (f" | {spec['expect']['kind']}: {c['result']} -- {c['detail']}" if c else "")
              + (f" !! {error}" if error else ""))
    return rows


def pct(xs: list[float], q: float):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))] if xs else None


def summarize(rows: list[dict]) -> dict:
    checks = {f"{r['cid']}.t{r['turn']}": {"kind": r["expect"]["kind"], "pattern": r["pattern"],
                                            **r["check"]} for r in rows if r["check"]}
    tally = {k: sum(1 for c in checks.values() if c["result"] == k)
             for k in ("pass", "fail", "setup_failed")}
    by_kind: dict[str, dict] = {}
    for c in checks.values():
        b = by_kind.setdefault(c["kind"], {"pass": 0, "fail": 0, "setup_failed": 0})
        b[c["result"]] += 1
    ran = [r for r in rows if "turn_s" in r]
    tool_turns = [r for r in ran if r["tool_calls"]]
    return {
        "n_conversations": len({r["cid"] for r in rows}), "n_turns": len(ran),
        "checks": tally, "by_kind": by_kind,
        "errors": sum(1 for r in rows if r.get("error")),
        "first_words_s": {"tool_turns_p50": pct([r["first_words_s"] for r in tool_turns], .5),
                          "tool_turns_p95": pct([r["first_words_s"] for r in tool_turns], .95),
                          "no_tool_turns_p50": pct([r["first_words_s"] for r in ran
                                                    if not r["tool_calls"]], .5),
                          "n_tool_turns": len(tool_turns),
                          "turns_showing_pre_tool_text": sum(1 for r in ran
                                                             if r.get("pre_tool_text_s"))},
        "turn_s_p50": pct([r["turn_s"] for r in ran], .5),
        "draft_card_s": {"p50": pct([r.get("card_s") for r in ran], .5),
                         "p95": pct([r.get("card_s") for r in ran], .95),
                         "n": sum(1 for r in ran if r.get("card_s") is not None)},
        "cost_usd": round(sum(r["cost_usd"] or 0 for r in ran), 4),
        "factcheck": {
            "model": factcheck.MODEL, "version": factcheck.VERSION,
            "calls": sum(len(r.get("factchecks", [])) for r in ran),
            "flagged": sum(1 for r in ran for c in r.get("factchecks", []) if c["ok"] is False),
            "errors": sum(1 for r in ran for c in r.get("factchecks", []) if c["ok"] is None),
            "rewrites": sum(r.get("sent_back", 0) for r in ran),
            "unconfirmed_at_card": sum(1 for r in ran if r.get("unconfirmed_at_card")),
            "heads_ups": sum(1 for r in ran if r.get("note")),
            "seconds_p50": pct([c["seconds"] for r in ran for c in r.get("factchecks", [])], .5),
            "cost_usd": round(sum(c["cost_usd"] for r in ran for c in r.get("factchecks", [])), 4)},
        "per_check": checks}


def git_state() -> tuple[str, bool]:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         cwd=ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                                cwd=ROOT).stdout.strip())
    return sha, dirty


def write_run(rows: list[dict], convs: list[dict], asker) -> str:
    sha, dirty = git_state()
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{sha}-conv"
    out = ROOT / "runs" / run_id
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({
        "run_id": run_id, "runner": "eval/conversation_run.py", "git": {"sha": sha, "dirty": dirty},
        "conversations": {"file": CONVERSATIONS.name, "ids": [c["id"] for c in convs],
                          "sha256": hashlib.sha256(CONVERSATIONS.read_bytes()).hexdigest()},
        "agent": {"model": agent.AGENT_MODEL, "web": "off", "model_picks": False,
                  "prompt_sha256": hashlib.sha256(agent.base_prompt(False, False).encode())
                  .hexdigest(), "hybrid_search": tools.HYBRID_SEARCH,
                  "keep_turns": agent.KEEP_TURNS, "cold_cache_s": agent.COLD_CACHE_S},
        "clock": {"start": START_CLOCK, "turn_gap_s": TURN_GAP_S},
        "asker_sha256_12": hashlib.sha256(asker.person_id.encode()).hexdigest()[:12],
        "limit": LIMIT, "only": ONLY}, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summarize(rows), indent=1), encoding="utf-8")
    (out / "results.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
                                       + "\n", encoding="utf-8")
    return run_id


def load_conversations() -> list[dict]:
    convs = [json.loads(line) for line in CONVERSATIONS.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    convs = [c for c in convs if c["id"] in ONLY] if ONLY else convs
    return convs[:LIMIT] if LIMIT else convs


def setup():
    from embedder import NomicLocal

    conn = store.connect()
    resolver = tools.build_resolver(conn)
    emb = NomicLocal()
    emb.warm()
    asker = agent.load_asker(conn, resolver, sorted(resolver.name_of)[0])
    return conn, resolver, emb, asker


def main() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    conn, resolver, emb, asker = setup()
    tick = {"t": START_CLOCK}
    agent._clock = lambda: tick["t"]
    box = MemoryOutbox()     # an eval never writes to the real outbox table
    chat = agent.build_agent(emb, conn, resolver, InMemorySaver(), web_enabled=False,
                             model_picks=False, outbox=box,
                             fact_checker=recording_checker(factcheck.check))
    convs, rows, spent = load_conversations(), [], 0.0
    for conv in convs:
        conv_rows = run_conversation(chat, resolver, asker, conv, tick, outbox=box)
        rows += conv_rows
        spent += sum(r.get("cost_usd") or 0 for r in conv_rows)
        if spent > SPEND_CEILING_USD:
            print(f"!! spend ceiling ${SPEND_CEILING_USD} crossed at ${spent:.3f}; stopping.")
            break
    run_id = write_run(rows, convs, asker)
    s = json.loads((ROOT / "runs" / run_id / "summary.json").read_text())
    print(f"{run_id}: checks {s['checks']} | by kind {s['by_kind']} | first words on tool turns "
          f"p50 {s['first_words_s']['tool_turns_p50']}s p95 {s['first_words_s']['tool_turns_p95']}s"
          f" | ${s['cost_usd']}")


# --- offline: scripted agent, self-check, cost quote ----------------------------------------

def play_scripted(conn, resolver, emb, asker, convs, wrong: bool = False, pad_words: int = 0):
    """Every conversation through the real agent graph, DB and tools, with a scripted model in
    place of Claude: it acts on the referred people (or, wrong=True, on their neighbour / without
    exclusions / network-wide), and its replies list who the tools returned, padded with filler
    to pad_words when pricing. Returns the rows and every model call's input (for the quote)."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
    from langgraph.checkpoint.memory import InMemorySaver

    ctx: dict = {"n": 0, "calls": [], "tools": []}
    name = resolver.name_of
    filler = " This member has experience that is directly relevant to what you asked about."

    def intro_call(exp, lists, short: bool) -> AIMessage:
        pid = resolve_ref(exp["target"], lists)
        if wrong:   # the neighbour on the list instead of the referred person
            people = lists[exp["target"].split(".")[0]]
            pid = people[(people.index(pid) + 1) % len(people)]
        first = name[pid].split()[0]
        text = (f"Hi {first}, would you have 20 minutes for a quick call next week?" if short
                else f"Hi {first}, I run a company in a similar space and your experience stood "
                     "out to me. Would you be open to a 20-minute call next week to compare "
                     "notes? Thanks, and no pressure either way.")
        ctx["n"] += 1
        return AIMessage(content="", tool_calls=[{"name": "send_intro", "id": f"c{ctx['n']}",
                                                  "args": {"recipient": name[pid],
                                                           "message": text}}])

    def recipient(exp, lists):
        pid = resolve_ref(exp["target"], lists)
        if wrong:
            people = lists[exp["target"].split(".")[0]]
            pid = people[(people.index(pid) + 1) % len(people)]
        return pid

    def looked_up(messages, pid) -> bool:
        return any(isinstance(m, ToolMessage) and m.name == "get_person_evidence"
                   and pid in agent._text(m) for m in messages)

    def act(messages) -> AIMessage:
        spec, lists, conv = ctx["spec"], ctx["lists"], ctx["conv"]
        ctx["calls"].append((conv["id"], ctx["turn"], list(messages)))
        last, exp = messages[-1], spec.get("expect") or {}
        drafting = exp.get("kind") == "intro" or (exp.get("kind") == "asks" and wrong)
        if isinstance(last, ToolMessage) and last.name == "send_intro":
            said = agent._text(last)
            if "did not send it and said" in said and exp.get("kind") == "intro":
                return intro_call(exp, lists, short=True)       # "make it shorter" -> redraft
            return AIMessage(content="Done, it's sent." if '"ok": true' in said
                             else "Okay, I won't send it.")
        if exp.get("kind") == "asks" and not wrong and isinstance(last, HumanMessage):
            return AIMessage(content="Happy to. What would you like to talk to them about?")
        if drafting and isinstance(last, (HumanMessage, ToolMessage)) and not any(
                isinstance(m, AIMessage) and any(c["name"] == "send_intro" for c in m.tool_calls)
                for m in messages[messages.index(next(m for m in reversed(messages)
                                                      if isinstance(m, HumanMessage))):]):
            pid = recipient(exp, lists)
            if not looked_up(messages, pid):      # D-056: look the recipient up first
                ctx["n"] += 1
                return AIMessage(content="", tool_calls=[{
                    "name": "get_person_evidence", "id": f"c{ctx['n']}",
                    "args": {"query": "what they could share", "person_refs": [pid]}}])
            return intro_call(exp, lists, short=False)
        if isinstance(messages[-1], ToolMessage):        # write the reply from this turn's results
            shown = []
            for m in reversed(messages):
                if isinstance(m, HumanMessage):
                    break
                if isinstance(m, ToolMessage):
                    out = json.loads(agent._text(m))
                    shown = [p["person_id"] for p in out.get("members", out.get("persons", []))] \
                        + shown
            lines = [f"{i}. {name[p]} fits here [profile: {name[p]}]." for i, p in
                     enumerate(shown, 1)]
            text = "Here is who stands out:\n" + "\n".join(lines)
            short = pad_words - len(text.split())
            return AIMessage(content=text + filler * max(0, short // len(filler.split())))
        ctx["n"] += 1
        call = {"id": f"c{ctx['n']}"}
        exp, text = spec.get("expect") or {}, agent._text(messages[-1])
        if exp.get("kind") in ("lookup", "lookup_set"):
            refs = [exp["target"]] if exp["kind"] == "lookup" else exp["targets"]
            pids = [resolve_ref(r, lists) for r in refs]
            if wrong:   # the neighbour on the list instead of the referred person
                people = lists[refs[0].split(".")[0]]
                pids = [people[(people.index(p) + 1) % len(people)] for p in pids]
            call.update(name="get_person_evidence",
                        args={"query": "their relevant experience", "person_refs": pids})
        elif exp.get("kind") == "exclude":
            shown = [p for label in exp["lists"] for p in lists[label]]
            call.update(name="find_members", args={"criterion": conv["turns"][0]["user"], **(
                {} if wrong else {"exclude_person_ids": [name[p] for p in shown]})})
        elif exp.get("kind") == "within":
            pool = [p for label in exp["lists"] for p in lists[label]]
            call.update(name="find_members", args={"criterion": text, **(
                {} if wrong else {"within_person_ids": pool})})
        else:
            call.update(name="find_members", args={"criterion": text})
        # The real model sometimes writes a line before a tool call; the opening search does
        # here, so the runner's first-words timing is tested against it.
        ahead = "Let me look into that. " if ctx["turn"] == 1 else ""
        return AIMessage(content=ahead, tool_calls=[call])

    class Acting(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "scripted-conversation"

        def bind_tools(self, tools_, **kwargs):
            ctx["tools"] = list(tools_)
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=act(messages))])

        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            msg = act(messages)
            for piece in re.findall(r"\S+\s*", agent._text(msg)):   # text streams first,
                yield ChatGenerationChunk(message=AIMessageChunk(content=piece))  # as in the API
            if msg.tool_calls:
                yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[
                    {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
                    for i, c in enumerate(msg.tool_calls)]))

    def on_turn(spec, lists, conv):
        turn = ctx["turn"] + 1 if ctx.get("conv") is conv else 1
        ctx.update(spec=spec, lists=lists, conv=conv, turn=turn)

    tick = {"t": START_CLOCK}
    agent._clock = lambda: tick["t"]
    box = MemoryOutbox()
    chat = agent.build_agent(emb, conn, resolver, InMemorySaver(), model=Acting(),
                             web_enabled=False, model_picks=False, outbox=box,
                             fact_checker=recording_checker(
                                 lambda *a: factcheck.FactCheck(ok=True)))   # never the network
    try:
        rows = [r for conv in convs
                for r in run_conversation(chat, resolver, asker, conv, tick, on_turn, box)]
    finally:
        agent._clock = time.time
    return rows, ctx


def selfcheck() -> None:
    """The scripted agent twice: acting correctly (every check must pass) and deliberately wrong
    (every check must fail). Also: the pause drops old results, first words are timed, and the
    committable summary names nobody. No API calls."""
    conn, resolver, emb, asker = setup()
    convs = load_conversations()
    right_rows, _ = play_scripted(conn, resolver, emb, asker, convs)
    wrong_rows, _ = play_scripted(conn, resolver, emb, asker, convs, wrong=True)
    right, wrong = summarize(right_rows), summarize(wrong_rows)
    n_checks = sum(1 for c in convs for t in c["turns"] if t.get("expect"))
    assert right["checks"] == {"pass": n_checks, "fail": 0, "setup_failed": 0}, right["checks"]
    assert wrong["checks"]["pass"] == 0, {k: v for k, v in wrong["per_check"].items()
                                          if v["result"] == "pass"}
    print(f"1. acting correctly: {n_checks}/{n_checks} checks pass; acting wrongly: "
          f"{wrong['checks']['fail']} fail, {wrong['checks']['setup_failed']} could not be set "
          "up, 0 pass")
    paused = [r for r in right_rows if r["pause_s"]]
    assert paused and all(r["dropped_through"] >= r["turn"] - agent.KEEP_TURNS for r in paused), \
        [(r["cid"], r["turn"], r["dropped_through"]) for r in paused]
    print(f"2. the pause dropped old search results before the late reference "
          f"({[(r['cid'], r['turn'], r['dropped_through']) for r in paused]})")
    # A reply turn is timed to its first words; a draft turn to its card (drafts are not
    # streamed before the fact-check, D-057).
    timed = [r for r in right_rows if r.get("first_words_s") is not None
             or r.get("card_s") is not None]
    assert len(timed) == right["n_turns"], (len(timed), right["n_turns"])
    assert all(r.get("first_words_s") is None for r in right_rows
               if r.get("card_s") is not None), "draft text reached the screen before the card"
    ahead = [r for r in right_rows if r["turn"] == 1]
    assert all(r["pre_tool_text_s"] is not None and r["first_words_s"] > r["pre_tool_text_s"]
               for r in ahead), [(r["cid"], r["pre_tool_text_s"], r["first_words_s"]) for r in ahead]
    assert not any(r["pre_tool_text_s"] for r in right_rows if r["turn"] > 1)
    print(f"3. all {len(timed)} turns timed (first words, or time to the card on draft turns); "
          f"text written ahead of a tool call is not counted as first words ({len(ahead)} turns)")
    blob = json.dumps(right["per_check"]) + json.dumps(wrong["per_check"])
    assert not set(resolver.full_names(blob)), "a name reached the committable summary"
    print("4. the committable summary names nobody (people appear as list positions)")
    print("conversation self-check passed")


# TUNABLE(quote assumptions: final replies ~450 tokens (measured 309-486 on real follow-ups,
#   2026-10-14), a tool-call message ~80 tokens, x1.4 for the real agent's extra lookups and
#   searches the script does not make. Symptom wrong: a real run costs more than the quote.)
QUOTE_REPLY_TOKENS, QUOTE_TOOL_CALL_TOKENS, QUOTE_MARGIN = 450, 80, 1.4
# TUNABLE(fact-check quote: ~9k tokens in at $0.75/M + ~500 out at $4.50/M ~ $0.009 per check,
#   1.5 checks per draft for rewrites. Symptom wrong: calibration's measured cost per check.)
QUOTE_CHECK_USD, QUOTE_CHECKS_PER_DRAFT = 0.009, 1.5
CACHE_MIN_TOKENS = 4096  # Haiku 4.5's minimum cacheable prefix


def quote() -> None:
    """Price one paid run before making it: the scripted agent plays every conversation with
    realistic reply lengths; every model call's exact input is counted with Anthropic's (free)
    token-counting endpoint, then priced with the agent's automatic caching -- below the
    4,096-token minimum nothing caches; above it, the previous call's input in the same thread is
    read at 0.1x (unless a pause expired the cache) and the rest written at 1.25x."""
    from langchain_anthropic import ChatAnthropic

    conn, resolver, emb, asker = setup()
    convs = load_conversations()
    rows, ctx = play_scripted(conn, resolver, emb, asker, convs,
                              pad_words=QUOTE_REPLY_TOKENS * 3 // 4)
    pauses = {(r["cid"], r["turn"]) for r in rows if r["pause_s"]}
    counter = ChatAnthropic(model=agent.AGENT_MODEL)
    p = cost.PRICING[agent.AGENT_MODEL]
    total, prev, per_conv, sizes = 0.0, {}, {}, []
    for cid, turn, msgs in ctx["calls"]:
        n = counter.get_num_tokens_from_messages(msgs, tools=ctx["tools"])
        sizes.append(n)
        last_n, last_turn = prev.get(cid, (0, None))
        cold = (cid, turn) in pauses and last_turn != turn   # first call after the pause
        read = last_n if n >= CACHE_MIN_TOKENS and last_n >= CACHE_MIN_TOKENS and not cold else 0
        written = n - read if n >= CACHE_MIN_TOKENS else 0
        plain = n if n < CACHE_MIN_TOKENS else 0
        out = QUOTE_REPLY_TOKENS if isinstance(msgs[-1], ToolMessage) else QUOTE_TOOL_CALL_TOKENS
        usd = (plain * p["input"] + read * p["input"] * cost._CACHE_READ_MULT
               + written * p["input"] * cost._CACHE_WRITE_MULT + out * p["output"]) / 1e6
        prev[cid] = (n, turn)
        per_conv[cid] = per_conv.get(cid, 0) + usd
        total += usd
    print(f"{len(ctx['calls'])} scripted model calls over {len(rows)} turns; input per call "
          f"{min(sizes):,}-{max(sizes):,} tokens")
    print("per conversation: " + ", ".join(f"{k} ${v:.3f}" for k, v in per_conv.items()))
    drafts = sum(1 for r in rows if r.get("draft"))
    checks_usd = drafts * QUOTE_CHECKS_PER_DRAFT * QUOTE_CHECK_USD
    print(f"scripted total ${total:.3f}; quote with x{QUOTE_MARGIN} for the real agent's extra "
          f"calls: ${total * QUOTE_MARGIN:.2f} (ceiling in code ${SPEND_CEILING_USD})")
    print(f"fact-checks (second provider, billed separately): {drafts} drafts x "
          f"{QUOTE_CHECKS_PER_DRAFT} checks x ${QUOTE_CHECK_USD} = ${checks_usd:.3f}")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    elif "--quote" in sys.argv:
        quote()
    else:
        main()
