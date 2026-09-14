"""Live wired smoke of the D-036 coordinator loop: real corpus, real resolver, real
Haiku coordinator, real harness law, and (Step 5) the real pinned synth-v1 answer call.

Usage:  python scripts/chat_smoke.py "Is there anyone in fintech I could talk to?"
        python scripts/chat_smoke.py            (runs the default three-turn census-C1 shape)

Needs Postgres up (docker compose up -d) and ANTHROPIC_API_KEY in .env.

This is a SMOKE, not an eval: it proves the plumbing (loop -> tools -> evidence -> declared
outcome -> pinned answer call -> session persistence -> trace spans) end to end. Scored
numbers come from the Step-5 re-baseline, not from here.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Windows consoles default to cp1252, which renders every em dash in the corpus as a black
# diamond. The DATA is fine -- checked: the stored chunks hold U+2014 -- but a transcript you
# read or paste would misrepresent it, so force UTF-8 on the way out.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # repo convention (eval/run.py)

import store  # noqa: E402
from coordinator import Coordinator, contract_versions  # noqa: E402
from embedder import NomicLocal  # noqa: E402
from provider import AnthropicProvider  # noqa: E402
from session import SessionStore  # noqa: E402
from synthesize import (  # noqa: E402
    answer_contract_versions,
    make_answer_writer,
    strip_display_markers,
)
from tools import Toolbox, build_resolver  # noqa: E402
from tracing import init_tracing, shutdown  # noqa: E402


def main() -> None:
    trace_path = Path("analysis") / "chat_smoke_trace.jsonl"
    # ProgressProcessor hands the callback the already-translated user-facing MESSAGE
    # (tracing.py routes span names through progress_message before calling).
    init_tracing(str(trace_path), progress=lambda msg, _attrs: print(f"  [{msg}]"))

    conn = store.connect()
    resolver = build_resolver(conn)
    emb = NomicLocal()
    # Pay the model load at STARTUP, not on the first question. The pre-Step-5 trace charged
    # 32.9s of one-time initialization to `find_members` on turn 1; steady-state embedding is
    # ~56ms. Timed out loud here so a regression in load cost is visible rather than folded
    # into a tool's latency.
    t0 = time.perf_counter()
    emb.warm()
    print(f"embedder warm in {time.perf_counter() - t0:.2f}s")

    sessions = SessionStore(conn)
    provider = AnthropicProvider()

    with conn.cursor() as cur:
        cur.execute("SELECT person_id FROM persons ORDER BY person_id LIMIT 1")
        asker = cur.fetchone()[0]

    # The pinned answer path: streams deltas straight to stdout so perceived latency (memo
    # section 8) is something you can watch rather than something we assert.
    write_answer = make_answer_writer(
        provider, on_delta=lambda d: print(d, end="", flush=True)
    )

    coord = Coordinator(
        provider,
        toolbox_factory=lambda a: Toolbox(emb, conn, resolver, asker=a),
        resolver=resolver,
        store=sessions,
        synthesize_fn=write_answer,
    )
    print(f"coordinator: {contract_versions()}  model: {provider.model}")
    print(f"answer path: {answer_contract_versions()}  model: {provider.synth_model}")
    thread = sessions.create(asker)
    print(f"thread {thread.thread_id}  asker {asker}\n")

    turns = sys.argv[1:] or [
        "Hi",
        "Is there anyone with fintech experience I could talk to?",
        "Tell me more about the first person",
    ]
    for user_text in turns:
        print(f"USER: {user_text}")
        t0 = time.perf_counter()
        print("BOT: ", end="", flush=True)  # the answer streams in here
        env = coord.run_turn(thread, user_text)
        elapsed = time.perf_counter() - t0
        if env.response_mode != "answer":
            print(env.response, end="")  # clarify/refuse/redirect do not stream
        print()
        display = strip_display_markers(env.response)
        if display != env.response:
            print(f"  DISPLAY (markers stripped): {display}")
        print(f"  outcome={env.response_mode}  tools={[c.tool for c in env.tool_calls]}  "
              f"evidence={len(env.evidence)}  artifacts={[a.artifact_id for a in env.artifacts]}"
              f"  {elapsed:.1f}s\n")

    reloaded = sessions.load(thread.thread_id)
    print(f"persistence: {len(reloaded.messages)} messages, "
          f"{len(reloaded.artifacts)} artifacts survive reload")
    shutdown()
    print(f"trace: {trace_path}")


if __name__ == "__main__":
    main()
