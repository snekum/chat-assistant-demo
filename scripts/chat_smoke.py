"""Live wired smoke of the D-036 coordinator loop: real corpus, real resolver, real
Haiku coordinator, real harness law. One or two turns, then prints the trajectory.

Usage:  python scripts/chat_smoke.py "Is there anyone in fintech I could talk to?"
        python scripts/chat_smoke.py            (runs the default two-turn census-C1 shape)

This is a SMOKE, not an eval: it proves the plumbing (loop -> tools -> evidence ->
declared outcome -> session persistence -> trace spans) end to end. Answer quality is
Step 5's business (synth-v1 wiring + re-baseline).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # repo convention (eval/run.py)

import store  # noqa: E402
from coordinator import Coordinator, contract_versions  # noqa: E402
from embedder import NomicLocal  # noqa: E402
from provider import AnthropicProvider  # noqa: E402
from session import SessionStore  # noqa: E402
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
    sessions = SessionStore(conn)
    provider = AnthropicProvider()

    with conn.cursor() as cur:
        cur.execute("SELECT person_id FROM persons ORDER BY person_id LIMIT 1")
        asker = cur.fetchone()[0]

    coord = Coordinator(
        provider,
        toolbox_factory=lambda a: Toolbox(emb, conn, resolver, asker=a),
        resolver=resolver,
        store=sessions,
    )
    print(f"contract: {contract_versions()}  coordinator model: {provider.model}")
    thread = sessions.create(asker)
    print(f"thread {thread.thread_id}  asker {asker}\n")

    turns = sys.argv[1:] or [
        "Hi",
        "Is there anyone with fintech experience I could talk to?",
        "Tell me more about the first person",
    ]
    for user_text in turns:
        print(f"USER: {user_text}")
        env = coord.run_turn(thread, user_text)
        print(f"  outcome={env.response_mode}  tools={[c.tool for c in env.tool_calls]}  "
              f"evidence={len(env.evidence)}  artifacts={[a.artifact_id for a in env.artifacts]}")
        print(f"BOT: {env.response[:400]}\n")

    reloaded = sessions.load(thread.thread_id)
    print(f"persistence: {len(reloaded.messages)} messages, "
          f"{len(reloaded.artifacts)} artifacts survive reload")
    shutdown()
    print(f"trace: {trace_path}")


if __name__ == "__main__":
    main()
