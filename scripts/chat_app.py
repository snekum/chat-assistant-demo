"""Chat interface (D-042): a browser chat window over the product agent (src/agent.py).

Usage:  python scripts/chat_app.py      then open the printed local URL.
Needs Postgres up (Docker Desktop) and ANTHROPIC_API_KEY in .env. Every message you send is a
real, billed model call (~$0.01-0.03 per message at Haiku 4.5 prices).

Quick and dirty on purpose: a member picker stands in for login, tool steps show while it
works, the reply streams in as it is written (D-047: first words within 3 s), and "New chat"
starts a fresh thread. Asked to message someone, the agent drafts and fact-checks it, then waits:
edit the text, "Approve and send" or "Don't send" (D-055; sending = a row in the outbox table).
Conversations persist in Postgres, so a thread survives a restart of this app.
"""
from __future__ import annotations

import logging
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
# Per-turn timing in the terminal ("turn timing: first words 2.1s, full reply 6.0s"), the D-047
# check while you chat; other libraries stay at WARNING.
logging.basicConfig(format="%(asctime)s %(name)s: %(message)s")
logging.getLogger("agent").setLevel(logging.INFO)

import gradio as gr  # noqa: E402

from agent import STATUS_SEP, ChatService, render_partial, render_reply  # noqa: E402

HIDDEN = gr.update(visible=False)

# TUNABLE(1.5 s per step line: long enough to read four or five words, short enough that a 7 s
#   news wait shows every step and then the closing line. Symptom wrong: testers say a line changed
#   before they could read it -> raise; the screen still looks stuck on one line -> lower.)
STEP_SECONDS = 1.5
POLL_SECONDS = 0.5            # how often the screen checks whether the next line is due
WRITING = "Putting your answer together"


class StepTicker:
    """D-063 (option B): steps the agent started together are shown one after another on a
    timer, then a closing line held until text appears. Display only: the timing is NOT real
    progress (the steps run at the same time); the agent's events and the eval's step timings
    stay real. A single step is held as it is, so intro-draft statuses never rotate."""

    def __init__(self, step: float = STEP_SECONDS):
        self.step, self.lines, self.i, self.since = step, [], 0, 0.0

    def start(self, status: str, now: float) -> str:
        parts = status.split(STATUS_SEP)
        self.lines = parts + [WRITING] if len(parts) > 1 else parts
        self.i, self.since = 0, now
        return self.lines[0]

    def tick(self, now: float) -> str | None:
        """The next line if it is due, else None (nothing to redraw)."""
        if self.i + 1 >= len(self.lines) or now - self.since < self.step:
            return None
        self.i, self.since = self.i + 1, now
        return self.lines[self.i]

    def stop(self) -> None:
        self.lines = []


def _ticking(events, poll: float = POLL_SECONDS):
    """The agent's events, plus ("tick", None) every `poll` s while the agent is quiet. The agent
    runs in a background thread so the screen can change during a long tool step. Closing this
    generator (the member starts a new chat) stops the agent at its next event."""
    q, done, stop = queue.Queue(), object(), threading.Event()

    def pump():
        try:
            for event in events:
                if stop.is_set():
                    break
                q.put(event)
        except Exception as exc:                  # re-raised on the screen side
            q.put(("_error", exc))
        finally:
            getattr(events, "close", lambda: None)()
            q.put(done)

    threading.Thread(target=pump, daemon=True).start()
    try:
        while True:
            try:
                item = q.get(timeout=poll)
            except queue.Empty:
                yield "tick", None
                continue
            if item is done:
                return
            if item[0] == "_error":
                raise item[1]
            yield item
    finally:
        stop.set()


def _drive(events, history: list, thread_id: str, step: float = STEP_SECONDS,
           poll: float = POLL_SECONDS):
    """Screen updates for one stream of agent events -> (chat, thread, input box, approval
    panel, its note, the editable draft). A turn ends with the reply, or with a message
    waiting for the member's approval (D-055): then the panel opens with the draft in it."""
    kept = ""   # answer text that stays while later steps run (D-061)
    ticker = StepTicker(step)

    def show(line: str) -> None:
        history[-1]["content"] = (f"{kept}\n\n" if kept else "") + f"_{line}..._"

    stream = _ticking(events, poll)
    try:
        for kind, payload in stream:
            if kind == "tick":
                line = ticker.tick(time.monotonic())
                if line is None:
                    continue
                show(line)
            elif kind == "kept":
                kept = render_reply(payload)
                continue
            elif kind == "status":
                show(ticker.start(payload, time.monotonic()))
            elif kind == "partial":
                ticker.stop()
                history[-1]["content"] = render_partial(payload)
            elif kind == "approval":
                # D-058: a poor-fit heads-up stays in the conversation, not only on the card.
                heads_up = f"_Heads-up: {payload['note']}_\n\n" if payload.get("note") else ""
                history[-1]["content"] = heads_up + (
                    f"Here's a draft to **{payload['recipient_name']}**. Edit it below if you "
                    "like, then approve to send. Nothing goes out until you do.")
                yield (history, thread_id, "", gr.update(visible=True),
                       payload["description"], payload["message"])
                return
            else:  # the stored reply is authoritative: links in, guard decisions applied
                history[-1]["content"] = render_reply(payload, svc.links(thread_id))
            yield history, thread_id, "", HIDDEN, gr.update(), gr.update()
    except Exception as exc:  # show it in the window; a demo should never die silently
        history[-1]["content"] = f"Something went wrong: {type(exc).__name__}: {exc}"
        yield history, thread_id, "", HIDDEN, gr.update(), gr.update()
    finally:
        stream.close()       # a turn the member abandoned stops at the agent's next event


def selfcheck() -> None:
    """Offline, no model, no DB: the D-063 rotation and the background-thread plumbing."""
    # 1. The ticker: steps in order on the timer, then the closing line, then nothing new.
    t = StepTicker(step=1.5)
    assert t.start(f"Searching the web: tariffs{STATUS_SEP}Finding members: steel", 0) \
        == "Searching the web: tariffs"
    assert t.tick(1.0) is None and t.tick(1.5) == "Finding members: steel"
    assert t.tick(3.0) == WRITING and t.tick(9.0) is None
    assert t.start("Checking the draft against their profile", 0) and t.tick(9.0) is None

    def screens(events, history=None):
        history = history or [{"role": "assistant", "content": ""}]
        out = []
        for _ in _drive(events, history, "t", step=0.1, poll=0.02):
            if not out or out[-1] != history[-1]["content"]:
                out.append(history[-1]["content"])
        return out

    def fake(*steps):
        for s in steps:
            if isinstance(s, (int, float)):
                time.sleep(s)
            else:
                yield s

    # 2. Steps started together rotate while the agent is quiet, then the answer replaces them.
    got = screens(fake(("status", f"A{STATUS_SEP}B"), 0.45, ("partial", "Hello")))
    assert got == ["_A..._", "_B..._", f"_{WRITING}..._", render_partial("Hello")], got
    # 3. A single step holds (draft statuses never rotate).
    got = screens(fake(("status", "Drafting your message"), 0.3, ("partial", "Hi")))
    assert got == ["_Drafting your message..._", render_partial("Hi")], got
    # 4. Kept text (D-061) stays above every rotated line.
    got = screens(fake(("kept", "Kept part."), ("status", f"A{STATUS_SEP}B"), 0.15,
                       ("partial", "Kept part. Rest")))
    top = render_reply("Kept part.")
    assert all(s.startswith(top) for s in got[:-1]) and len(got) >= 3, got
    # 5. An agent error still reaches the window.
    def boom():
        yield "status", "A"
        raise RuntimeError("boom")
    assert screens(boom())[-1] == "Something went wrong: RuntimeError: boom"
    # 6. Abandoning a turn stops the agent at its next event (the thread does not run on).
    closed = threading.Event()
    def long_turn():
        try:
            yield "status", f"A{STATUS_SEP}B"
            for _ in range(50):
                time.sleep(0.05)
                yield "partial", "x"
        finally:
            closed.set()
    drive = _drive(long_turn(), [{"role": "assistant", "content": ""}], "t", 0.1, 0.02)
    next(drive)
    drive.close()
    assert closed.wait(1.0), "agent kept running after the turn was abandoned"
    print("chat app self-check passed (6 checks)")


if "--selfcheck" in sys.argv:
    selfcheck()
    sys.exit(0)

svc = ChatService()
MEMBERS = svc.members()  # (display name, person_id), from the DB at runtime -- never in source


def send(message: str, history: list, person_id: str, thread_id: str):
    if not message.strip():
        yield history, thread_id, "", gr.update(), gr.update(), gr.update()
        return
    thread_id = thread_id or f"ui-{uuid.uuid4().hex[:12]}"
    history = list(history) + [{"role": "user", "content": message},
                               {"role": "assistant", "content": "_Thinking..._"}]
    # Typing while a draft waits declines it with these words as the reason (ChatService.ask).
    yield history, thread_id, "", HIDDEN, gr.update(), gr.update()
    yield from _drive(svc.ask(thread_id, person_id, message), history, thread_id)


def decide(approve: bool):
    def handler(draft: str, history: list, person_id: str, thread_id: str):
        said = ("Send it." if approve else "Don't send it.")
        history = list(history) + [{"role": "user", "content": said},
                                   {"role": "assistant", "content": "_Sending..._" if approve
                                    else "_Okay..._"}]
        yield history, thread_id, "", HIDDEN, gr.update(), gr.update()
        yield from _drive(svc.decide(thread_id, person_id, approve, draft if approve else None),
                          history, thread_id)
    return handler


def reset():
    return [], "", HIDDEN


with gr.Blocks(title="Network Assistant") as demo:
    gr.Markdown("### Network Assistant")
    with gr.Row():
        who = gr.Dropdown(choices=MEMBERS, value=MEMBERS[0][1], label="Signed in as", scale=4)
        new = gr.Button("New chat", scale=1)
    chat = gr.Chatbot(height=560, label="Conversation")
    with gr.Group(visible=False) as approval:
        note = gr.Markdown()
        draft = gr.Textbox(lines=7, label="Message (edit it if you like)")
        with gr.Row():
            send_btn = gr.Button("Approve and send", variant="primary")
            skip_btn = gr.Button("Don't send")
    box = gr.Textbox(placeholder="Ask about the network, or just say hi", show_label=False)
    thread = gr.State("")

    outputs = [chat, thread, box, approval, note, draft]
    box.submit(send, [box, chat, who, thread], outputs)
    send_btn.click(decide(True), [draft, chat, who, thread], outputs)
    skip_btn.click(decide(False), [draft, chat, who, thread], outputs)
    new.click(reset, None, [chat, thread, approval])
    who.change(reset, None, [chat, thread, approval])

if __name__ == "__main__":
    demo.launch()
