"""Chat interface (D-042): a browser chat window over the product agent (src/agent.py).

Usage:  python scripts/chat_app.py      then open the printed local URL.
Needs Postgres up (Docker Desktop) and ANTHROPIC_API_KEY in .env. Every message you send is a
real, billed model call (~$0.01-0.03 per message at Haiku 4.5 prices).

Quick and dirty on purpose: a member picker stands in for login, tool steps show while it
works, the reply streams in as it is written (D-047: first words within 3 s), and "New chat"
starts a fresh thread.
Conversations persist in Postgres, so a thread survives a restart of this app.
"""
from __future__ import annotations

import logging
import sys
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

from agent import ChatService, render_partial, render_reply  # noqa: E402

svc = ChatService()
MEMBERS = svc.members()  # (display name, person_id), from the DB at runtime -- never in source


def send(message: str, history: list, person_id: str, thread_id: str):
    if not message.strip():
        yield history, thread_id, ""
        return
    thread_id = thread_id or f"ui-{uuid.uuid4().hex[:12]}"
    history = list(history) + [{"role": "user", "content": message},
                               {"role": "assistant", "content": "_Thinking..._"}]
    yield history, thread_id, ""
    try:
        for kind, text in svc.ask(thread_id, person_id, message):
            if kind == "status":
                history[-1]["content"] = f"_{text}..._"
            elif kind == "partial":
                history[-1]["content"] = render_partial(text)
            else:  # the stored reply is authoritative: links in, guard decisions applied
                history[-1]["content"] = render_reply(text, svc.links(thread_id))
            yield history, thread_id, ""
    except Exception as exc:  # show it in the window; a demo should never die silently
        history[-1]["content"] = f"Something went wrong: {type(exc).__name__}: {exc}"
        yield history, thread_id, ""


def reset():
    return [], ""


with gr.Blocks(title="Network Assistant") as demo:
    gr.Markdown("### Network Assistant")
    with gr.Row():
        who = gr.Dropdown(choices=MEMBERS, value=MEMBERS[0][1], label="Signed in as", scale=4)
        new = gr.Button("New chat", scale=1)
    chat = gr.Chatbot(height=560, label="Conversation")
    box = gr.Textbox(placeholder="Ask about the network, or just say hi", show_label=False)
    thread = gr.State("")

    box.submit(send, [box, chat, who, thread], [chat, thread, box])
    new.click(reset, None, [chat, thread])
    who.change(reset, None, [chat, thread])

if __name__ == "__main__":
    demo.launch()
