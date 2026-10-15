"""Chat interface (D-042): a browser chat window over the product agent (src/agent.py).

Usage:  python scripts/chat_app.py      then open the printed local URL.
Needs Postgres up (Docker Desktop) and ANTHROPIC_API_KEY in .env. Every message you send is a
real, billed model call (~$0.01-0.03 per message at Haiku 4.5 prices).

Quick and dirty on purpose: a member picker stands in for login, tool steps show while it
works, the reply streams in as it is written (D-047: first words within 3 s), and "New chat"
starts a fresh thread. Asked to message someone, the agent drafts it (streaming), then waits:
edit the text, "Approve and send" or "Don't send" (D-055; sending = a row in the outbox table).
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


HIDDEN = gr.update(visible=False)


def _drive(events, history: list, thread_id: str):
    """Screen updates for one stream of agent events -> (chat, thread, input box, approval
    panel, its note, the editable draft). A turn ends with the reply, or with a message
    waiting for the member's approval (D-055): then the panel opens with the draft in it."""
    try:
        for kind, payload in events:
            if kind == "status":
                history[-1]["content"] = f"_{payload}..._"
            elif kind == "partial":
                history[-1]["content"] = render_partial(payload)
            elif kind == "draft":       # the message being written, before it is complete
                history[-1]["content"] = f"_Drafting your message..._\n\n{payload} ▌"
            elif kind == "approval":
                history[-1]["content"] = (
                    f"Here's a draft to **{payload['recipient_name']}**. Edit it below if you "
                    "like, then approve to send. Nothing goes out until you do.")
                yield (history, thread_id, "", gr.update(visible=True),
                       f"**{payload['description']}**", payload["message"])
                return
            else:  # the stored reply is authoritative: links in, guard decisions applied
                history[-1]["content"] = render_reply(payload, svc.links(thread_id))
            yield history, thread_id, "", HIDDEN, gr.update(), gr.update()
    except Exception as exc:  # show it in the window; a demo should never die silently
        history[-1]["content"] = f"Something went wrong: {type(exc).__name__}: {exc}"
        yield history, thread_id, "", HIDDEN, gr.update(), gr.update()


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
