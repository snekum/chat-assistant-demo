"""The product agent (D-042): a standard LangChain agent on the LangGraph runtime.

Owner's call, 2026-10-13: get a usable chat product now, rework when complexity actually grows.
The Phase-3 coordinator (coordinator.py: separate pinned writer, declared outcomes, greeting
fast path) stays in the repo as the measured instrument the agent baselines were cut with; this
module is what the chat interface serves.

What the framework now owns -- the custom code it replaces:
  * the tool loop and message plumbing      (coordinator._run_rounds / _execute)
  * per-conversation memory, keyed by thread (session.py -> LangGraph PostgresSaver, same DB)
  * the provider adapter                     (provider.py -> langchain_anthropic.ChatAnthropic)
  * the round budget                         (MAX_ROUNDS -> ModelCallLimitMiddleware)

What stays ours, because it is the product rather than plumbing:
  * the tools and the resolver (tools.py, resolve.py) -- unchanged, wrapped as LangChain tools
  * UNSEEN-ID RULE (D-028 / A15), inside the tools: a raw person_id is accepted only if the
    conversation already showed it; the model once invented guessable slugs
  * NAME CHECK (A5 / A12), an after-model hook: a final reply naming a member nobody looked up
    is replaced with the refusal sentence BEFORE it is saved or shown -- fails safe
  * one merged system prompt (prompts/agent/core-v1.md) with the refusal and clarify rules

A standard agent writes its own reply, so a greeting with no tool call is ONE model call by
construction -- the D-041 fast path existed only because of the separate writer.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import psycopg
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    ModelCallLimitMiddleware,
    ModelRequest,
    after_model,
    dynamic_prompt,
)
from langchain.tools import ToolRuntime, tool
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row

import store
from synthesize import REFUSAL_STRING
from tools import SCHEME, Toolbox

log = logging.getLogger("agent")

PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "agent" / "core-v1.md"
AGENT_MODEL = "claude-haiku-4-5"  # the model every measured run used; swap is a BIG call

# TUNABLE(6 model calls per user message: the measured trajectories peaked at 5 (4 tool rounds +
#   the reply); symptom wrong: turns ending with no reply at the limit -> raise.)
MODEL_CALL_LIMIT = 6


@dataclass
class Asker:
    """Who is speaking -- passed per request as runtime context, never stored in the prompt file."""

    person_id: str
    name: str
    profile_summary: str  # executive-summary chunk of their own dossier, loaded by SQL


def _text(msg) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


# Display boundary (D-043): profile, [self] and [stated] tags come OFF the screen and stay in
# the stored text. Users cannot open a dossier, so a profile citation is unverifiable clutter
# for them; the tags exist for the system -- fabricated-citation scoring (D-020), claim-level
# verification later, and tracing a complaint to the excerpt that produced it. [web: ...] tags
# are kept: once web search exists they become links a user CAN check.
_HIDDEN_TAG = re.compile(r"[ \t]*\[(?:profile:[^\]]*|self|stated)\]", re.IGNORECASE)


def render_reply(text: str) -> str:
    """What the member sees. Never feed this to the eval, the judge or the name check."""
    cleaned = _HIDDEN_TAG.sub("", text)
    lines = (re.sub(r"[ \t]{2,}", " ", line).rstrip() for line in cleaned.splitlines())
    return "\n".join(lines).strip()


def _shown_text(messages) -> str:
    """What the conversation SHOWED the model: user text, tool results, finished replies --
    never the model's own tool-call arguments (an id typed there is what the ID rule catches)."""
    parts = []
    for m in messages:
        if isinstance(m, (HumanMessage, ToolMessage)):
            parts.append(_text(m))
        elif isinstance(m, AIMessage) and not m.tool_calls:
            parts.append(_text(m))
    return "\n".join(parts)


def _dump(payload: dict) -> str:
    return json.dumps(payload, default=lambda o: o.to_dict() if hasattr(o, "to_dict") else str(o))


def build_agent(emb, conn, resolver, checkpointer, model=None):
    """One compiled agent serves every member; who is asking arrives as runtime context."""
    core = PROMPT_FILE.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
    core = core.format(refusal_string=REFUSAL_STRING)

    def unseen_ids(refs, runtime: ToolRuntime) -> str | None:
        ids = [r.strip() for r in refs or [] if isinstance(r, str) and r.strip() in resolver.name_of]
        if not ids:
            return None
        shown = _shown_text(runtime.state["messages"])
        bad = [i for i in ids if i != runtime.context.person_id and i not in shown]
        if not bad:
            return None
        return _dump({"ok": False, "error": "unseen_person_id",
                      "detail": f"{bad} did not appear in this conversation. Pass people as the "
                                "user named them (name or company), or use person_ids from "
                                "earlier tool results. Never construct an id."})

    @tool
    def find_members(criterion: str, runtime: ToolRuntime,
                     exclude_person_ids: list[str] | None = None,
                     within_person_ids: list[str] | None = None) -> str:
        """Discover members of the network who match a criterion (an industry, an experience,
        a company, a deal). The member you are speaking with is never included.

        Args:
            criterion: what to look for, in plain words.
            exclude_person_ids: person_ids to leave out (for "anyone else?").
            within_person_ids: narrow an earlier result list to these person_ids.
        """
        if err := unseen_ids(within_person_ids, runtime):
            return err
        box = Toolbox(emb, conn, resolver, asker=runtime.context.person_id)
        return _dump(box.find_members(criterion, exclude_person_ids, within_person_ids))

    @tool
    def get_person_evidence(query: str, person_refs: list[str], runtime: ToolRuntime,
                            full_dossier: bool = False) -> str:
        """Facts about specific people. One person for a fact or brief; several to compare.

        Args:
            query: what you want to know about them.
            person_refs: people as the user named them (name or company), or person_ids from
                earlier tool results. Never construct a person_id.
            full_dossier: fetch whole dossiers, for meeting-prep briefs.
        """
        if err := unseen_ids(person_refs, runtime):
            return err
        box = Toolbox(emb, conn, resolver, asker=runtime.context.person_id)
        return _dump(box.get_person_evidence(query, person_refs, full_dossier))

    @dynamic_prompt
    def system_prompt(request: ModelRequest) -> str:
        a: Asker = request.runtime.context
        return (f"{core}\n\n## The member you are speaking with\nName: {a.name}\n"
                f"person_id: {a.person_id}\nTheir own profile summary (cite it as [self]):\n"
                f"{a.profile_summary}")

    @after_model
    def name_check(state: AgentState, runtime) -> dict | None:
        """Fail safe: a final reply naming a member the conversation never surfaced is replaced
        with the refusal sentence before it is saved or shown. Full-name pairs only (A12)."""
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None
        allowed = set(resolver.full_names(_shown_text(state["messages"][:-1])))
        allowed.add(runtime.context.person_id)
        leaked = set(resolver.full_names(_text(last))) - allowed
        if not leaked:
            return None
        log.warning("name check replaced a reply naming %d unretrieved member(s)", len(leaked))
        return {"messages": [AIMessage(id=last.id, content=REFUSAL_STRING)]}

    if model is None:
        # Automatic prompt caching (A11), passed straight through to the Messages API. The
        # system prompt is stable per member and the thread is append-only, so later turns
        # re-read earlier ones at ~0.1x.
        model = ChatAnthropic(model=AGENT_MODEL, max_tokens=2048,
                              model_kwargs={"cache_control": {"type": "ephemeral"}})

    return create_agent(
        model,
        tools=[find_members, get_person_evidence],
        middleware=[system_prompt, name_check,
                    ModelCallLimitMiddleware(run_limit=MODEL_CALL_LIMIT, exit_behavior="end")],
        context_schema=Asker,
        checkpointer=checkpointer,
    )


def open_checkpointer() -> PostgresSaver:
    """Conversation memory in the SAME Postgres the corpus lives in (LangGraph's own tables)."""
    conn = psycopg.connect(store.DSN, autocommit=True, prepare_threshold=0, row_factory=dict_row)
    saver = PostgresSaver(conn)
    saver.setup()
    return saver


def load_asker(conn, resolver, person_id: str) -> Asker:
    rows = store.fetch_person_chunks(conn, person_id, scheme=SCHEME)[:1]
    return Asker(person_id=person_id, name=resolver.name_of.get(person_id, ""),
                 profile_summary=rows[0]["text"] if rows else "")


class ChatService:
    """What the interface talks to: one agent, one checkpointer, many threads."""

    def __init__(self):
        from embedder import NomicLocal
        from tools import build_resolver

        self.conn = store.connect()
        self.resolver = build_resolver(self.conn)
        self.emb = NomicLocal()
        self.emb.warm()  # model load at startup, not on the first question (Step-5 finding)
        self.agent = build_agent(self.emb, self.conn, self.resolver, open_checkpointer())

    def members(self) -> list[tuple[str, str]]:
        return sorted((name, pid) for pid, name in self.resolver.name_of.items())

    def ask(self, thread_id: str, person_id: str, text: str):
        """Yields ("status", str) while tools run, then exactly one ("reply", str)."""
        asker = load_asker(self.conn, self.resolver, person_id)
        config = {"configurable": {"thread_id": thread_id}}
        for update in self.agent.stream({"messages": [{"role": "user", "content": text}]},
                                        config=config, context=asker, stream_mode="updates"):
            for node in update.values():
                for m in (node or {}).get("messages", []) if isinstance(node, dict) else []:
                    for call in getattr(m, "tool_calls", None) or []:
                        what = call["args"].get("criterion") or call["args"].get("query") or ""
                        yield "status", f"Looking into: {what}" if what else "Looking into it"
        final = self.agent.get_state(config).values["messages"][-1]
        reply = _text(final) if isinstance(final, AIMessage) and not final.tool_calls else ""
        yield "reply", reply or REFUSAL_STRING


if __name__ == "__main__":
    # Self-check with a SCRIPTED fake model: real Postgres, real tools, real middleware, real
    # checkpointer semantics (in-memory) -- and no API call. Asserts only; prints no names.
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langgraph.checkpoint.memory import InMemorySaver

    from embedder import NomicLocal
    from tools import build_resolver

    class Scripted(GenericFakeChatModel):
        calls: int = 0

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, *a, **k):
            self.calls += 1
            return super()._generate(*a, **k)

    conn = store.connect()
    resolver = build_resolver(conn)
    emb = NomicLocal()
    ids = sorted(resolver.name_of)
    me, other = ids[0], ids[-1]
    asker = load_asker(conn, resolver, me)

    def run(script, text, thread="t1", saver=None):
        model = Scripted(messages=iter(script))
        agent = build_agent(emb, conn, resolver, saver or InMemorySaver(), model=model)
        cfg = {"configurable": {"thread_id": thread}}
        agent.invoke({"messages": [{"role": "user", "content": text}]}, config=cfg, context=asker)
        msgs = agent.get_state(cfg).values["messages"]
        return msgs, model.calls

    # 1 -- greeting: plain reply, no tools, exactly one model call.
    msgs, n = run([AIMessage(content="Hello! What can I help you find today?")], "Hi")
    assert n == 1 and _text(msgs[-1]).startswith("Hello"), (n, msgs[-1])
    print("1. greeting: 1 model call")

    # 2 -- search then answer: the tool runs against the real corpus.
    script = [AIMessage(content="", tool_calls=[{"name": "find_members", "id": "c1",
                                                 "args": {"criterion": "fintech payments"}}]),
              AIMessage(content="Here are some people.")]
    msgs, n = run(script, "Anyone in fintech?")
    tool_out = json.loads(next(m.content for m in msgs if isinstance(m, ToolMessage)))
    assert n == 2 and tool_out["ok"] and tool_out["members"], tool_out.get("error")
    print(f"2. search then answer: {len(tool_out['members'])} members found, 2 model calls")

    # 3 -- name check: a reply naming a member nobody looked up becomes the refusal sentence.
    msgs, _ = run([AIMessage(content=f"You should meet {resolver.name_of[other]}.")], "Who?")
    assert _text(msgs[-1]) == REFUSAL_STRING, _text(msgs[-1])
    print("3. name check replaced an unretrieved name with the refusal sentence")

    # 4 -- unseen-id rule: a constructed person_id is rejected inside the tool.
    script = [AIMessage(content="", tool_calls=[{"name": "get_person_evidence", "id": "c2",
                                                 "args": {"query": "q", "person_refs": [other]}}]),
              AIMessage(content=REFUSAL_STRING)]
    msgs, _ = run(script, "When was that company sold?")
    out = json.loads(next(m.content for m in msgs if isinstance(m, ToolMessage)))
    assert out.get("error") == "unseen_person_id", out
    print("4. invented person_id rejected inside the tool")

    # 5 -- memory: the second message in the same thread sees the first exchange.
    saver = InMemorySaver()
    run([AIMessage(content="Hello there.")], "Hi", thread="mem", saver=saver)
    msgs, _ = run([AIMessage(content="Still here.")], "You there?", thread="mem", saver=saver)
    assert [type(m).__name__ for m in msgs] == ["HumanMessage", "AIMessage"] * 2, msgs
    print("5. thread memory: second turn carries the first")
    # 6 -- display: profile/self/stated tags hidden, web tags and line breaks kept.
    raw = ("He founded it in 2021 [profile: Jane Rivera]. You met in Lagos [stated].\n\n"
           "Your firm grew [self]. Rates rose [web: Reuters, 2026].")
    shown = render_reply(raw)
    assert shown == ("He founded it in 2021. You met in Lagos.\n\nYour firm grew. "
                     "Rates rose [web: Reuters, 2026]."), shown
    print("6. display hides profile/self/stated tags, keeps web tags and paragraphs")
    print("agent self-check passed")
