"""The coordinator loop (D-036) and its harness law (the parts the model cannot skip).

Shape of a turn: the model sees the whole thread plus the tool menu and works in rounds --
call tools, read results, repeat -- until it calls `respond` with a DECLARED outcome. With
tool_choice forced, there is no bare-text path out of the loop: every turn ends in a
machine-readable outcome the deterministic checks can trust.

HARNESS LAW lives here, in code, not in the prompt (a prompt is a request; a harness is a
law):
  * ROUND BUDGET  -- the loop cannot run away. On exhaustion the model gets one forced
    respond round, then the turn fails loudly.
  * DUPLICATE BREAKER -- an identical (tool, args) repeat gets a structured error instead
    of a second execution; a stalled loop announces itself in the trace.
  * BYPASS TRIPWIRE -- D-013's logic promoted to the agent era: a declared "answer" with no
    retrieval call in the trajectory is checked by NAME SCAN -- every member name in the
    response must trace to retrieval evidence, the user's own words, or the asker. A
    greeting may say "welcome back"; it has no business naming a member. Violation raises;
    zero tolerance. (Residue -- a factual claim naming nobody -- is the sampled judge's job,
    Phase 5.)
  * OUTCOME VALIDATION -- clarify only ever follows an ambiguous resolution (G-001);
    refuse is the EXACT contract string (D-019's detector anchors on it); the envelope is
    validated at the end of every turn.
  * NOMINATE-VS-APPOINT (D-028) is enforced one layer down, in tools.py: the model's
    person references never become ids except through the resolver.

ANSWER PATH (Step 5, done): a declared `answer` does NOT use the model's respond-text.
It runs a SEPARATE streamed call under the pinned synth-v1 prompt (synthesize.py), over
this turn's whole thread. Owner's call: the product is conversational, so the writer needs
the conversation. Two consequences live here rather than in the prompt:
  * `respond` no longer asks for `text` on an answer -- the answer is written by the synth
    call, so a coordinator-written draft would be output tokens paid for and discarded
    (measured: 432 wasted output tokens on the first smoke turn).
  * The asker's own words are recorded as UserStatedEvidence. The writer can see them in
    the thread, so it WILL use them ("as you mentioned, your Vietnam expansion") -- and the
    groundedness judge reads env.evidence, not the thread. Without this the judge scores
    correct conversational memory as an unsupported claim: a false alarm on the primary
    metric. Typed, so a user's claim can never become a fact about a member (no person_id).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from contracts import (
    Envelope,
    Resolution,
    ResultSetArtifact,
    ToolCall,
    UserStatedEvidence,
    answered_without_retrieval,
    validate,
)
from synthesize import REFUSAL_STRING
from tracing import record_llm_call, tracer

COORD_CONTRACT_VERSION = "coord-v1"
PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts" / "coordinator"
CORE_FILE = PROMPT_DIR / "core-v1.md"

# TUNABLE(5 rounds per turn: the census's worst within-turn trajectory is 2-3 tool rounds
#         plus respond; 5 = that plus headroom. Symptom wrong: budget-abort spans on real
#         questions in traces -> raise; p95 rounds should sit near 2.)
MAX_ROUNDS = 5

# Coordinator rounds are decisions, not prose; the cap keeps a confused round from writing
# an essay. The final user-facing text is NOT generated under this cap (synth call owns it).
ROUND_MAX_TOKENS = 1024


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").strip()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def core_template() -> str:
    return _read(CORE_FILE)


def build_system(asker_id: str, asker_name: str) -> str:
    """Pinned core + the per-session asker line. The asker line is outside the pinned
    bytes by design -- same status as retrieved context in synth-v1: config records the
    CORE hash, and the assembled prompt varies per session without breaking comparability."""
    return f"{core_template()}\n\n## Session\nASKER: {asker_id} ({asker_name})"


def contract_versions() -> dict[str, str]:
    return {
        "coord_contract_version": COORD_CONTRACT_VERSION,
        "core_sha256": _sha256(core_template()),
    }


# The turn's exit door. Schema-forced enum: the model can never invent an outcome, and the
# harness never infers one from prose (declared, not inferred).
RESPOND_TOOL: dict = {
    "name": "respond",
    "description": (
        "End the turn with a declared outcome. outcome=answer for a substantive reply "
        "grounded in this conversation's evidence (put the self-contained question being "
        "answered in standalone_query); refuse when the evidence cannot support an answer; "
        "clarify ONLY after a tool reported an ambiguous person reference; redirect for "
        "advice questions, after finding the relevant members to point at. Do NOT write the "
        "answer text yourself for outcome=answer -- it is written separately from the "
        "evidence you gathered. `text` is required only for clarify and redirect."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "outcome": {"type": "string",
                        "enum": ["answer", "refuse", "clarify", "redirect"]},
            "text": {"type": "string",
                     "description": "The user-facing message. Required for clarify and "
                                    "redirect; omit for answer and refuse."},
            "standalone_query": {
                "type": "string",
                "description": "outcome=answer only: the question being answered, "
                               "self-contained, pronouns and ordinals resolved.",
            },
        },
        "required": ["outcome"],
    },
}


class BypassViolation(RuntimeError):
    """A declared answer named a member with no evidence trail. Zero tolerance."""


class BudgetExhausted(RuntimeError):
    """The model would not end the turn inside the round budget."""


class AnswerPathMissing(RuntimeError):
    """A turn declared `answer` with no synthesize_fn wired. Hard failure on purpose: the
    old placeholder path would silently hand a metric the coordinator's draft instead of a
    synth-v1 answer, which is exactly what Step 5 existed to prevent."""


class Coordinator:
    def __init__(self, provider, toolbox_factory: Callable, resolver, store,
                 synthesize_fn: Callable | None = None):
        """toolbox_factory(asker_id) -> Toolbox (or a test double with the same three
        tool methods). synthesize_fn(env, messages, tools) -> provider.ModelTurn is the
        pinned synth-v1 answer path; build it with synthesize.make_answer_writer."""
        self.provider = provider
        self.toolbox_factory = toolbox_factory
        self.resolver = resolver
        self.store = store
        self.synthesize_fn = synthesize_fn
        self.tools = None  # set per turn

    # --- the loop -----------------------------------------------------------------------

    def run_turn(self, thread, user_text: str) -> Envelope:
        toolbox = self.toolbox_factory(thread.asker)
        env = Envelope(query=user_text, asker=thread.asker, thread_id=thread.thread_id)
        system = build_system(thread.asker, self.resolver.name_of.get(thread.asker, ""))
        tool_menu = list(getattr(toolbox, "schemas", [])) + [RESPOND_TOOL]

        messages = list(thread.messages)
        messages.append({"role": "user", "content": user_text})
        # C5: the asker's own words, typed as evidence so the judge sees what the writer sees.
        env.evidence.append(UserStatedEvidence(text=user_text, turn=len(thread.messages),
                                               thread_id=thread.thread_id))
        seen_calls: set[str] = set()
        respond_args: dict | None = None
        trc = tracer("coordinator")

        # ONE parent span per turn, so every round and tool call nests under a single
        # trace -- the first-divergence debugging story (D-031) depends on this; orphan
        # root spans were the first thing the first real trace read got wrong.
        with trc.start_as_current_span("turn") as turn_span:
            turn_span.set_attribute("thread_id", thread.thread_id)
            respond_args, rounds_used = self._run_rounds(
                toolbox, env, thread, messages, system, tool_menu, seen_calls, trc)
            if respond_args is None:
                turn_span.set_attribute("outcome", "budget_exhausted")
                raise BudgetExhausted(
                    f"no respond after {MAX_ROUNDS} rounds + forced final (thread "
                    f"{thread.thread_id})"
                )
            self._finalize(env, respond_args, user_text, messages, tool_menu, trc)
            turn_span.set_attribute("outcome", env.response_mode or "")
            # THIS turn's rounds. The first version counted assistant messages in
            # `messages`, which carries the whole thread -- so a 2-round turn reported 8 on
            # the third turn of a conversation. The defense pack's overhead row reads this
            # number, so an inflated count would have overstated the loop's own cost.
            turn_span.set_attribute("rounds", rounds_used)

        # Close the thread record: the final text as an ordinary assistant turn -- so the
        # NEXT turn's model reads the conversation exactly as the user experienced it.
        messages.append({"role": "assistant", "content": env.response})
        thread.messages = messages
        self.store.save(thread)
        return env

    def _run_rounds(self, toolbox, env, thread, messages, system, tool_menu,
                    seen_calls, trc) -> tuple[dict | None, int]:
        respond_args: dict | None = None
        for round_no in range(MAX_ROUNDS + 1):
            forced_final = round_no == MAX_ROUNDS
            if forced_final:
                # Budget exhausted: one last round where respond is the ONLY tool.
                messages.append({"role": "user", "content": (
                    "[harness] Round budget reached. Call respond now with the most "
                    "honest outcome the evidence gathered so far supports."
                )})
                menu = [RESPOND_TOOL]
            else:
                menu = tool_menu

            with trc.start_as_current_span("coordinator_round") as span:
                turn = self.provider.complete(system, messages, tools=menu,
                                              force_tool=True, max_tokens=ROUND_MAX_TOKENS)
                record_llm_call(span, model=getattr(self.provider, "model", "unknown"),
                                usage={"input_tokens": turn.input_tokens,
                                       "output_tokens": turn.output_tokens})
                span.set_attribute("round", round_no)

            messages.append({"role": "assistant", "content": turn.raw_content})

            results = []
            for call in turn.tool_calls:
                if call.name == "respond":
                    respond_args = call.args
                    results.append(self._tool_result(call.id, {"ok": True}, ok=True))
                    continue
                results.append(self._execute(toolbox, env, thread, call, seen_calls, trc))
            if results:
                messages.append({"role": "user", "content": results})

            if respond_args is not None:
                return respond_args, round_no + 1
        return None, MAX_ROUNDS + 1

    # --- execution ------------------------------------------------------------------------

    def _execute(self, toolbox, env: Envelope, thread, call, seen: set[str], trc) -> dict:
        key = f"{call.name}:{json.dumps(call.args, sort_keys=True)}"
        if key in seen:
            # A stalled loop announces itself instead of burning a second execution.
            payload = {"ok": False, "error": "duplicate_call",
                       "detail": "You already made this exact call this turn; use its "
                                 "result, or take a different action."}
            env.record_tool_call(ToolCall(tool=call.name, args=call.args, ok=False,
                                          error="duplicate_call"))
            return self._tool_result(call.id, payload, ok=False)
        seen.add(key)

        method = getattr(toolbox, call.name, None)
        if method is None:
            payload = {"ok": False, "error": "unknown_tool",
                       "detail": f"{call.name!r} is not a tool."}
            env.record_tool_call(ToolCall(tool=call.name, args=call.args, ok=False,
                                          error="unknown_tool"))
            return self._tool_result(call.id, payload, ok=False)

        with trc.start_as_current_span(call.name) as span:
            try:
                payload = method(**call.args)
            except TypeError as exc:  # bad/missing args from the model: a result, not a crash
                payload = {"ok": False, "error": "bad_arguments", "detail": str(exc)}
            span.set_attribute("ok", bool(payload.get("ok")))

        ok = bool(payload.get("ok"))
        env.record_tool_call(ToolCall(tool=call.name, args=call.args, ok=ok,
                                      error=payload.get("error", "") if not ok else ""))

        if ok:
            env.evidence.extend(payload.get("evidence", []))
            if call.name == "find_members":
                art = ResultSetArtifact(
                    artifact_id=thread.next_artifact_id(),
                    criterion=call.args.get("criterion", ""),
                    person_ids=tuple(m["person_id"] for m in payload.get("members", [])),
                    turn=len(thread.messages),
                    thread_id=thread.thread_id,
                )
                env.artifacts.append(art)
                thread.artifacts.append(art)
        elif payload.get("error") == "ambiguous_person":
            # Record the ambiguity so clarify's precondition (G-001) is checkable state,
            # not an inference from prose.
            env.resolution = Resolution(
                status="ambiguous",
                candidates={m: [c["person_id"] for c in cands]
                            for m, cands in payload.get("candidates", {}).items()},
            )

        return self._tool_result(call.id, payload, ok=ok)

    @staticmethod
    def _tool_result(call_id: str, payload: dict, ok: bool) -> dict:
        body = dict(payload)
        for k, v in body.items():
            if isinstance(v, list):
                body[k] = [e.to_dict() if hasattr(e, "to_dict") else e for e in v]
            elif isinstance(v, dict):
                body[k] = {kk: ([e.to_dict() if hasattr(e, "to_dict") else e for e in vv]
                                if isinstance(vv, list) else vv)
                           for kk, vv in v.items()}
        return {"type": "tool_result", "tool_use_id": call_id,
                "content": json.dumps(body, default=lambda o: o.to_dict()),
                "is_error": not ok}

    # --- finalization: harness law ----------------------------------------------------------

    def _finalize(self, env: Envelope, args: dict, user_text: str, messages: list[dict],
                  tool_menu: list[dict], trc) -> None:
        outcome = args["outcome"]
        env.response_mode = outcome
        env.resolved_query = args.get("standalone_query", "")

        if outcome == "refuse":
            # The exact contract sentence, always -- D-019's deterministic detector anchors
            # on this string; a generated paraphrase would blind it.
            env.response = REFUSAL_STRING
        elif outcome == "answer":
            if self.synthesize_fn is None:
                raise AnswerPathMissing(
                    "declared outcome 'answer' with no synthesize_fn; wire "
                    "synthesize.make_answer_writer(provider) before serving or scoring"
                )
            # The separate pinned call. It reads the SAME messages the coordinator read, so
            # ordinals, corrections and stated facts are available to the writer; the tool
            # menu rides along only to satisfy the wire format (tool_choice "none").
            with trc.start_as_current_span("synthesize") as span:
                turn = self.synthesize_fn(env, messages, tool_menu)
                record_llm_call(
                    span,
                    model=getattr(self.provider, "synth_model", "unknown"),
                    usage={"input_tokens": turn.input_tokens,
                           "output_tokens": turn.output_tokens},
                )
                span.set_attribute("stop_reason", turn.stop_reason)
            env.response = turn.text.strip()
        else:
            # clarify and redirect ARE the model's text -- there is no evidence to synthesize
            # from, only a question to ask or people to point at.
            text = args.get("text", "").strip()
            if not text:
                raise ValueError(
                    f"outcome {outcome!r} requires text on the respond call and none was given"
                )
            env.response = text

        if answered_without_retrieval(env):
            self._name_scan(env, user_text)

        validate(env)

    def _name_scan(self, env: Envelope, user_text: str) -> None:
        """Name-provenance check for the no-retrieval answer path. Deterministic and
        universal: 268 names are a closed set, so every member name in an outgoing
        no-retrieval answer must appear in the user's own message (echoes are honest) or
        be the asker. Anything else is a bypass -- the answer names a member it never
        looked up."""
        spotted = self._member_ids(env.response)
        allowed = self._member_ids(user_text) | {env.asker}
        leaked = spotted - allowed
        if leaked:
            raise BypassViolation(
                f"answer names member(s) {sorted(leaked)} with no retrieval call in the "
                f"trajectory (thread {env.thread_id})"
            )

    def _member_ids(self, text: str) -> set[str]:
        res = self.resolver.resolve(text)
        ids = set(res.person_ids)
        for cands in res.candidates.values():
            ids.update(cands)
        return ids


if __name__ == "__main__":
    # Pure self-check: scripted provider + fake toolbox + fake answer writer + in-memory
    # store. No DB, no API. The answer writer is a DOUBLE, not the placeholder that used to
    # live in _finalize -- the point of Step 5 is that nothing can read a coordinator draft.
    from dataclasses import dataclass, field as dc_field

    from contracts import CorpusEvidence
    from contracts import UserStatedEvidence as USE
    from provider import ModelTurn, ToolCallRequest
    from resolve import Resolver

    people = [{"person_id": "p-jane", "canonical_name": "Jane Rivera"},
              {"person_id": "p-dana", "canonical_name": "Dana Whitfield"}]
    resolver = Resolver(people)

    @dataclass
    class FakeThread:
        thread_id: str = "t-test"
        asker: str = "p-jane"
        messages: list = dc_field(default_factory=list)
        artifacts: list = dc_field(default_factory=list)

        def next_artifact_id(self):
            return f"rs-{len(self.artifacts) + 1}"

    class FakeStore:
        def save(self, thread):
            pass

    class FakeToolbox:
        schemas = []

        def find_members(self, criterion, exclude_person_ids=None, within_person_ids=None):
            ev = CorpusEvidence(person_id="p-dana", doc_id="Dana Whitfield", chunk_id="c#h5",
                                text="...", score=0.8, section=5)
            return {"ok": True, "members": [{"person_id": "p-dana", "name": "Dana Whitfield",
                                             "best_score": 0.8, "snippets": [ev]}],
                    "evidence": [ev], "candidates_considered": 1}

    class ScriptedProvider:
        model = "fake"
        synth_model = "fake-synth"

        def __init__(self, script):
            self.script = list(script)

        def complete(self, system, messages, tools, force_tool, max_tokens):
            calls = self.script.pop(0)
            return ModelTurn(text="", tool_calls=tuple(calls), stop_reason="tool_use",
                             raw_content=[{"type": "tool_use", "id": c.id, "name": c.name,
                                           "input": c.args} for c in calls])

    class FakeWriter:
        """Stands in for the pinned synth-v1 call; records what it was handed."""

        def __init__(self, text):
            self.text = text
            self.seen_messages = None
            self.seen_tools = None

        def __call__(self, env, messages, tools):
            self.seen_messages, self.seen_tools = messages, tools
            return ModelTurn(text=self.text, tool_calls=(), stop_reason="end_turn",
                             input_tokens=11, output_tokens=7)

    def coord(script, answer="An answer.", writer=True):
        w = FakeWriter(answer) if writer else None
        c = Coordinator(ScriptedProvider(script), lambda asker: FakeToolbox(),
                        resolver, FakeStore(), synthesize_fn=w)
        return c, w

    # 1 -- happy path: search, then answer. The ANSWER comes from the writer, not respond.
    c, w = coord([
        [ToolCallRequest("1", "find_members", {"criterion": "pharma"})],
        [ToolCallRequest("2", "respond", {"outcome": "answer",
                                          "standalone_query": "who is in pharma?"})],
    ], answer="Dana Whitfield fits what you described.")
    env = c.run_turn(FakeThread(), "Anyone in pharma I can talk to?")
    assert env.response_mode == "answer" and len(env.artifacts) == 1
    assert env.response == "Dana Whitfield fits what you described."
    assert [cl.tool for cl in env.tool_calls] == ["find_members"]
    print("1. answer: text comes from the pinned writer, artifact recorded")

    # 2 -- the writer is handed the THREAD and the tool menu (the owner's whole-thread call).
    assert w.seen_messages is not None and len(w.seen_messages) >= 3
    assert any(m.get("role") == "user" for m in w.seen_messages)
    assert w.seen_tools is not None, "tool menu must ride along for the wire format"
    print(f"2. writer saw {len(w.seen_messages)} messages + the tool menu")

    # 3 -- C5: the asker's own words are on the envelope as typed evidence, no person_id.
    stated = [e for e in env.evidence if isinstance(e, USE)]
    assert len(stated) == 1 and stated[0].text == "Anyone in pharma I can talk to?"
    assert "person_id" not in stated[0].to_dict()
    print("3. user-stated evidence recorded, carries no person_id")

    # 4 -- greeting: a no-retrieval answer naming nobody still passes the tripwire.
    c, _ = coord([[ToolCallRequest("1", "respond", {"outcome": "answer"})]],
                 answer="Good to see you again. What can I help with?")
    env = c.run_turn(FakeThread(), "Hi")
    assert env.response.startswith("Good to see you")
    print("4. greeting: no-retrieval answer with no member names passes")

    # 5 -- bypass: the tripwire now reads the SYNTH output, which is what users see.
    try:
        c, _ = coord([[ToolCallRequest("1", "respond", {"outcome": "answer"})]],
                     answer="Dana Whitfield runs a China supply chain.")
        c.run_turn(FakeThread(), "Hi")
        raise AssertionError("bypass tripwire failed to fire")
    except BypassViolation as e:
        print(f"5. bypass tripwire fired on the written answer: {e}")

    # 6 -- duplicate call: second identical call gets a structured error, not an execution.
    c, _ = coord([
        [ToolCallRequest("1", "find_members", {"criterion": "pharma"})],
        [ToolCallRequest("2", "find_members", {"criterion": "pharma"})],
        [ToolCallRequest("3", "respond", {"outcome": "answer", "standalone_query": "who?"})],
    ], answer="Dana Whitfield.")
    env = c.run_turn(FakeThread(), "Anyone in pharma?")
    assert len([cl for cl in env.tool_calls if cl.error == "duplicate_call"]) == 1
    print("6. duplicate breaker: repeat call structured-errored")

    # 7 -- clarify without ambiguity is an envelope violation (G-001 precondition).
    try:
        c, _ = coord([[ToolCallRequest("1", "respond",
                                       {"outcome": "clarify", "text": "Which one?"})]])
        c.run_turn(FakeThread(), "Tell me about Ross")
        raise AssertionError("clarify precondition failed to fire")
    except ValueError as e:
        print(f"7. clarify precondition held: {e}")

    # 8 -- clarify/redirect still owe their text; an empty one fails loudly.
    try:
        c, _ = coord([[ToolCallRequest("1", "respond", {"outcome": "redirect"})]])
        c.run_turn(FakeThread(), "How should I pivot to SaaS?")
        raise AssertionError("missing redirect text failed to fire")
    except ValueError as e:
        print(f"8. redirect without text rejected: {e}")

    # 9 -- refusal is the exact contract string regardless of anything the model wrote.
    c, _ = coord([[ToolCallRequest("1", "respond", {"outcome": "refuse",
                                                   "text": "Sorry, no idea."})]])
    env = c.run_turn(FakeThread(), "What is Dana Whitfield's shoe size?")
    assert env.response == REFUSAL_STRING
    print("9. refuse: exact contract string enforced")

    # 10 -- no answer path wired = hard failure, never a silent coordinator draft.
    try:
        c, _ = coord([[ToolCallRequest("1", "respond", {"outcome": "answer"})]], writer=False)
        c.run_turn(FakeThread(), "Hi")
        raise AssertionError("missing answer path failed to fire")
    except AnswerPathMissing as e:
        print(f"10. unwired answer path rejected: {e}")

    print("coordinator harness self-check passed")
