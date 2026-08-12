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

OWED TO STEP 5, marked where it happens: the `answer` outcome currently returns the
model's respond-text as a placeholder. The real answer path is a SEPARATE streamed
generation under the pinned synth-v1 prompt (D-032's byte-comparability instrument) --
wired at Step 5 with the re-baseline, so no metric ever reads the placeholder.
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
        "advice questions, after finding the relevant members to point at."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "outcome": {"type": "string",
                        "enum": ["answer", "refuse", "clarify", "redirect"]},
            "text": {"type": "string",
                     "description": "The user-facing message for this outcome."},
            "standalone_query": {
                "type": "string",
                "description": "outcome=answer only: the question being answered, "
                               "self-contained, pronouns and ordinals resolved.",
            },
        },
        "required": ["outcome", "text"],
    },
}


class BypassViolation(RuntimeError):
    """A declared answer named a member with no evidence trail. Zero tolerance."""


class BudgetExhausted(RuntimeError):
    """The model would not end the turn inside the round budget."""


class Coordinator:
    def __init__(self, provider, toolbox_factory: Callable, resolver, store,
                 synthesize_fn: Callable | None = None):
        """toolbox_factory(asker_id) -> Toolbox (or a test double with the same three
        tool methods). synthesize_fn lands at Step 5 (the pinned synth-v1 answer path)."""
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
        seen_calls: set[str] = set()
        respond_args: dict | None = None
        trc = tracer("coordinator")

        # ONE parent span per turn, so every round and tool call nests under a single
        # trace -- the first-divergence debugging story (D-031) depends on this; orphan
        # root spans were the first thing the first real trace read got wrong.
        with trc.start_as_current_span("turn") as turn_span:
            turn_span.set_attribute("thread_id", thread.thread_id)
            respond_args = self._run_rounds(toolbox, env, thread, messages, system,
                                            tool_menu, seen_calls, trc)
            if respond_args is None:
                turn_span.set_attribute("outcome", "budget_exhausted")
                raise BudgetExhausted(
                    f"no respond after {MAX_ROUNDS} rounds + forced final (thread "
                    f"{thread.thread_id})"
                )
            self._finalize(env, respond_args, user_text)
            turn_span.set_attribute("outcome", env.response_mode or "")
            turn_span.set_attribute("rounds", len([m for m in messages
                                                   if m.get("role") == "assistant"]))

        # Close the thread record: the final text as an ordinary assistant turn -- so the
        # NEXT turn's model reads the conversation exactly as the user experienced it.
        messages.append({"role": "assistant", "content": env.response})
        thread.messages = messages
        self.store.save(thread)
        return env

    def _run_rounds(self, toolbox, env, thread, messages, system, tool_menu,
                    seen_calls, trc) -> dict | None:
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
                return respond_args
        return None

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

    def _finalize(self, env: Envelope, args: dict, user_text: str) -> None:
        outcome = args["outcome"]
        env.response_mode = outcome
        env.resolved_query = args.get("standalone_query", "")

        if outcome == "refuse":
            # The exact contract sentence, always -- D-019's deterministic detector anchors
            # on this string; a generated paraphrase would blind it.
            env.response = REFUSAL_STRING
        elif outcome == "answer" and self.synthesize_fn is not None:
            env.response = self.synthesize_fn(env)  # Step 5: pinned synth-v1, streamed
        else:
            # answer (placeholder until Step 5), clarify, redirect: the model's text.
            env.response = args.get("text", "")

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
    # Pure self-check: scripted provider + fake toolbox + in-memory store. No DB, no API.
    from dataclasses import dataclass, field as dc_field

    from contracts import CorpusEvidence
    from provider import ModelTurn, ToolCallRequest
    from resolve import Resolver

    people = [{"person_id": "p-aaron", "canonical_name": "Aaron Silva"},
              {"person_id": "p-craig", "canonical_name": "Craig Hunter"}]
    resolver = Resolver(people)

    @dataclass
    class FakeThread:
        thread_id: str = "t-test"
        asker: str = "p-aaron"
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
            ev = CorpusEvidence(person_id="p-craig", doc_id="Craig Hunter", chunk_id="c#h5",
                                text="...", score=0.8, section=5)
            return {"ok": True, "members": [{"person_id": "p-craig", "name": "Craig Hunter",
                                             "best_score": 0.8, "snippets": [ev]}],
                    "evidence": [ev], "candidates_considered": 1}

    class ScriptedProvider:
        model = "fake"

        def __init__(self, script):
            self.script = list(script)

        def complete(self, system, messages, tools, force_tool, max_tokens):
            calls = self.script.pop(0)
            return ModelTurn(text="", tool_calls=tuple(calls), stop_reason="tool_use",
                             raw_content=[{"type": "tool_use", "id": c.id, "name": c.name,
                                           "input": c.args} for c in calls])

    def coord(script):
        return Coordinator(ScriptedProvider(script), lambda asker: FakeToolbox(),
                           resolver, FakeStore())

    # 1 -- happy path: search, then answer. Artifact created, bypass silent (retrieval ran).
    env = coord([
        [ToolCallRequest("1", "find_members", {"criterion": "pharma"})],
        [ToolCallRequest("2", "respond", {"outcome": "answer",
                                          "text": "Craig Hunter fits.",
                                          "standalone_query": "who is in pharma?"})],
    ]).run_turn(FakeThread(), "Anyone in pharma I can talk to?")
    assert env.response_mode == "answer" and len(env.artifacts) == 1
    assert [c.tool for c in env.tool_calls] == ["find_members"]
    print("1. answer-with-retrieval: outcome declared, artifact recorded")

    # 2 -- greeting: answer with no tools and no member names is allowed.
    env = coord([[ToolCallRequest("1", "respond",
                                  {"outcome": "answer", "text": "Hello! How can I help?"})]]
                ).run_turn(FakeThread(), "Hi")
    assert env.response == "Hello! How can I help?"
    print("2. greeting: no-retrieval answer with no member names passes")

    # 3 -- bypass: naming a member with no retrieval trips the wire.
    try:
        coord([[ToolCallRequest("1", "respond",
                                {"outcome": "answer",
                                 "text": "Craig Hunter runs a China supply chain."})]]
              ).run_turn(FakeThread(), "Hi")
        raise AssertionError("bypass tripwire failed to fire")
    except BypassViolation as e:
        print(f"3. bypass tripwire fired: {e}")

    # 4 -- duplicate call: second identical call gets a structured error, not an execution.
    env = coord([
        [ToolCallRequest("1", "find_members", {"criterion": "pharma"})],
        [ToolCallRequest("2", "find_members", {"criterion": "pharma"})],
        [ToolCallRequest("3", "respond", {"outcome": "answer", "text": "Craig Hunter.",
                                          "standalone_query": "who?"})],
    ]).run_turn(FakeThread(), "Anyone in pharma?")
    dup = [c for c in env.tool_calls if c.error == "duplicate_call"]
    assert len(dup) == 1
    print("4. duplicate breaker: repeat call structured-errored")

    # 5 -- clarify without ambiguity is an envelope violation (G-001 precondition).
    try:
        coord([[ToolCallRequest("1", "respond",
                                {"outcome": "clarify", "text": "Which one?"})]]
              ).run_turn(FakeThread(), "Tell me about Ross")
        raise AssertionError("clarify precondition failed to fire")
    except ValueError as e:
        print(f"5. clarify precondition held: {e}")

    # 6 -- refusal is the exact contract string regardless of the model's text.
    env = coord([[ToolCallRequest("1", "respond",
                                  {"outcome": "refuse", "text": "Sorry, no idea."})]]
                ).run_turn(FakeThread(), "What is Craig Hunter's shoe size?")
    assert env.response == REFUSAL_STRING
    print("6. refuse: exact contract string enforced")

    print("coordinator harness self-check passed")
