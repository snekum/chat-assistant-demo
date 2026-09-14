"""Phase 3 handoff contracts (D-029, amended D-036/D-037): the shapes the coordinator loop,
its tools, the harness, and the evals all share.

These live in a plain module with no framework import at all (D-037: the substrate is a
hand-rolled loop, and nothing a metric touches depends on it). Every shape here is
unit-testable on its own.

TWO versioned shapes (D-029 chose "one envelope + hard-typed evidence" over one loose dict and
over per-seam contracts):

  HANDOFF_SCHEMA_VERSION   -- the per-turn envelope the loop, tools, and evals share
  EVIDENCE_SCHEMA_VERSION  -- the evidence items, which come in three STRICT kinds

The evidence split is the load-bearing part. CorpusEvidence carries person_id; WebEvidence has
no person_id field AT ALL -- not empty, absent. A web result therefore cannot be attributed to
a member no matter what later code does (census open-call 2, the source-scoping invariant).
Putting a Reuters claim in a real semi-public person's mouth is the exact reputational failure
D-009 exists to prevent, and an absent field is the one thing a refactor cannot quietly undo.
UserStatedEvidence (D-036 census C5) is the third kind: facts the asker supplied in
conversation ("we discussed collaborating on X"). Citable in drafts, never dossier-grade,
never written back to any corpus -- and like web, it has no person_id to put words in a
member's mouth.

FLOW/LANE ARE EVAL-SIDE METADATA, NOT CONTROL FLOW (D-036). The D-024 router died; the
coordinator model chooses tools per turn. Flow labels survive on GOLD ROWS as
expected-behavior metadata, and derive_lane() survives so evals can state which sources a
gold question should have touched. Product code neither classifies flow nor derives lane.
What the product DECLARES per turn is response_mode (the outcome), which the harness
validates -- a declared label the deterministic checks can trust (LEGAL_MODES, clarify rule,
bypass tripwire in coordinator harness law).

Both versions are snapshotted into run config exactly like PROMPT_CONTRACT_VERSION and
RUBRIC_VERSION (D-021): a shape change re-defines the system under test.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

# Bump when the envelope's fields change; runs across versions are not comparable.
# v2 (2026-08-11, D-036): + tool_calls (trajectory record) + artifacts (session result sets);
# no run artifact was ever written at v1, so nothing became incomparable.
HANDOFF_SCHEMA_VERSION = "v2"

# Bump when an evidence kind gains/loses/renames a field, or a kind is added/removed.
# Versioned SEPARATELY from the envelope so adding an envelope field does not invalidate
# evidence-level comparability. v2 (2026-08-11, D-036): + UserStatedEvidence (census C5).
EVIDENCE_SCHEMA_VERSION = "v2"

# --- The three enums the eval scores -------------------------------------------------------

# What shape of work a question needs. EVAL-SIDE since D-036: no router classifies this at
# runtime; gold rows carry it as expected-behavior metadata, and the tool-choice confusion
# matrix (G-006) is read against it.
Flow = Literal[
    "person_fact",   # targeted question about resolved person(s), incl. self-queries
    "comparison",    # 2+ resolved people, symmetric per-person retrieval
    "brief",         # meeting-prep summary; whole-doc retrieval
    "aggregate",     # people discovery; capped top-5 ranked by asker relevance
    "topic",         # subject-matter question, not scoped to a named person
    "advice",        # pure advice with no people-ask -> redirect mode
    "composition",   # grounded artifact (intro email draft)
    "meta",          # "what can you do?" -- canned, no retrieval
    "off_domain",    # junk / out of scope -> refuse
]

# Whether the QUESTION scopes itself to the network by its own wording ("what do the fintech
# leaders IN THE NETWORK think about AI?") or leaves the source open ("what pressures are
# community banks facing?"). A property of the words the user typed, NOT a routing verdict --
# which is what makes it safe for the LLM to supply and cheap for stage-1 cues to pre-empt.
# Closes G-005: a network-scoped question must never be answered with web commentary about
# non-members, and the cheapest way to guarantee that is to never fetch it (D-030).
Scope = Literal["network", "open"]

# Where evidence may come from. DERIVED from (flow, scope); never predicted.
Lane = Literal["corpus", "web", "both", "neither"]

# What the response does. Fixed by the requirements memo section 4.
ResponseMode = Literal["answer", "refuse", "clarify", "redirect"]

# How a flow label was produced. Product-side this died with the D-026 cascade (D-036); it
# survives for gold tooling ("rules" = deterministically derivable label, "llm" = LLM-assisted
# authoring), so label provenance stays reportable in eval artifacts.
FlowSource = Literal["rules", "llm"]

# What name resolution concluded. Drives behaviour independently of flow: a not_found person
# short-circuits regardless of what shape of work the question asked for.
ResolutionStatus = Literal["resolved", "ambiguous", "not_found", "none_named"]

# Dossier section 12 grades its own facts; a low-confidence fact must not be asserted flatly.
# "ungraded" = the section stated no grade for this span, which is NOT the same as low.
ConfidenceGrade = Literal["high", "medium", "low", "ungraded"]

# --- Lane derivation -----------------------------------------------------------------------

# EVAL-SIDE TABLE since D-036: states which sources a gold question of each flow SHOULD touch;
# no runtime router reads it. The reasoning it encodes still binds the coordinator PROMPT:
# a similarity score cannot tell "the corpus has this" from "the corpus does not" -- measured
# on run 20260730T022029Z-5d5bcf2, where answerable questions scored top-1 0.586-0.849 and
# unanswerable ones 0.642-0.831, near-total overlap -- so topic questions consult BOTH corpus
# and web and synthesis grounds per-source in whichever half is real.
# TUNABLE(both-on-topic as the coordinator's prompted default; symptom wrong: web evidence
#         goes uncited on >70% of topic turns -> stop prompting the web call; uncited on <30%
#         -> the corpus half is the dead weight. Revisit at the first conversational run.)
FLOW_TO_LANE: dict[str, str | None] = {
    "person_fact": "corpus",
    "comparison": "corpus",
    "brief": "corpus",
    "aggregate": "corpus",
    "advice": "corpus",       # redirect still needs real people to point at
    "composition": "corpus",
    "topic": None,            # the ONLY scope-dependent flow -- see TOPIC_SCOPE_TO_LANE
    "meta": "neither",        # canned capability answer, no retrieval
    "off_domain": "neither",
}

# Topic flow is where lane genuinely varies, and scope is the whole reason it varies (D-030).
# A network-scoped topic question is answered from the network, full stop; an open one fans out.
TOPIC_SCOPE_TO_LANE: dict[str, str] = {"network": "corpus", "open": "both"}

# NOTE: "web" is currently UNREACHABLE by derivation -- no flow maps to web-only, because topic
# flow always fans out. It stays in the Lane enum deliberately: it is exactly what the TUNABLE
# above would produce if the always-both policy is downgraded. An unreachable value that names
# its own trigger is documentation, not dead code.

# Which response modes each flow may legally end in. Since D-036 this is HARNESS LAW read
# against the coordinator's DECLARED outcome (on gold rows, where flow is known) and an eval
# table: a (flow, mode) pair outside it is an instrument bug, not a model failure.
LEGAL_MODES: dict[str, set[str]] = {
    "person_fact": {"answer", "refuse", "clarify"},
    "comparison": {"answer", "refuse", "clarify"},
    "brief": {"answer", "refuse", "clarify"},
    "aggregate": {"answer", "refuse", "clarify"},  # refuse covers computed-superlative
    "topic": {"answer", "refuse"},
    "advice": {"redirect", "refuse"},
    "composition": {"answer", "refuse", "clarify"},
    "meta": {"answer"},
    "off_domain": {"refuse"},
}


def derive_lane(flow: str, scope: str = "open") -> str:
    """Eval-side derivation (D-036): which sources a gold question of this flow should touch.
    Kept a function so gold tooling and trajectory checks call the same code.

    scope is consulted for topic flow ONLY -- every other flow's lane is fixed by what the flow
    IS, so a network-scoped person_fact query is still just corpus.
    """
    if flow not in FLOW_TO_LANE:
        raise ValueError(f"unknown flow {flow!r}; expected one of {sorted(FLOW_TO_LANE)}")
    lane = FLOW_TO_LANE[flow]
    if lane is not None:
        return lane
    if scope not in TOPIC_SCOPE_TO_LANE:
        raise ValueError(f"unknown scope {scope!r}; expected one of {sorted(TOPIC_SCOPE_TO_LANE)}")
    return TOPIC_SCOPE_TO_LANE[scope]


# --- Evidence: two strict kinds ------------------------------------------------------------


@dataclass(frozen=True)
class CorpusEvidence:
    """One retrieved dossier chunk. Person-attributable BY DESIGN -- this is the only kind that
    may carry an identity."""

    person_id: str
    doc_id: str                     # the dossier's display name; what citations tag
    chunk_id: str
    text: str
    score: float                    # cosine similarity; DIAGNOSTIC ONLY, gates nothing
    section: int | None = None      # dossier section number, from section_hdr chunking
    confidence: str = "ungraded"    # ConfidenceGrade, from section 12
    retrieved_at: str = ""          # ISO8601; when this chunk was read out of the store
    source_type: str = "corpus"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class WebEvidence:
    """One external search result. NO person_id field exists here, and none may be added --
    see the module docstring. Web claims are TOPIC claims; person claims come from dossiers
    only (census open-call 2, owner-confirmed)."""

    url: str
    title: str
    snippet: str
    query: str                      # the search string actually issued; fixtures key on it
    published: str | None = None    # publisher's date if given; often absent
    retrieved_at: str = ""          # ISO8601; web evidence is only true as-of this moment
    source_type: str = "web"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class UserStatedEvidence:
    """A fact the ASKER supplied in conversation (D-036, census C5: "we talked about how we
    can collaborate…"). Exists in no dossier and never will: citable in drafts and answers
    ("as you mentioned"), never dossier-grade, never written back to any corpus. Like
    WebEvidence it has NO person_id field -- a user's account of a meeting must not become a
    claim in a member's mouth."""

    text: str                       # the asker's statement, as close to verbatim as the turn allows
    turn: int                       # which turn of the thread supplied it; provenance for traces
    thread_id: str = ""             # session it was stated in; empty only in unit tests
    source_type: str = "user_stated"

    def to_dict(self) -> dict:
        return asdict(self)


EvidenceItem = CorpusEvidence | WebEvidence | UserStatedEvidence


# --- Session artifacts (D-036) --------------------------------------------------------------


@dataclass(frozen=True)
class ResultSetArtifact:
    """A ranked people-list a prior turn produced, persisted so later turns can operate ON it:
    "who else amongst the people you suggested…" (census C1), counterfactual re-rank (C8),
    "bring them up again" (C7 -- artifacts outlive the session; the session store persists
    them with the thread). person_ids are ORDERED as shown to the user, because ordinals
    ("the first person", "the 2nd one") resolve against display order and nothing else."""

    artifact_id: str                # unique within the store; tools accept it as an operand
    criterion: str                  # the discovery criterion that produced the list
    person_ids: tuple[str, ...]     # display order; tuple so the artifact is genuinely frozen
    why: tuple[str, ...] = ()       # per-person "why them" one-liners, same order, cited+gated
    turn: int = 0                   # turn that produced it
    thread_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation in a turn's trajectory, recorded on the envelope so harness law
    (bypass tripwire, step budget) and trajectory evals read the SAME record the traces show.
    args are the validated post-resolver values, not the model's raw proposal."""

    tool: str
    args: dict
    ok: bool = True
    error: str = ""                 # structured error string when ok is False

    def to_dict(self) -> dict:
        return asdict(self)


# Tools that count as retrieval for the bypass tripwire (D-013's logic, harness law): a turn
# whose declared outcome is "answer" with none of these in its trajectory is a hard fail.
RETRIEVAL_TOOLS: frozenset[str] = frozenset({"find_members", "get_person_evidence", "web_search"})


# --- Resolution ----------------------------------------------------------------------------


@dataclass
class Resolution:
    """Output of the deterministic name-spot (D-028). The ONLY producer of person_ids: the
    stage-2 router LLM may nominate a mention string, but only this lookup appoints an id."""

    status: str                                     # ResolutionStatus
    person_ids: list[str] = field(default_factory=list)
    # The query referred to the asker ("my profile", "help me"). NOT a resolution: the asker is
    # authenticated, so their id is already on the envelope and never needs looking up. This
    # flag says the asker's own dossier is relevant CONTEXT -- which is a different thing from
    # the query being ABOUT them, and keeping them out of person_ids is what stops every
    # "who can help me..." aggregate from looking like a question about the asker.
    self_reference: bool = False
    # mention -> candidate ids, populated when status == "ambiguous" -- a bare first name or
    # surname shared by two subjects. Never auto-resolved: >1 candidate means clarify (D-016).
    candidates: dict[str, list[str]] = field(default_factory=dict)
    # Names spotted in the query that matched no member. Covers BOTH someone outside the corpus
    # entirely and someone merely mentioned inside a dossier (a colleague, a client). The
    # resolver deliberately does not distinguish them: status `not_found` means "not a member",
    # never "refuse". Retrieval still runs, and the refuse-if-absent contract decides -- if the
    # corpus mentions them, synthesis answers honestly from that mention; if not, it refuses.
    # That general rule covers every name in every dossier, which no index of a chosen subset
    # could (D-034 amended).
    unresolved: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --- The envelope --------------------------------------------------------------------------


@dataclass
class Envelope:
    """The per-turn record the loop, tools, harness law, and evals all share (one object so
    the trajectory eval and the Phase-4 citation gate have a single thing to read); the
    strictness lives one level down, in the evidence."""

    query: str
    asker: str                                      # person_id of the authenticated member
    thread_id: str = ""                             # session key; artifacts + follow-ups live here

    # The self-contained form of the user's ask for THIS turn, produced by the coordinator
    # reading the thread ("does he..." -> "does <person> have China experience"). Empty until
    # set; tools always receive the resolved form when present.
    resolved_query: str = ""

    resolution: Resolution | None = None
    # flow/scope/flow_source/lane are EVAL-SIDE since D-036: populated on gold replays and by
    # labeling tooling, never by product control flow (see module docstring).
    flow: str | None = None                         # Flow, expected-behavior metadata
    scope: str | None = None                        # Scope; only topic flow consults it
    flow_source: str | None = None                  # FlowSource; label provenance
    lane: str | None = None                         # Lane, DERIVED from (flow, scope)

    evidence: list[EvidenceItem] = field(default_factory=list)

    # The turn's trajectory: every tool invocation in order, harness-recorded (D-036).
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Ranked-list artifacts this turn produced (find_members results as shown to the user);
    # the session store persists them so later turns can pass them back as operands.
    artifacts: list[ResultSetArtifact] = field(default_factory=list)

    response_mode: str | None = None                # ResponseMode -- DECLARED by the coordinator
    response: str = ""
    # doc_ids / urls the response actually cited. The gap between this and `evidence` is the
    # residue signal the always-both TUNABLE reads, and the per-source false-refusal metric.
    cited: list[str] = field(default_factory=list)

    schema_version: str = HANDOFF_SCHEMA_VERSION
    evidence_schema_version: str = EVIDENCE_SCHEMA_VERSION

    def set_flow(self, flow: str, source: str, scope: str = "open") -> None:
        """Record a gold row's two labels and derive the expected lane in one step, so the
        three can never drift apart in an eval artifact (eval-side since D-036)."""
        self.flow = flow
        self.scope = scope
        self.flow_source = source
        self.lane = derive_lane(flow, scope)

    def record_tool_call(self, call: ToolCall) -> None:
        self.tool_calls.append(call)

    def retrieval_calls(self) -> list[ToolCall]:
        return [c for c in self.tool_calls if c.tool in RETRIEVAL_TOOLS and c.ok]

    def corpus_evidence(self) -> list[CorpusEvidence]:
        return [e for e in self.evidence if isinstance(e, CorpusEvidence)]

    def web_evidence(self) -> list[WebEvidence]:
        return [e for e in self.evidence if isinstance(e, WebEvidence)]

    def user_stated_evidence(self) -> list[UserStatedEvidence]:
        return [e for e in self.evidence if isinstance(e, UserStatedEvidence)]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = [e.to_dict() for e in self.evidence]
        return d


# --- Validation ----------------------------------------------------------------------------


def validate(env: Envelope) -> None:
    """Raise on any envelope a correct run cannot produce. Called at the seams (after routing,
    after retrieval, after synthesis) so a contract break surfaces where it happened rather
    than as a puzzling metric three stages later."""
    if env.schema_version != HANDOFF_SCHEMA_VERSION:
        raise ValueError(f"envelope schema {env.schema_version} != {HANDOFF_SCHEMA_VERSION}")
    if env.evidence_schema_version != EVIDENCE_SCHEMA_VERSION:
        raise ValueError(
            f"evidence schema {env.evidence_schema_version} != {EVIDENCE_SCHEMA_VERSION}"
        )
    if not env.asker:
        raise ValueError("asker is required: every query is asked by an authenticated member")

    if env.flow is not None:
        if env.scope not in TOPIC_SCOPE_TO_LANE:
            raise ValueError(
                f"scope {env.scope!r} must be one of {sorted(TOPIC_SCOPE_TO_LANE)}; the router "
                "emits it on every query even though only topic flow consults it"
            )
        if env.lane != derive_lane(env.flow, env.scope):
            raise ValueError(
                f"lane {env.lane!r} is not the derived lane for flow {env.flow!r} / scope "
                f"{env.scope!r} ({derive_lane(env.flow, env.scope)!r}); lane is derived, "
                "never set by hand"
            )
        if env.flow_source not in ("rules", "llm"):
            raise ValueError("flow_source must record which router stage decided the label")
        if env.response_mode is not None and env.response_mode not in LEGAL_MODES[env.flow]:
            raise ValueError(
                f"flow {env.flow!r} cannot end in mode {env.response_mode!r}; "
                f"legal: {sorted(LEGAL_MODES[env.flow])}"
            )

    # The source-scoping invariant, checked as well as typed. The type makes person-attributed
    # web evidence unconstructable; this catches the other direction -- web evidence appearing
    # where no web search should have run (eval-side: lane is set on gold replays only).
    if env.lane in ("corpus", "neither") and env.web_evidence():
        raise ValueError(f"lane {env.lane!r} carries web evidence; no web search should have run")
    if env.lane == "neither" and (env.corpus_evidence() or env.web_evidence()):
        raise ValueError("lane 'neither' carries retrieved evidence; no retrieval should have run")

    # Clarify is a resolution outcome, not a synthesis choice: it may only follow ambiguity.
    if env.response_mode == "clarify":
        if env.resolution is None or env.resolution.status != "ambiguous":
            raise ValueError("clarify requires resolution.status == 'ambiguous' (G-001/D-016)")


def answered_without_retrieval(env: Envelope) -> bool:
    """The bypass predicate (D-013's logic, D-036 harness law): declared outcome is "answer"
    yet no successful retrieval tool ran this turn. A pure function here so the harness
    tripwire, trajectory evals, and Phase-5 monitoring all read the SAME definition -- bypass
    is a TRAJECTORY property, not a content property. clarify/refuse/redirect are exempt by
    construction: they answer nothing."""
    return env.response_mode == "answer" and not env.retrieval_calls()


# --- Flow-label schema (EVAL TOOLING ONLY since D-036) ---------------------------------------

# Was the D-026 router's stage-2 output schema; the router died with D-036. Retained because
# gold expansion still LLM-drafts flow/scope labels for review (D-012 discipline), and a
# schema-forced closed label set is how a drafting LLM is kept from inventing a flow.
# Nominate-vs-appoint (D-028) survives at the TOOL layer: the coordinator may propose mention
# strings, but only the resolver's lookup appoints person_ids.
FLOW_LABEL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "flow": {"type": "string", "enum": list(FLOW_TO_LANE)},
        "scope": {
            "type": "string",
            "enum": list(TOPIC_SCOPE_TO_LANE),
            "description": (
                "'network' if the question scopes itself to this network of members by its own "
                "wording (in the network / our members / this group / here); 'open' otherwise."
            ),
        },
        "mentions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Person or company mentions to look up. Proposals only -- never ids.",
        },
        "rationale": {"type": "string", "description": "One sentence, for trace review."},
    },
    "required": ["flow", "scope", "mentions", "rationale"],
    "additionalProperties": False,
}


if __name__ == "__main__":
    # Self-check: the invariants this design exists for. Run `python src/contracts.py`.
    env = Envelope(query="How will Trump's tariffs affect my business?", asker="p042")
    env.set_flow("topic", source="llm", scope="open")  # eval-side labels on a gold replay
    env.record_tool_call(ToolCall(tool="get_person_evidence",
                                  args={"query": "industry, markets", "person_ids": ["p042"]}))
    env.record_tool_call(ToolCall(tool="web_search",
                                  args={"query": "2026 US tariffs electronics imports"}))
    env.evidence = [
        CorpusEvidence(
            person_id="p042", doc_id="Jane Rivera", chunk_id="c1", text="...", score=0.71,
            section=5, confidence="medium", retrieved_at="2026-08-01T00:00:00Z",
        ),
        WebEvidence(
            url="https://example.com/tariffs", title="New tariff schedule", snippet="...",
            query="2026 US tariffs electronics imports", published="2026-07-28",
            retrieved_at="2026-08-01T00:00:00Z",
        ),
    ]
    env.response_mode = "answer"
    validate(env)
    print(f"flow={env.flow} -> lane={env.lane} (derived, eval-side)")
    print(f"corpus items: {len(env.corpus_evidence())}  web items: {len(env.web_evidence())}")
    assert "person_id" not in env.web_evidence()[0].to_dict(), "invariant broken"
    print("invariant holds: no person_id exists on web evidence")
    assert not answered_without_retrieval(env), "retrieval ran; bypass must not fire"

    # The bypass tripwire fires on a declared answer with no retrieval in the trajectory.
    bare = Envelope(query="What is Dana Whitfield's revenue?", asker="p042")
    bare.response_mode = "answer"
    assert answered_without_retrieval(bare), "bypass predicate must fire on tool-less answers"
    bare.response_mode = "clarify"  # clarify/refuse/redirect answer nothing -> exempt
    assert not answered_without_retrieval(bare)
    print("bypass predicate: fires on tool-less answers, exempts clarify/refuse/redirect")

    # C5: user-stated facts are typed evidence with no person_id to misattribute.
    stated = UserStatedEvidence(text="we discussed collaborating on semiconductors", turn=1,
                                thread_id="t-1")
    assert "person_id" not in stated.to_dict()
    print("user_stated evidence carries provenance, no person_id")

    # C1/C8: a prior turn's ranked list is a frozen artifact; ordinals resolve display order.
    art = ResultSetArtifact(artifact_id="a1", criterion="pharmaceuticals",
                            person_ids=("p007", "p101", "p055"), turn=2, thread_id="t-1")
    assert art.person_ids[0] == "p007", "'the first person' is display order, nothing else"
    print("result-set artifact frozen; ordinal operand resolves against display order")

    # G-005 / D-030: the same flow, scoped to the network by the user's own words, must not
    # touch web evidence -- now an eval expectation read via derive_lane on gold rows.
    scoped = Envelope(
        query="What do the fintech leaders in the network think about AI?", asker="p042"
    )
    scoped.set_flow("topic", source="rules", scope="network")
    validate(scoped)
    print(f"same flow, scope=network -> expected lane={scoped.lane} (web evidence is a violation)")
    assert derive_lane("person_fact", "network") == "corpus", "scope must not leak past topic"
    print("scope is consulted for topic flow only")
