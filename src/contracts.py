"""Phase 3 handoff contracts (D-029): the shapes each workflow step hands to the next.

These live in a plain module with NO LangGraph import, per D-025 mitigation (a) -- thin graph,
thick pure functions. Nothing a metric touches may depend on the framework, so rip-out stays
cheap and every shape here is unit-testable on its own.

TWO versioned shapes (D-029 chose "one envelope + hard-typed evidence" over one loose dict and
over per-seam contracts):

  HANDOFF_SCHEMA_VERSION   -- the envelope every node reads and writes
  EVIDENCE_SCHEMA_VERSION  -- the evidence items, which come in two STRICT kinds

The evidence split is the load-bearing part. CorpusEvidence carries person_id; WebEvidence has
no person_id field AT ALL -- not empty, absent. A web result therefore cannot be attributed to
a member no matter what later code does (census open-call 2, the source-scoping invariant).
Putting a Reuters claim in a real semi-public person's mouth is the exact reputational failure
D-009 exists to prevent, and an absent field is the one thing a refactor cannot quietly undo.

LANE IS DERIVED, NOT CLASSIFIED (D-029, amended by D-030). The router's LLM step supplies two
SIGNALS -- flow (what shape of work) and scope (network or open) -- and code supplies the
VERDICT: lane = derive_lane(flow, scope). Lane is never predicted, because a predicted lane
invites labels the flow table cannot produce ("lane: web" on a person_fact query) and would
report an accuracy that is 100% by construction for eight flows out of nine. Lane stays a
recorded field: traces show it, evals score it, Phase 5 monitors it. Nothing guesses it.

Both versions are snapshotted into run config exactly like PROMPT_CONTRACT_VERSION and
RUBRIC_VERSION (D-021): a shape change re-defines the system under test.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

# Bump when the envelope's fields change; runs across versions are not comparable.
HANDOFF_SCHEMA_VERSION = "v1"

# Bump when an evidence kind gains/loses/renames a field. Versioned SEPARATELY from the
# envelope so adding an envelope field does not invalidate evidence-level comparability.
EVIDENCE_SCHEMA_VERSION = "v1"

# --- The three enums the eval scores -------------------------------------------------------

# What shape of work the question needs. This is the ONLY label the router classifies
# (D-026 hybrid cascade: stage-1 rules, stage-2 LLM on the residue).
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

# How the flow label was produced -- D-026 commitment: every trace records WHICH STAGE decided,
# so stage-1-alone coverage is reportable whether or not the LLM stage earns its place.
FlowSource = Literal["rules", "llm"]

# What name resolution concluded. Drives behaviour independently of flow: a not_found person
# short-circuits regardless of what shape of work the question asked for.
ResolutionStatus = Literal["resolved", "ambiguous", "not_found", "none_named"]

# Dossier section 12 grades its own facts; a low-confidence fact must not be asserted flatly.
# "ungraded" = the section stated no grade for this span, which is NOT the same as low.
ConfidenceGrade = Literal["high", "medium", "low", "ungraded"]

# --- Lane derivation -----------------------------------------------------------------------

# Everything with a resolved person is corpus by construction. Topic flow always fans out to
# BOTH corpus and web in parallel (D-029): a similarity score cannot tell "the corpus has this"
# from "the corpus does not" -- measured on run 20260730T022029Z-5d5bcf2, where answerable
# questions scored top-1 0.586-0.849 and unanswerable ones 0.642-0.831, near-total overlap.
# So the policy stops guessing and lets synthesis ground per-source in whichever half is real.
# TUNABLE(always-both on topic flow; symptom wrong: web evidence goes uncited on >70% of topic
#         queries -> collapse to a predicted lane and let the router guess; uncited on <30%
#         -> the corpus half is the dead weight. Revisit at the first routed run.)
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

# Which response modes each flow may legally end in. The eval reads this table too: a
# (flow, mode) pair outside it is an instrument bug, not a model failure.
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
    """The whole lane 'classifier'. Kept a function so traces and evals call the same code.

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


EvidenceItem = CorpusEvidence | WebEvidence


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
    # Names spotted in the query that matched nothing anywhere -- people outside the corpus
    # entirely (a public figure who was never a member).
    unresolved: list[str] = field(default_factory=list)
    # Named inside a member's dossier (Key Personnel of their company) but holding no dossier
    # of their own: mention -> ids of the dossiers naming them. NOT person_ids -- these people
    # are not members, and the honest answer names where they appear rather than refusing
    # outright. Kept distinct from `unresolved` because "in a profile but not a member" and
    # "not in the corpus at all" deserve different answers.
    non_subject: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# --- The envelope --------------------------------------------------------------------------


@dataclass
class Envelope:
    """What every node reads and writes. One object so the routed eval and the Phase-4 citation
    gate have a single thing to read; the strictness lives one level down, in the evidence."""

    query: str
    asker: str                                      # person_id of the authenticated member
    thread_id: str = ""                             # session key for follow-ups / two-turn clarify

    # Set once session state has substituted pronouns and ellipsis ("what has SHE said about
    # AI?"). Empty until then; retrieval always uses the resolved form when present.
    resolved_query: str = ""

    resolution: Resolution | None = None
    flow: str | None = None                         # Flow, signal 1 from the router
    scope: str | None = None                        # Scope, signal 2; only topic flow uses it
    flow_source: str | None = None                  # FlowSource; D-026 stage attribution
    lane: str | None = None                         # Lane, DERIVED from (flow, scope)

    evidence: list[EvidenceItem] = field(default_factory=list)

    response_mode: str | None = None                # ResponseMode
    response: str = ""
    # doc_ids / urls the response actually cited. The gap between this and `evidence` is the
    # residue signal the always-both TUNABLE reads, and the per-source false-refusal metric.
    cited: list[str] = field(default_factory=list)

    schema_version: str = HANDOFF_SCHEMA_VERSION
    evidence_schema_version: str = EVIDENCE_SCHEMA_VERSION

    def set_flow(self, flow: str, source: str, scope: str = "open") -> None:
        """Record the router's two signals and derive the lane in one step, so the three can
        never drift apart in a trace."""
        self.flow = flow
        self.scope = scope
        self.flow_source = source
        self.lane = derive_lane(flow, scope)

    def corpus_evidence(self) -> list[CorpusEvidence]:
        return [e for e in self.evidence if isinstance(e, CorpusEvidence)]

    def web_evidence(self) -> list[WebEvidence]:
        return [e for e in self.evidence if isinstance(e, WebEvidence)]

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
    # in a lane that never ran a web search.
    if env.lane in ("corpus", "neither") and env.web_evidence():
        raise ValueError(f"lane {env.lane!r} carries web evidence; no web search should have run")
    if env.lane == "neither" and env.evidence:
        raise ValueError("lane 'neither' carries evidence; no retrieval should have run")

    # Clarify is a resolution outcome, not a synthesis choice: it may only follow ambiguity.
    if env.response_mode == "clarify":
        if env.resolution is None or env.resolution.status != "ambiguous":
            raise ValueError("clarify requires resolution.status == 'ambiguous' (G-001/D-016)")


# --- The router's stage-2 output schema (D-026) ---------------------------------------------

# Hand-written JSON schema, same idiom as the judge's lanes (eval/judge.py): the LLM stage is
# schema-FORCED to a closed label set, so it can never invent a flow. Nominate-vs-appoint
# (D-028): it may propose mention strings, but only the resolver's lookup appoints person_ids.
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
    # Self-check: the invariant this whole design exists for. Run it to see the failure the
    # types prevent (`python src/contracts.py`).
    env = Envelope(query="How will Trump's tariffs affect my business?", asker="p042")
    env.set_flow("topic", source="llm", scope="open")
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
    print(f"flow={env.flow} -> lane={env.lane} (derived), decided by {env.flow_source}")
    print(f"corpus items: {len(env.corpus_evidence())}  web items: {len(env.web_evidence())}")
    print("web evidence fields:", sorted(env.web_evidence()[0].to_dict()))
    assert "person_id" not in env.web_evidence()[0].to_dict(), "invariant broken"
    print("invariant holds: no person_id exists on web evidence")

    # G-005 / D-030: the same flow, scoped to the network by the user's own words, never
    # reaches the web at all -- so no web commentary can be read as the network's view.
    scoped = Envelope(
        query="What do the fintech leaders in the network think about AI?", asker="p042"
    )
    scoped.set_flow("topic", source="rules", scope="network")
    validate(scoped)
    print(f"same flow, scope=network -> lane={scoped.lane} (no web search runs)")
    assert derive_lane("person_fact", "network") == "corpus", "scope must not leak past topic"
    print("scope is consulted for topic flow only")
