"""Synthesis contract synth-v1 (D-032): the ONE generation step in the system.

Replaces f6-v1 on the product path. The workers return evidence, never prose (D-029), so this
is the only place text is written -- which makes this contract half the measurement instrument,
exactly as f6-v1 was, and versioned for the same reason.

STRUCTURE: core + per-flow section. The CORE holds the rules the metrics measure -- grounding,
citations, the refusal string, confidence handling, voice. Each FLOW SECTION holds only the
shape of that flow's output. The split is about what the model SEES: an edit to the composition
section leaves the bytes sent for a person_fact question byte-identical, so that question's
groundedness stays comparable across the edit. Under a single blended prompt every edit changes
every question's input and re-baselines everything.

    # TUNABLE(core+section split; symptom it was ceremony: once all nine sections are written,
    #         if they average under ~5 lines apiece the shared core dominates and a single
    #         prompt was the right call -- collapse back and retire the section versions.)

PROMPTS LIVE IN FILES, NOT IN CODE (D-032). prompts/synth/*.md, loaded at import and hashed
into run config. Rationale is D-021's own precedent: sha256(questions.jsonl) went into config so
an edited gold quote could never silently change what two runs scored. Prompt text is the same
hazard and gets the same treatment -- comparability becomes provable rather than asserted. It
also keeps prose out of Python string literals, where escaped quotes and implicit concatenation
are an error surface on text whose exact bytes ARE the instrument.

THE ONE STRING THAT STAYS IN CODE is REFUSAL_STRING, interpolated into the core template. It is
the same discipline generate.py used: the deterministic detector must check the SAME string the
prompt demands, so the two can never drift apart. The template carries a placeholder; config
records the assembled hash, so the substituted text is still pinned.

MIGRATION COST, accepted eyes-open at D-029: f6-v1 leaves the product path, so generation
metrics re-baseline at this new surface (a phase-boundary re-baseline, not a regression), and
the D-010 rubric re-anchors -> calibrate.py re-runs per D-022.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

SYNTH_CONTRACT_VERSION = "synth-v1"

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts" / "synth"

# Per-section versions. A section bump invalidates comparability for THAT flow's questions only.
# The value is also the filename suffix: flow-<name>-<version>.md.
FLOW_SECTION_VERSIONS: dict[str, str] = {
    "comparison": "v1",
    "topic": "v1",
}

CORE_FILE = PROMPT_DIR / "core-v1.md"

# The exact sentence that marks a WHOLE-ANSWER refusal -- both evidence kinds came back
# unsupportive. D-019 re-anchor (G-003): f6-v1's string named "the provided reports", which the
# coordinator voice forbids (memo section 5b), so the string changes and the deterministic
# detector re-points at this constant. Per-source silence emits NO string and is measured a
# different way (retrieved-but-uncited evidence, D-029(c) consequence 3) -- so the presence of
# this sentence still means exactly one thing, which is what keeps refusal_exact deterministic.
REFUSAL_STRING = "I don't have information on that."

# Asker-fact citations (D-032, owner's call: exempt WITH A MARKER). Citing a member's own profile
# back to them by name reads absurd in conversation ("your AI startup in Hyderabad [profile: Joe
# Zhou]"), so rule 6 has the model tag those claims [self] instead.
#
# Why a marker rather than plain omission -- this is the whole point of the choice. An omitted
# citation would make asker evidence look RETRIEVED-BUT-UNCITED, which is exactly the residue
# signal that means "synthesis dropped good evidence" (D-029(c) consequence 3). Silent omission
# would therefore pollute the per-source false-refusal metric with false positives. With a marker
# the pipeline stays uniform -- every claim is cited internally, the citation parser and the
# Phase-4 gate see it -- and ONLY the display layer drops it.
SELF_CITATION_MARKER = "[self]"

# User-stated facts (contracts.UserStatedEvidence, census C5) get the SAME treatment for the
# SAME reason, so this is D-032's decided principle applied rather than a new call: an omitted
# citation would make the stated fact look retrieved-but-uncited, which is the residue signal
# meaning "synthesis dropped good evidence". A marker keeps every claim traceable internally
# and drops out only at the display boundary. Distinct from [self] on purpose -- [self] is a
# fact from the asker's DOSSIER, [stated] is a fact the asker SAID, and collapsing the two
# would throw away the provenance UserStatedEvidence exists to carry.
STATED_CITATION_MARKER = "[stated]"

DISPLAY_MARKERS = (SELF_CITATION_MARKER, STATED_CITATION_MARKER)


def strip_display_markers(answer: str) -> str:
    """Render for the user: internal-only citation markers come out. Applied at the display
    boundary ONLY -- never before the citation parser, the judge, or the gate, all of which must
    score the answer as written.

    Removes horizontal whitespace preceding a marker too, otherwise "...in 2019 [self]."
    renders as "...in 2019 ." -- a space before the full stop on every asker claim.

    NEWLINES SURVIVE. The first version collapsed all whitespace, which was invisible while
    answers were one short paragraph and wrong the moment they were not: the first live
    smoke rendered a structured multi-paragraph brief as a single wall of text. Only runs of
    spaces and tabs WITHIN a line are collapsed.
    """
    pattern = "|".join(r"[ \t]*" + re.escape(m) for m in DISPLAY_MARKERS)
    cleaned = re.sub(pattern, "", answer)
    return "\n".join(" ".join(line.split()) for line in cleaned.splitlines()).strip()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read(path: Path) -> str:
    """Prompt files are read as UTF-8 with newlines normalized, so a checkout that converts
    line endings cannot change a hash and silently break run comparability on Windows."""
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").strip()


def core_template() -> str:
    """The core contract as authored, placeholder un-substituted."""
    return _read(CORE_FILE)


def flow_file(flow: str) -> Path:
    try:
        version = FLOW_SECTION_VERSIONS[flow]
    except KeyError:
        raise ValueError(
            f"no synth-v1 section for flow {flow!r}; have {sorted(FLOW_SECTION_VERSIONS)}"
        ) from None
    return PROMPT_DIR / f"flow-{flow}-{version}.md"


def build_system(flow: str) -> str:
    """Assemble the system prompt for one flow: core (refusal string substituted) + section.

    Raises on a flow with no section rather than silently sending the bare core -- a missing
    section would be an invisible contract hole, and invisible holes are what the version
    discipline exists to prevent.
    """
    core = core_template().format(refusal_string=REFUSAL_STRING)
    section = _read(flow_file(flow))
    return f"{core}\n\n## For this question\n\n{section}"


def contract_versions(flow: str) -> dict[str, str]:
    """What the run config snapshots (D-021), so a moved metric is attributable.

    Three hashes, each answering a different question:
      core_sha256    -- did the shared rules change?
      section_sha256 -- did THIS flow's output shape change?
      system_sha256  -- did the exact bytes the model saw change? (the one that settles
                        comparability; it also covers the refusal-string substitution)
    """
    return {
        "synth_contract_version": SYNTH_CONTRACT_VERSION,
        "flow": flow,
        "flow_section_version": FLOW_SECTION_VERSIONS[flow],
        "refusal_string": REFUSAL_STRING,
        "core_sha256": _sha256(core_template()),
        "section_sha256": _sha256(_read(flow_file(flow))),
        "system_sha256": _sha256(build_system(flow)),
    }


# --- The answer path (D-032 wired here at Step 5) --------------------------------------------

# WHICH PROMPT THE ANSWER IS WRITTEN UNDER -- OPEN OWNER CALL, provisionally "core_only".
# Nothing has been measured under either setting; the Step-5 re-baseline runs AFTER this is
# settled, so no number depends on the provisional value.
#
# D-032(a) chose core + per-flow section, with the D-024 router supplying `flow`. D-036 deleted
# the router, so nothing selects a section at runtime any more -- and only 2 of the 9 sections
# were ever written. That reopens D-032's own PRE-REGISTERED TELL ("once all nine sections
# exist, if they average under ~5 lines apiece the shared core dominates and a single prompt
# was the right call -- collapse back and retire the section versions").
#
#   "core_only" -- the 13-rule core IS the contract. Output shape is left to the model, which
#                  under the whole-thread answer call reads the conversation and rule 12
#                  ("match length to what was asked"). Retires FLOW_SECTION_VERSIONS.
#   "sections"  -- keep D-032(a) as written. Costs the 7 missing section files PLUS a selector,
#                  since no router exists: either a shape enum declared on `respond`, or a
#                  shape derived from the turn's recorded trajectory.
ANSWER_PROMPT_MODE = "core_only"

# TUNABLE(2048 tokens for the answer: the last baseline's answers averaged ~75 output tokens
#         and the longest was well under 500, so this is ~4x headroom for the aggregate shape
#         (5 members + why-them lines). Symptom wrong: answers truncated mid-sentence, visible
#         as stop_reason "max_tokens" on the synthesize span -> raise.)
ANSWER_MAX_TOKENS = 2048


def answer_system(flow: str | None = None) -> str:
    """The system prompt the ANSWER is written under -- the pinned instrument, kept separate
    from the coordinator's prompt on purpose. A tool-description edit changes what the
    coordinator sees and must NOT change the bytes an answer was generated under, or every
    generation metric re-baselines on a change that has nothing to do with how answers are
    written (D-021's own argument for hashing questions.jsonl, one layer over)."""
    core = core_template().format(refusal_string=REFUSAL_STRING)
    if ANSWER_PROMPT_MODE == "core_only":
        return core
    if ANSWER_PROMPT_MODE == "sections":
        if flow is None:
            raise ValueError(
                "ANSWER_PROMPT_MODE='sections' needs a flow label, and D-036 left nothing to "
                "produce one; decide the selector before switching modes"
            )
        return build_system(flow)
    raise ValueError(f"unknown ANSWER_PROMPT_MODE {ANSWER_PROMPT_MODE!r}")


def answer_contract_versions(flow: str | None = None) -> dict[str, str]:
    """What run config snapshots for the answer path (D-021). system_sha256 is the one that
    settles comparability: it covers the core, the refusal-string substitution, and the section
    if one is in play."""
    versions = {
        "synth_contract_version": SYNTH_CONTRACT_VERSION,
        "answer_prompt_mode": ANSWER_PROMPT_MODE,
        "refusal_string": REFUSAL_STRING,
        "core_sha256": _sha256(core_template()),
        "system_sha256": _sha256(answer_system(flow)),
    }
    if ANSWER_PROMPT_MODE == "sections":
        versions["flow"] = flow or ""
        versions["flow_section_version"] = FLOW_SECTION_VERSIONS[flow]
        versions["section_sha256"] = _sha256(_read(flow_file(flow)))
    return versions


def make_answer_writer(provider, on_delta=None, flow: str | None = None):
    """Build the coordinator's `synthesize_fn`: the SEPARATE, pinned, streamed answer call.

    Shape of the call, and why each part is what it is:
      * system   = answer_system() -- pinned bytes, its own version clock (above).
      * messages = the turn's thread VERBATIM, including tool results. The owner's call: this
        is a conversational product, so the writer needs the conversation to do its job. The
        cost is that it also sees the tool scaffolding, which core rule 13 forbids it narrating.
      * tools    = re-sent with tool_choice "none". The tools are NOT for calling -- the wire
        format requires tool definitions whenever the history carries tool_use blocks, and
        "none" is what forbids a second round of calls. (Harmless for caching: tool_choice is
        not part of the cache prefix, so the tools+system prefix still hits.)

    Returns text; the caller owns what it does with the deltas (streaming passthrough, D-037).
    """
    def write(env, messages: list[dict], tools: list[dict]) -> ModelTurnLike:
        return provider.stream_text(
            answer_system(flow), messages, tools=tools,
            max_tokens=ANSWER_MAX_TOKENS, on_delta=on_delta,
        )
    return write


# Typing shim: provider.stream_text returns a provider.ModelTurn, but importing it here would
# make this module depend on the vendor seam for a type alone. The contract is structural:
# .text, .input_tokens, .output_tokens, .stop_reason.
ModelTurnLike = object


if __name__ == "__main__":
    for flow in sorted(FLOW_SECTION_VERSIONS):
        v = contract_versions(flow)
        print("=" * 78)
        print(f"FLOW: {flow}")
        print(f"  core    {v['core_sha256'][:16]}")
        print(f"  section {v['section_sha256'][:16]}  ({v['flow_section_version']})")
        print(f"  system  {v['system_sha256'][:16]}")
        print("=" * 78)
        print(build_system(flow))
        print()

    core_lines = len([ln for ln in core_template().splitlines() if ln.strip()])
    print("-" * 78)
    print(f"core: {core_lines} non-blank lines")
    for flow in sorted(FLOW_SECTION_VERSIONS):
        print(f"section {flow}: {len(_read(flow_file(flow)).split())} words")
