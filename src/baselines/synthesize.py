"""Synthesis contract synth-v1 (D-032): the ONE generation step in the system.

Replaces f6-v1 on the product path. The workers return evidence, never prose (D-029), so this
is the only place text is written -- which makes this contract half the measurement instrument,
exactly as f6-v1 was, and versioned for the same reason.

STRUCTURE: ONE core prompt (D-032(a) revised 2026-10-12). D-032 chose core + per-flow section,
with the router supplying the flow. D-036 deleted the router, only 2 of 9 sections were ever
written (40 and 53 words), and D-032's own pre-registered tell ("if sections average under ~5
lines, collapse back") fired. The two written sections became core rules 16-17. What this gives
up: an edit to any rule now changes the bytes for EVERY question type, so per-type metrics show
WHICH type moved but not which rule moved it.

    # TUNABLE(single prompt; revisit when the re-baseline shows one question type drifting
    #         while the others hold across a core edit -- that is the attribution the
    #         sections bought, and the signal that it is needed back.)

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

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts" / "synth"

CORE_FILE = PROMPT_DIR / "core-v1.md"

# The exact sentence that marks a WHOLE-ANSWER refusal -- both evidence kinds came back
# unsupportive. D-019 re-anchor (G-003): f6-v1's string named "the provided reports", which the
# coordinator voice forbids (memo section 5b), so the string changes and the deterministic
# detector re-points at this constant. Per-source silence emits NO string and is measured a
# different way (retrieved-but-uncited evidence, D-029(c) consequence 3) -- so the presence of
# this sentence still means exactly one thing, which is what keeps refusal_exact deterministic.
from contracts import REFUSAL_STRING  # noqa: E402  (shared; defined once)

# Asker-fact citations (D-032, owner's call: exempt WITH A MARKER). Citing a member's own profile
# back to them by name reads absurd in conversation ("your lending startup [profile: Jane
# Rivera]"), so rule 6 has the model tag those claims [self] instead.
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


# --- The answer path (D-032 wired here at Step 5) --------------------------------------------

# TUNABLE(2048 tokens for the answer: the last baseline's answers averaged ~75 output tokens
#         and the longest was well under 500, so this is ~4x headroom for the aggregate shape
#         (5 members + why-them lines). Symptom wrong: answers truncated mid-sentence, visible
#         as stop_reason "max_tokens" on the synthesize span -> raise.)
ANSWER_MAX_TOKENS = 2048


def answer_system(asker_name: str = "", context: str = "") -> str:
    """The system prompt the ANSWER is written under -- the pinned instrument, kept separate
    from the coordinator's prompt on purpose. A tool-description edit changes what the
    coordinator sees and must NOT change the bytes an answer was generated under, or every
    generation metric re-baselines on a change that has nothing to do with how answers are
    written (D-021's own argument for hashing questions.jsonl, one layer over).

    The ASKER line rides OUTSIDE the pinned bytes, the same seam coordinator.build_system uses.
    Without it the writer could not apply rules 6, 10, 14 or 15 -- it would not know whose
    profile is [self], whom to greet, or whom it must never speak as. (The first wiring sent
    the core alone; the coordinator knew the asker, the writer did not.)"""
    core = core_template().format(refusal_string=REFUSAL_STRING)
    if not asker_name:
        return core
    session = f"{core}\n\n## Session\nYou are speaking to: {asker_name}"
    # Greeting preload (D-041): the asker's own profile summary and last topic, loaded by SQL.
    # Same unpinned seam as the asker line -- config pins the core, not this.
    return f"{session}\n\n{context}" if context else session


def answer_contract_versions() -> dict[str, str]:
    """What run config snapshots for the answer path (D-021). core_sha256 settles
    comparability: the assembled prompt differs per asker by the session line only."""
    return {
        "synth_contract_version": SYNTH_CONTRACT_VERSION,
        "refusal_string": REFUSAL_STRING,
        "core_sha256": _sha256(core_template()),
        "pinned_sha256": _sha256(answer_system()),
    }


def make_answer_writer(provider, name_of: dict[str, str], on_delta=None):
    """Build the coordinator's `synthesize_fn`: the SEPARATE, pinned, streamed answer call.
    Serves both `answer` and `redirect` -- every model-written sentence a user sees goes
    through this one measured contract.

    Shape of the call, and why each part is what it is:
      * system   = answer_system(asker name) -- pinned core plus the per-session asker line.
      * messages = the turn's thread VERBATIM, including tool results. The owner's call: this
        is a conversational product, so the writer needs the conversation to do its job. The
        cost is that it also sees the tool scaffolding, which core rule 13 forbids it narrating.
      * tools    = re-sent with tool_choice "none". The tools are NOT for calling -- the wire
        format requires tool definitions whenever the history carries tool_use blocks, and
        "none" is what forbids a second round of calls. (Harmless for caching: tool_choice is
        not part of the cache prefix, so the tools+system prefix still hits.)

    name_of maps person_id -> display name (Resolver.name_of).
    Returns text; the caller owns what it does with the deltas (streaming passthrough, D-037).
    """
    def write(env, messages: list[dict], tools: list[dict], context: str = "") -> ModelTurnLike:
        return provider.stream_text(
            answer_system(name_of.get(env.asker, ""), context), messages, tools=tools,
            max_tokens=ANSWER_MAX_TOKENS, on_delta=on_delta,
        )
    return write


# Typing shim: provider.stream_text returns a provider.ModelTurn, but importing it here would
# make this module depend on the vendor seam for a type alone. The contract is structural:
# .text, .input_tokens, .output_tokens, .stop_reason.
ModelTurnLike = object


if __name__ == "__main__":
    v = answer_contract_versions()
    print(f"core   {v['core_sha256'][:16]}")
    print(f"pinned {v['pinned_sha256'][:16]}")
    print("=" * 78)
    print(answer_system("Jane Rivera"))
    rules = [ln for ln in core_template().splitlines() if ln[:1].isdigit()]
    print("-" * 78)
    print(f"core: {len(rules)} numbered rules")
