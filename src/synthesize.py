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


def strip_display_markers(answer: str) -> str:
    """Render for the user: internal-only citation markers come out. Applied at the display
    boundary ONLY -- never before the citation parser, the judge, or the gate, all of which must
    score the answer as written.

    Removes any whitespace preceding the marker too, otherwise "...in 2019 [self]." renders as
    "...in 2019 ." -- a space before the full stop on every asker claim.
    """
    return " ".join(re.sub(r"\s*" + re.escape(SELF_CITATION_MARKER), "", answer).split())


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
