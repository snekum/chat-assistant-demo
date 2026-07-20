#!/usr/bin/env python3
"""Local pseudonymizer for Deep Research Reports.

Reads reports from data/raw/*.md, replaces the subject (and discovered orgs /
other people) with consistent fakes, and writes clean copies to data/clean/.
The real->fake map is persisted to data/mapping/map.json so the SAME real
entity maps to the SAME fake entity across every file and every run.

WHAT THIS GUARANTEES (enforced by the verification pass at the end):
  - No value that is IN THE MAP (subject name/surname/first, discovered orgs,
    manual overrides) and no derived domain fragment appears anywhere in
    data/clean/ or in any clean filename. If one does, the run HARD-FAILS.

WHAT THIS DOES NOT GUARANTEE:
  - It cannot prove "no PII anywhere" — only "nothing we mapped leaked."
    Completeness of PII removal == quality of entity discovery (spaCy/heuristic)
    plus the manual override loop. Unmapped entities pass verification silently,
    which is why review_candidates.json and overrides.json exist.

Design (see DECISIONS.md D-001..D-007):
  - Subject is mapped DETERMINISTICALLY from the filename (bulletproof core).
  - spaCy en_core_web_sm is a DISCOVERY aid only, used if it loads; otherwise a
    pure-Python capitalized-phrase heuristic. NER never silently redacts.
  - Public orgs are KEPT via an allowlist (D-001). Locations are left real (D-006).
  - Bare-surname replacement is per-file / subject-scoped (D-004).
  - Domains are fuzzed from mapped org roots + subject surname (D-007).

Developer note: the AUTHOR of this script is intentionally blind to data/raw
(writing against PII not cleared to view). Only this script reads raw content;
it is never echoed. The verification pass may print a leaked value on FAILURE —
that is the intended iterate-and-override signal for the data owner.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean"
MAP_DIR = ROOT / "data" / "mapping"
MAP_PATH = MAP_DIR / "map.json"
REVIEW_PATH = MAP_DIR / "review_candidates.json"
OVERRIDES_PATH = MAP_DIR / "overrides.json"  # optional manual leak fixes (real->fake)

# --------------------------------------------------------------------------- #
# TUNABLEs
# --------------------------------------------------------------------------- #
# TUNABLE(deterministic/reproducible fake assignments so reruns are stable;
#         revisit when the map is regenerated from scratch and you want fresh names)
SEED = 20260705

# Minimum length for a BARE first/last name token to be replaced on its own.
# TUNABLE(shorter tokens are usually initials/common words -> false positives;
#         revisit if a real 2-char surname leaks in verification)
MIN_BARE_TOKEN_LEN = 3

# Public companies / institutions kept per D-001. Matched case-insensitively as
# whole entities; NOT mapped, NOT fuzzed in domains.
# TUNABLE(keep-list of genuinely public, non-identifying entities;
#         revisit when a kept org turns out to be small/regional enough to
#         fingerprint a subject -> remove it so it gets mapped instead)
PUBLIC_ORG_ALLOWLIST = {
    "fiserv", "fis", "jack henry", "jack henry & associates", "d+h", "dh",
    "finastra", "temenos", "ncr", "diebold", "diebold nixdorf", "oracle",
    "microsoft", "google", "amazon", "aws", "ibm", "sap", "salesforce",
    "mastercard", "visa", "american express", "paypal", "stripe", "square",
    "jpmorgan", "jpmorgan chase", "wells fargo", "bank of america", "citibank",
    "citigroup", "goldman sachs", "morgan stanley", "us bank", "pnc",
    "linkedin", "crunchbase", "bloomberg", "reuters", "forbes", "techcrunch",
    "wikipedia", "youtube", "twitter", "x", "facebook", "meta",
}

# Name tokens that are also common English / finance words. These are matched
# FULL-NAME ONLY (never bare) to avoid corrupting text like "Sharpe ratio".
# TUNABLE(precision guard for D-004; revisit by adding any word-like name that
#         verification cannot safely grep, or that mangles ordinary text)
COMMON_WORD_NAMES = {
    "long", "short", "case", "bell", "rice", "bull", "bear", "sharpe", "sharp",
    "goad", "lai", "kane", "young", "moore", "grace", "art", "may", "june",
    "will", "mark", "rich", "frank", "sunny", "hope", "faith", "penny", "drew",
    "king", "law", "cash", "gold", "green", "brown", "white", "black", "stone",
    "wells", "banks", "fields", "rivers", "summers", "winters", "day", "noble",
}

# Opaque redirect hosts to leave alone (D-007 / prompt constraint).
OPAQUE_URL_MARKERS = ("vertexaisearch",)

# --------------------------------------------------------------------------- #
# Fake-value pools (D-002: random from built-in pools, no gender preservation)
# Kept deliberately free of real-world-loaded / finance words to avoid a fake
# accidentally reading like a real entity.
# --------------------------------------------------------------------------- #
FAKE_FIRST = [
    "Marcus", "Elena", "Devin", "Priya", "Oscar", "Nadia", "Felix", "Rosa",
    "Trevor", "Iris", "Damon", "Celia", "Miles", "Tessa", "Rowan", "Vera",
    "Quinn", "Lena", "Bruno", "Mara", "Silas", "Nina", "Cyrus", "Dahlia",
    "Emil", "Freya", "Hugo", "Juno", "Kai", "Livia", "Nolan", "Opal",
    "Reid", "Sonia", "Theo", "Uma", "Vince", "Willa", "Xander", "Yara",
    "Zeke", "Adele", "Boris", "Cleo", "Dorian", "Esme", "Gideon", "Hazel",
]
FAKE_LAST = [
    "Reyes", "Voss", "Calder", "Marsh", "Ellison", "Prieto", "Quint", "Novak",
    "Aldridge", "Brandt", "Cortez", "Dunmore", "Everly", "Fenn", "Gallo",
    "Harmon", "Ingram", "Jarrett", "Kessler", "Larkin", "Mercer", "Ness",
    "Orsini", "Pruett", "Rourke", "Sandoval", "Thorne", "Ulrich", "Vance",
    "Whitlock", "Ackerman", "Beckett", "Corliss", "Dresden", "Eastwood",
    "Farrow", "Grimaldi", "Halloran", "Isley", "Jennings", "Kavanagh",
]
FAKE_ORG_ROOTS = [
    "Vanguard", "Meridian", "Cobalt", "Ironwood", "Northgate", "Silverline",
    "Kestrel", "Halcyon", "Brightpoint", "Anvil", "Cedarhill", "Falcon",
    "Granite", "Harborview", "Juniper", "Lodestar", "Monarch", "Onyx",
    "Pinnacle", "Redwood", "Summit", "Tidewater", "Union", "Westfield",
    "Aspen", "Beacon", "Crestline", "Drayton", "Emberly", "Foundry",
]
# Lowercase 2-3 char suffixes to mirror org quirks like "Paladin fs" (D-002/D-007).
FAKE_ORG_SUFFIXES = ["cs", "rc", "ai", "co", "gp", "ix", "cx", "qs", "vt", "nd"]


# --------------------------------------------------------------------------- #
# Map persistence
# --------------------------------------------------------------------------- #
def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def new_map() -> dict:
    return {
        # real full name -> {first,last,fake_full,fake_first,fake_last}
        "persons": {},
        # real org string -> fake org string
        "orgs": {},
        # set (as list) of every fake string handed out, to prevent collisions
        "used_fakes": [],
    }


class FakeFactory:
    """Hands out unique fakes, seeded for reproducibility."""

    def __init__(self, used: set[str], seed: int):
        self.rng = random.Random(seed)
        self.used = used

    def unique_person(self):
        for _ in range(500):
            first = self.rng.choice(FAKE_FIRST)
            last = self.rng.choice(FAKE_LAST)
            full = f"{first} {last}"
            if full.lower() not in self.used:
                self.used.add(full.lower())
                return first, last, full
        # exhausted -> numeric suffix (D-003)
        n = 1
        while True:
            first = self.rng.choice(FAKE_FIRST)
            last = f"{self.rng.choice(FAKE_LAST)}{n}"
            full = f"{first} {last}"
            if full.lower() not in self.used:
                self.used.add(full.lower())
                return first, last, full
            n += 1

    def unique_org(self, shape_suffix: bool):
        for _ in range(500):
            root = self.rng.choice(FAKE_ORG_ROOTS)
            if shape_suffix:
                org = f"{root} {self.rng.choice(FAKE_ORG_SUFFIXES)}"
            else:
                org = root
            if org.lower() not in self.used:
                self.used.add(org.lower())
                return org
        n = 1
        while True:
            org = f"{self.rng.choice(FAKE_ORG_ROOTS)}{n}"
            if org.lower() not in self.used:
                self.used.add(org.lower())
                return org
            n += 1


# --------------------------------------------------------------------------- #
# Subject parsing (deterministic core — D-005)
# --------------------------------------------------------------------------- #
def parse_subject(filename_stem: str) -> tuple[str, str, str]:
    """'Aaron Silva' -> ('Aaron Silva', 'Aaron', 'Silva'). Last token is surname."""
    full = filename_stem.strip()
    tokens = full.split()
    first = tokens[0]
    last = tokens[-1]
    return full, first, last


# --------------------------------------------------------------------------- #
# Entity discovery (spaCy if available, else heuristic — D-005)
# --------------------------------------------------------------------------- #
def load_nlp():
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        # smoke-test inference in case the DLL block extends here
        nlp("Test Person works at Test Corp.")
        return nlp
    except Exception as exc:  # noqa: BLE001 - any failure -> heuristic fallback
        print(f"[discovery] spaCy unavailable ({type(exc).__name__}); "
              f"falling back to heuristic.", file=sys.stderr)
        return None


_CAP_PHRASE = re.compile(
    r"\b[A-Z][a-zA-Z.&'-]+(?:\s+(?:of|and|&|the|for)\s+)?"
    r"(?:\s+[A-Z][a-zA-Z.&'-]+)*(?:\s+[a-z]{2,4}\b)?"
)


def discover(text: str, nlp) -> list[tuple[str, str]]:
    """Return [(surface, kind)] where kind in {'PERSON','ORG'}.

    Discovery only. spaCy mislabels here (e.g. 'Paladin' as PERSON) — for
    redaction we do not trust the label, we just need the SPAN. ORG spans are
    extended to swallow a trailing lowercase 2-4 char token (the 'fs' in
    'Paladin fs').
    """
    found: list[tuple[str, str]] = []
    if nlp is not None:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ not in ("PERSON", "ORG"):
                continue  # GPE/locations left alone per D-006
            surface = ent.text
            # extend across a trailing lowercase acronym-ish token (org quirk)
            if ent.end < len(doc):
                nxt = doc[ent.end]
                if nxt.is_alpha and nxt.text.islower() and 2 <= len(nxt.text) <= 4:
                    surface = f"{surface} {nxt.text}"
            found.append((surface.strip(), ent.label_))
    else:
        for m in _CAP_PHRASE.finditer(text):
            found.append((m.group(0).strip(), "PERSON"))
    # de-dup, keep longest surfaces first later
    seen = set()
    out = []
    for surface, kind in found:
        key = surface.lower()
        if key and key not in seen:
            seen.add(key)
            out.append((surface, kind))
    return out


def is_public(surface: str) -> bool:
    return surface.lower().strip() in PUBLIC_ORG_ALLOWLIST


# --------------------------------------------------------------------------- #
# Domain-fragment fuzzing (D-007)
# --------------------------------------------------------------------------- #
def frag_variants(tokens: list[str]) -> set[str]:
    """From ['paladin','fs'] -> {'paladinfs','paladin-fs','paladin_fs',
    'paladin.fs','paladin'}. Suffix alone ('fs') is intentionally excluded."""
    toks = [t.lower() for t in tokens if t]
    out = set()
    if not toks:
        return out
    out.add(toks[0])  # root alone
    if len(toks) > 1:
        for sep in ("", "-", "_", "."):
            out.add(sep.join(toks))
    return out


def frag_map_for_org(real_org: str, fake_org: str) -> list[tuple[str, str]]:
    """Ordered (real_fragment, fake_fragment) pairs, longest real first."""
    real_toks = real_org.split()
    fake_toks = fake_org.split()
    pairs = []
    seps = {"": "", "-": "-", "_": "_", ".": "."}
    # multi-token joined variants (matched separator preserved)
    if len(real_toks) > 1 and len(fake_toks) > 1:
        for sep in seps:
            real_f = sep.join(t.lower() for t in real_toks)
            fake_f = sep.join(t.lower() for t in fake_toks)
            pairs.append((real_f, fake_f))
    # root alone
    pairs.append((real_toks[0].lower(), fake_toks[0].lower()))
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return pairs


def replace_fragment(text: str, real_frag: str, fake_frag: str) -> str:
    """Case-insensitive replace of real_frag when bounded by non-alnum (i.e. a
    whole label inside a URL/domain), so partials inside larger words are safe."""
    pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(real_frag) + r"(?![A-Za-z0-9])",
                     re.IGNORECASE)
    return pat.sub(fake_frag, text)


# --------------------------------------------------------------------------- #
# Core replacement for one document
# --------------------------------------------------------------------------- #
def build_alternation(pairs: list[tuple[str, str]]):
    """Compile one regex from real->fake pairs, longest real first, word-bounded.
    Returns (regex, lookup) where lookup is case-insensitive keyed."""
    pairs = sorted(pairs, key=lambda p: len(p[0]), reverse=True)
    lookup = {}
    parts = []
    for real, fake in pairs:
        if not real:
            continue
        lk = real.lower()
        if lk in lookup:
            continue
        lookup[lk] = fake
        parts.append(re.escape(real))
    if not parts:
        return None, {}
    rx = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(parts) + r")(?![A-Za-z0-9])",
                    re.IGNORECASE)
    return rx, lookup


def pseudonymize_text(text: str, subject, doc_persons, doc_orgs) -> str:
    """subject = (full, first, last, fake_full, fake_first, fake_last)."""
    full, first, last, fake_full, fake_first, fake_last = subject

    # --- Pass A: full names + org strings (longest-first, case-insensitive) ---
    pairs: list[tuple[str, str]] = []
    pairs.append((full, fake_full))
    for real_full, meta in doc_persons.items():
        pairs.append((real_full, meta["fake_full"]))
    for real_org, fake_org in doc_orgs.items():
        pairs.append((real_org, fake_org))
    rx, lookup = build_alternation(pairs)
    if rx:
        text = rx.sub(lambda m: lookup[m.group(1).lower()], text)

    # --- Pass B: bare tokens (subject + other discovered persons) ---
    # Case-sensitive, word-bounded; skip common-word / too-short tokens (D-004).
    bare_pairs: list[tuple[str, str]] = []

    def add_bare(real_tok, fake_tok):
        if (len(real_tok) >= MIN_BARE_TOKEN_LEN
                and real_tok.lower() not in COMMON_WORD_NAMES):
            bare_pairs.append((real_tok, fake_tok))

    add_bare(last, fake_last)
    add_bare(first, fake_first)
    for real_full, meta in doc_persons.items():
        toks = real_full.split()
        add_bare(toks[-1], meta["fake_last"])
        add_bare(toks[0], meta["fake_first"])
    bare_pairs.sort(key=lambda p: len(p[0]), reverse=True)
    for real_tok, fake_tok in bare_pairs:
        pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(real_tok) + r"(?![A-Za-z0-9])")
        text = pat.sub(fake_tok, text)

    # --- Pass C: domain fragments for mapped orgs + subject surname (D-007) ---
    # Leave opaque redirect hosts untouched.
    for real_org, fake_org in doc_orgs.items():
        for real_f, fake_f in frag_map_for_org(real_org, fake_org):
            text = replace_fragment(text, real_f, fake_f)
    if len(last) >= MIN_BARE_TOKEN_LEN and last.lower() not in COMMON_WORD_NAMES:
        text = replace_fragment(text, last.lower(), fake_last.lower())

    return text


# --------------------------------------------------------------------------- #
# Verification (hard gate)
# --------------------------------------------------------------------------- #
def collect_real_values(mp: dict, overrides: dict):
    """Returns (case_sensitive_tokens, case_insensitive_values)."""
    cs, ci = set(), set()
    for real_full, meta in mp["persons"].items():
        ci.add(real_full)  # full name safe case-insensitive
        toks = real_full.split()
        for t in (toks[0], toks[-1]):
            if len(t) >= MIN_BARE_TOKEN_LEN and t.lower() not in COMMON_WORD_NAMES:
                cs.add(t)                 # bare token: case-sensitive
                ci.add(t.lower())         # domain fragment: case-insensitive
    for real_org in mp["orgs"]:
        ci.add(real_org)
        for f in frag_variants(real_org.split()):
            ci.add(f)
    for k in overrides:
        ci.add(k)
    return cs, ci


def verify(mp: dict, overrides: dict) -> bool:
    cs_tokens, ci_values = collect_real_values(mp, overrides)
    clean_files = sorted(CLEAN_DIR.glob("*.md"))
    hits = []

    # filenames
    for f in clean_files:
        name = f.name
        for v in cs_tokens:
            if re.search(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])", name):
                hits.append(("FILENAME", f.name, v))
        for v in ci_values:
            if re.search(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])",
                         name, re.IGNORECASE):
                hits.append(("FILENAME", f.name, v))

    # contents
    for f in clean_files:
        body = f.read_text(encoding="utf-8")
        for v in cs_tokens:
            if re.search(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])", body):
                n = len(re.findall(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])", body))
                hits.append((f.name, v, n))
        for v in ci_values:
            if re.search(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])",
                         body, re.IGNORECASE):
                n = len(re.findall(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])",
                                   body, re.IGNORECASE))
                hits.append((f.name, v, n))

    if hits:
        print("\n=== VERIFICATION: FAIL ===")
        print(f"{len(hits)} leak(s) found. Add each real value to "
              f"data/mapping/overrides.json and rerun.")
        for h in hits[:200]:
            print("  LEAK:", h)
        return False

    print("\n=== VERIFICATION: PASS ===")
    print(f"0 real values found across {len(clean_files)} clean files + filenames "
          f"({len(cs_tokens)} bare tokens, {len(ci_values)} values/fragments checked).")
    return True


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    MAP_DIR.mkdir(parents=True, exist_ok=True)

    mp = load_json(MAP_PATH, None) or new_map()
    overrides = load_json(OVERRIDES_PATH, {})  # real -> fake, applied to every doc
    used = set(mp.get("used_fakes", []))
    factory = FakeFactory(used, SEED)
    nlp = load_nlp()

    review = {"orgs_mapped": {}, "persons_discovered": {}, "kept_public": []}
    used_fake_filenames: dict[str, int] = {}

    raw_files = sorted(RAW_DIR.glob("*.md"))
    if not raw_files:
        print(f"No .md files in {RAW_DIR}", file=sys.stderr)
        return 2
    print(f"Processing {len(raw_files)} reports "
          f"(discovery: {'spaCy' if nlp else 'heuristic'})...")

    for path in raw_files:
        full, first, last = parse_subject(path.stem)

        # subject -> persistent map (deterministic core)
        if full not in mp["persons"]:
            ffirst, flast, ffull = factory.unique_person()
            mp["persons"][full] = {
                "first": first, "last": last,
                "fake_full": ffull, "fake_first": ffirst, "fake_last": flast,
            }
        subj_meta = mp["persons"][full]
        subject = (full, first, last, subj_meta["fake_full"],
                   subj_meta["fake_first"], subj_meta["fake_last"])

        text = path.read_text(encoding="utf-8")

        # discovery -> grow the map (orgs + other persons)
        doc_persons: dict[str, dict] = {}
        doc_orgs: dict[str, str] = {}
        for surface, kind in discover(text, nlp):
            if surface.lower() == full.lower() or surface in mp["persons"]:
                if surface in mp["persons"] and surface != full:
                    doc_persons[surface] = mp["persons"][surface]
                continue
            if is_public(surface):
                review["kept_public"].append(surface)
                continue
            # subject's own surname/first alone -> handled deterministically
            if surface.lower() in (first.lower(), last.lower()):
                continue
            if kind == "PERSON" and len(surface.split()) >= 2:
                if surface not in mp["persons"]:
                    ff, fl, fu = factory.unique_person()
                    mp["persons"][surface] = {
                        "first": surface.split()[0], "last": surface.split()[-1],
                        "fake_full": fu, "fake_first": ff, "fake_last": fl,
                    }
                doc_persons[surface] = mp["persons"][surface]
                review["persons_discovered"][surface] = mp["persons"][surface]["fake_full"]
            else:
                # treat as org (NER label untrusted; redact the span)
                if surface not in mp["orgs"]:
                    shape = (len(surface.split()) >= 2
                             and surface.split()[-1].islower()
                             and 2 <= len(surface.split()[-1]) <= 4)
                    mp["orgs"][surface] = factory.unique_org(shape_suffix=shape)
                doc_orgs[surface] = mp["orgs"][surface]
                review["orgs_mapped"][surface] = mp["orgs"][surface]

        # manual overrides participate as pseudo-orgs (exact-string, longest-first)
        for real_v, fake_v in overrides.items():
            doc_orgs.setdefault(real_v, fake_v)

        clean = pseudonymize_text(text, subject, doc_persons, doc_orgs)

        # clean filename with collision suffix (D-003)
        base = subj_meta["fake_full"]
        fname = base
        if base.lower() in used_fake_filenames:
            used_fake_filenames[base.lower()] += 1
            fname = f"{base} {used_fake_filenames[base.lower()]}"
        else:
            used_fake_filenames[base.lower()] = 1
        (CLEAN_DIR / f"{fname}.md").write_text(clean, encoding="utf-8")

    # persist
    mp["used_fakes"] = sorted(used)
    MAP_PATH.write_text(json.dumps(mp, indent=2, ensure_ascii=False), encoding="utf-8")
    REVIEW_PATH.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
    if not OVERRIDES_PATH.exists():
        OVERRIDES_PATH.write_text("{}\n", encoding="utf-8")

    print(f"Wrote {len(raw_files)} clean files. "
          f"Map: {len(mp['persons'])} persons, {len(mp['orgs'])} orgs. "
          f"Review candidates -> {REVIEW_PATH.name}")

    ok = verify(mp, overrides)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
