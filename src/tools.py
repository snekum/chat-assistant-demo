"""Typed tools for the coordinator loop (D-036): the deterministic layer the model drives.

The census verdict this module implements: the atomic OPERATIONS are few and known; the
SEQUENCE and the OPERAND are chosen by the conversation. So each tool here is deterministic,
evidence-only (D-029: chunks + citations, no prose -- exactly ONE generation in the system,
at the coordinator), and takes session artifacts as operands where the census demands it
(within_person_ids for "who else amongst the people you suggested…", census C1/C8).

NOMINATE-VS-APPOINT AT THE TOOL LAYER (D-028, promoted by D-036): the model nominates person
references -- names as the user typed them, or person_ids it has seen in this conversation --
and only the resolver appoints. An unknown or ambiguous reference comes back as a STRUCTURED
ERROR steering the loop (clarify text included), never as an exception: the harness retries
transport failures; the model handles semantic results.

Errors are results. Every tool returns {"ok": bool, ...}; ok=False carries "error" (a short
machine-readable tag) and "detail" (what the model should do about it). Nothing here raises
on bad model input -- a raised exception would be retried by the harness, and a nomination
the resolver rejects is not a transport failure.

OWED, deliberately not built here: asker-personalised ranking for find_members (Step 6, behind
the memo §11 facet fork -- an OWNER call) and §12 confidence grades on evidence (synthesis
wiring). find_members ranks by evidence strength alone until that fork is decided.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone

sys.path.insert(0, "src")
import store  # noqa: E402
from aliases import alias_stop_terms, build_company_index  # noqa: E402
from contracts import CorpusEvidence  # noqa: E402
from resolve import Resolver, build_stop_tokens  # noqa: E402

# Adopted retrieval config (D-023): section_hdr @ k=5. The whole_doc scheme is RETAINED and
# serves the brief flow (full_dossier=True) -- the two schemes coexist in one DB by design.
SCHEME = "section_hdr"

# TUNABLE(5 chunks per person, the D-023 section_hdr hit@k plateau; symptom wrong: comparison
#         answers missing facts the dossier holds -> check rank 6+ in traces before raising)
K_PER_PERSON = 5

# TUNABLE(30-chunk wide sweep before grouping by person, from the G-004 design sketch; symptom
#         wrong: a member the offline sweep-gold marks strong never enters the pool -> raise)
SWEEP_K = 30

# TUNABLE(top-5 members returned, the memo §2 product cap; symptom wrong: users routinely ask
#         for more -> raise the cap, or ranking buries the right people -> fix ranking not cap)
TOP_N_MEMBERS = 5

# TUNABLE(<=3 supporting chunks per member in aggregates, memo §2 "no dossier hogging"; symptom
#         wrong: "why them" lines unsupported by the shown snippets -> raise)
SNIPPETS_PER_MEMBER = 3

# TUNABLE(6 people per evidence call; "compare five people" is a real ask, 268 is abuse. 6 x 5
#         chunks x ~200 tok ~= 6k tok of context. Symptom wrong: legitimate group asks bounce ->
#         raise and re-check the token math)
MAX_PERSONS_PER_CALL = 6


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_resolver(conn) -> Resolver:
    """Assemble the production Resolver from the DB: canonical names, the derived stop-list,
    and the company-alias index (all DERIVED at build time -- no real name or company is ever
    written into source; that rule is why this assembly exists instead of a data file)."""
    with conn.cursor() as cur:
        cur.execute("SELECT person_id, canonical_name FROM persons")
        persons = [{"person_id": pid, "canonical_name": name} for pid, name in cur.fetchall()]
        cur.execute("SELECT person_id, text FROM chunks WHERE chunk_scheme = 'whole_doc'")
        docs = [{"person_id": pid, "text": text} for pid, text in cur.fetchall()]

    names = [p["canonical_name"] for p in persons]
    texts = [d["text"] for d in docs]
    companies, covered = build_company_index(docs)
    # A silently thinner alias map is a silent resolution regression (the Phase-2 parser
    # lesson); the product refuses to start rather than degrade quietly.
    assert covered == len(persons), (
        f"company header parsed for {covered}/{len(persons)} persons -- corpus changed; "
        "fix aliases._COMPANY_HEADER before serving"
    )
    return Resolver(
        persons,
        stop_tokens=build_stop_tokens(names, texts),
        companies=companies,
        company_stop_terms=alias_stop_terms(companies, texts),
    )


class Toolbox:
    """One per request: binds the shared embedder/connection/resolver to the ASKER, so
    self-exclusion (census C1 "anyone else") is structural rather than a prompt hope."""

    # The loop reads the menu off the toolbox instance (test doubles carry their own).
    @property
    def schemas(self) -> list[dict]:
        return TOOL_SCHEMAS

    def __init__(self, emb, conn, resolver: Resolver, asker: str):
        self.emb = emb
        self.conn = conn
        self.resolver = resolver
        self.asker = asker

    # --- appointment ------------------------------------------------------------------------

    def _appoint(self, person_refs: list[str]) -> tuple[list[str], dict | None]:
        """Turn nominations into person_ids, or say exactly why not. A ref is either an id the
        model saw earlier in this conversation (validated -- ids are only ever minted here) or
        a name/company mention (resolved; a company appoints its member SET, D-035)."""
        ids: list[str] = []
        for ref in person_refs:
            ref = ref.strip()
            if ref in self.resolver.name_of:
                if ref not in ids:
                    ids.append(ref)
                continue
            res = self.resolver.resolve(ref)
            if res.status == "ambiguous":
                return [], {
                    "ok": False,
                    "error": "ambiguous_person",
                    "detail": self.resolver.clarify_question(res),
                    "candidates": {
                        mention: [{"person_id": i, "name": self.resolver.name_of[i]} for i in c]
                        for mention, c in res.candidates.items()
                    },
                }
            appointed = list(res.person_ids)
            # Self-reference is not a resolution (the asker is authenticated, resolve.py takes
            # no asker) -- it is appointed HERE, which is what makes "do we have anything in
            # common?" = person_refs ["me", "Ana"] work (census C3).
            if res.self_reference and self.asker not in appointed:
                appointed.append(self.asker)
            if appointed:
                for pid in appointed:
                    if pid not in ids:
                        ids.append(pid)
                continue
            return [], {
                "ok": False,
                "error": "not_a_member",
                "detail": (
                    f"{ref!r} is not a member of the network. If they might be mentioned inside "
                    "members' dossiers, search with find_members using the name as the criterion; "
                    "otherwise say honestly that they are not in the network."
                ),
            }
        if not ids:
            return [], {
                "ok": False,
                "error": "no_person_resolved",
                "detail": "No person reference resolved; name the member or pass a known person_id.",
            }
        return ids, None

    # --- tools ------------------------------------------------------------------------------

    def get_person_evidence(self, query: str, person_refs: list[str],
                            full_dossier: bool = False) -> dict:
        """Evidence about specific people. 1 person = lookup/fact; 2+ = comparison (the boundary
        is the argument count, not a flow). person-scoped pre-filter per person (D-016 made
        load-bearing -- the wrong-Ross class is structurally impossible once the id is right).
        full_dossier=True fetches the whole dossier per person (brief flow, census C3; retrieval
        is by-person, not by-similarity, so no query embedding runs). Include the asker's own id
        to join their profile against a target ("do we have anything in common?" = [asker, X])."""
        ids, err = self._appoint(person_refs)
        if err:
            return err
        if len(ids) > MAX_PERSONS_PER_CALL:
            return {
                "ok": False,
                "error": "too_many_persons",
                "detail": f"{len(ids)} people in one call; max {MAX_PERSONS_PER_CALL}. "
                          "Narrow the set or make separate calls.",
            }

        evidence: list[CorpusEvidence] = []
        ts = _now()
        if full_dossier:
            for pid in ids:
                for row in store.fetch_person_chunks(self.conn, pid, scheme="whole_doc"):
                    evidence.append(self._as_evidence(row, ts))
        else:
            qvec = self.emb.embed_query(query)
            # Sequential per-person scoped search: exact scan is ~20ms/person at this scale, so
            # a pooled-connection fan-out would buy nothing measurable yet. Symptom it stops
            # holding: per-person spans dominating the evidence-tool span in traces at real k.
            for pid in ids:
                for row in store.search(self.conn, qvec, k=K_PER_PERSON,
                                        person_id=pid, scheme=SCHEME):
                    evidence.append(self._as_evidence(row, ts))

        return {
            "ok": True,
            "persons": {pid: self.resolver.name_of[pid] for pid in ids},
            "evidence": evidence,
        }

    def find_members(self, criterion: str, exclude_person_ids: list[str] | None = None,
                     within_person_ids: list[str] | None = None) -> dict:
        """Discover members by criterion (aggregate flow). Wide sweep -> group by person ->
        rank by evidence strength -> top members with supporting snippets. The ASKER is always
        excluded (memo §6 self-exclusion). within_person_ids re-queries INSIDE a prior result
        set (census C1/C8: "who else amongst the people you suggested…") -- pass the ids from
        the earlier result, and each is searched person-scoped so nobody drops out merely by
        losing a global ranking race. No relevance cutoff yet: the cutoff is G-004 instrument
        design (Step 6, owner fork); until then weak tail members are visible, not hidden."""
        excluded = set(exclude_person_ids or ())
        excluded.add(self.asker)
        qvec = self.emb.embed_query(criterion)
        ts = _now()

        per_person: dict[str, list[CorpusEvidence]] = {}
        if within_person_ids is not None:
            ids, err = self._appoint(within_person_ids)
            if err:
                return err
            for pid in ids:
                if pid in excluded:
                    continue
                rows = store.search(self.conn, qvec, k=SNIPPETS_PER_MEMBER,
                                    person_id=pid, scheme=SCHEME)
                per_person[pid] = [self._as_evidence(r, ts) for r in rows]
        else:
            for row in store.search(self.conn, qvec, k=SWEEP_K, scheme=SCHEME):
                pid = row["person_id"]
                if pid in excluded:
                    continue
                bucket = per_person.setdefault(pid, [])
                if len(bucket) < SNIPPETS_PER_MEMBER:
                    bucket.append(self._as_evidence(row, ts))

        # Evidence strength = best supporting chunk. Deliberately the dumbest ranking that
        # still measures; coverage bonuses / asker-fit reordering arrive with the Step-6 fork,
        # pulled by the G-004 instrument's numbers rather than pushed by design taste.
        ranked = sorted(per_person.items(), key=lambda kv: -max(e.score for e in kv[1]))
        members = [
            {
                "person_id": pid,
                "name": self.resolver.name_of[pid],
                "best_score": round(max(e.score for e in items), 4),
                "snippets": items,
            }
            for pid, items in ranked[:TOP_N_MEMBERS]
        ]
        return {
            "ok": True,
            "members": members,
            "evidence": [e for m in members for e in m["snippets"]],
            "candidates_considered": len(per_person),
        }

    def web_search(self, query: str) -> dict:
        """Out-of-corpus knowledge (D-027: external search API returning typed results).
        Provider lands at Step 7 with recorded fixtures; until then the tool EXISTS and
        degrades honestly, so trajectories and schemas are real from day one."""
        return {
            "ok": False,
            "error": "web_unavailable",
            "detail": "Web search is not configured. Answer from corpus evidence or state "
                      "plainly that current external information is unavailable.",
        }

    # --- helpers ----------------------------------------------------------------------------

    def _as_evidence(self, row: dict, ts: str) -> CorpusEvidence:
        return CorpusEvidence(
            person_id=row["person_id"],
            doc_id=row["doc_id"],
            chunk_id=row["chunk_id"],
            text=row["text"],
            score=float(row.get("score", 0.0)),
            section=row.get("chunk_index"),
            retrieved_at=ts,
        )


# --- Tool definitions the provider adapter hands the model ----------------------------------

# Hand-written (D-029 idiom: schema design is eval design; the trajectory eval scores calls
# against these shapes). Descriptions are model-facing prompt surface: versioned with the
# coordinator prompt, not tweaked casually.
TOOL_SCHEMAS: list[dict] = [
    {
        "name": "get_person_evidence",
        "description": (
            "Retrieve dossier evidence about specific member(s). One person for a fact or "
            "brief; several to compare them (max 6). Include the asker's own id to check "
            "commonality with a target. Set full_dossier=true for meeting-prep briefs. "
            "person_refs: names as the user said them, or person_ids already seen in this "
            "conversation. Never invent an id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "Self-contained question, pronouns resolved."},
                "person_refs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "full_dossier": {"type": "boolean", "default": False},
            },
            "required": ["query", "person_refs"],
        },
    },
    {
        "name": "find_members",
        "description": (
            "Discover members matching a criterion (industry, experience, background). Returns "
            "the top members ranked by evidence strength, each with supporting snippets. The "
            "asker is never included. Pass within_person_ids to re-query INSIDE a previous "
            "result list (e.g. 'who among them has China experience'); pass exclude_person_ids "
            "for people the user has already ruled out."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "criterion": {"type": "string",
                              "description": "What kind of person is sought, self-contained."},
                "exclude_person_ids": {"type": "array", "items": {"type": "string"}},
                "within_person_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["criterion"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web for out-of-corpus, current information (news, policy, markets). "
            "Never use it for facts about members -- member claims come from dossiers only."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]


if __name__ == "__main__":
    # Live smoke against local PG (same pattern as retrieve.py). Prints shapes, not names,
    # so output is paste-safe.
    from embedder import NomicLocal

    conn = store.connect()
    resolver = build_resolver(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT person_id FROM persons ORDER BY person_id LIMIT 1")
        asker = cur.fetchone()[0]
    tb = Toolbox(NomicLocal(), conn, resolver, asker=asker)

    r = tb.find_members("experience expanding a business into China")
    print(f"find_members: ok={r['ok']} members={len(r.get('members', []))} "
          f"considered={r.get('candidates_considered')}")
    assert r["ok"] and all(m["person_id"] != asker for m in r["members"]), "self-exclusion broke"

    ids = [m["person_id"] for m in r["members"]]
    r2 = tb.find_members("manufacturing operations", within_person_ids=ids)
    print(f"find_members(within prior set): ok={r2['ok']} members={len(r2.get('members', []))}")
    assert r2["ok"] and all(m["person_id"] in ids for m in r2.get("members", [])), \
        "within_person_ids leaked outside the prior result set"

    r3 = tb.get_person_evidence("current role and company", [ids[0]])
    print(f"get_person_evidence: ok={r3['ok']} evidence={len(r3.get('evidence', []))} "
          f"sections={[e.section for e in r3.get('evidence', [])]}")

    r4 = tb.get_person_evidence("background", ["Zzyzx Nobody"])
    print(f"unknown person -> ok={r4['ok']} error={r4.get('error')}")
    assert r4["error"] in ("not_a_member", "no_person_resolved")

    r5 = tb.web_search("strait of hormuz closure shipping")
    print(f"web_search stub -> ok={r5['ok']} error={r5.get('error')}")
    print("toolbox smoke passed")
