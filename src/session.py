"""Session store (D-036): threads and their artifacts, persisted.

The census demands exactly two things from state:
  1. The THREAD -- every message including tool results, replayed to the model each turn.
     That is what makes "tell me more about the first person" work: the model reads the
     prior find_members result in the thread and resolves the ordinal itself. No framework
     machinery resolves referents; storage + replay is the harness's entire job here.
  2. ARTIFACTS -- ranked result sets a turn produced, kept addressable so later turns can
     pass them back as tool operands (census C1/C8) and so a RETURNING session can recall
     them (census C7: "bring them up again").

Cross-session scope, per the census finding: this store persists threads + artifacts and
can list a user's prior threads. That is STATE persistence. The deferred thing stays
deferred: no learned-preference extraction, no memory layer -- the memo §12 boundary is
untouched. HOW a new session reaches into an old thread (auto-inject vs recall tool vs
UI thread-resume) is an owner call still owed; every option needs exactly this store.

Storage is a single jsonb-backed table in the same Postgres that holds the corpus -- one
store, no drift (D-014's lesson applied sideways). Messages are stored in the provider
wire shape verbatim, because the thread's exact bytes are what the model saw: rewriting
them on save would falsify the replay.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from contracts import ResultSetArtifact

SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    thread_id  text PRIMARY KEY,
    asker      text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    messages   jsonb NOT NULL DEFAULT '[]',
    artifacts  jsonb NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS threads_by_asker ON threads (asker, updated_at DESC);
"""


@dataclass
class Thread:
    thread_id: str
    asker: str
    messages: list[dict] = field(default_factory=list)
    artifacts: list[ResultSetArtifact] = field(default_factory=list)

    def next_artifact_id(self) -> str:
        return f"rs-{len(self.artifacts) + 1}"

    def find_artifact(self, artifact_id: str) -> ResultSetArtifact | None:
        for a in self.artifacts:
            if a.artifact_id == artifact_id:
                return a
        return None


class SessionStore:
    def __init__(self, conn):
        self.conn = conn
        conn.execute(SCHEMA)

    def create(self, asker: str) -> Thread:
        t = Thread(thread_id=f"t-{uuid.uuid4().hex[:12]}", asker=asker)
        self.save(t)
        return t

    def load(self, thread_id: str) -> Thread:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT asker, messages, artifacts FROM threads WHERE thread_id = %s",
                (thread_id,),
            )
            row = cur.fetchone()
        if row is None:
            raise KeyError(f"unknown thread {thread_id!r}")
        asker, messages, artifacts = row
        return Thread(
            thread_id=thread_id,
            asker=asker,
            messages=messages,
            artifacts=[ResultSetArtifact(**{**a, "person_ids": tuple(a["person_ids"]),
                                            "why": tuple(a.get("why", ()))})
                       for a in artifacts],
        )

    def save(self, thread: Thread) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO threads (thread_id, asker, messages, artifacts, updated_at)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (thread_id) DO UPDATE SET
                     messages = EXCLUDED.messages,
                     artifacts = EXCLUDED.artifacts,
                     updated_at = EXCLUDED.updated_at""",
                (thread.thread_id, thread.asker,
                 json.dumps(thread.messages),
                 json.dumps([a.to_dict() for a in thread.artifacts]),
                 datetime.now(timezone.utc)),
            )

    def list_for(self, asker: str, limit: int = 10) -> list[dict]:
        """Prior threads for a member, newest first -- the C7 recall surface. Returns
        metadata only; the recall MECHANISM (auto-inject / tool / UI resume) is the
        owner call flagged in the census, and each option starts from this list."""
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT thread_id, created_at, updated_at FROM threads
                   WHERE asker = %s ORDER BY updated_at DESC LIMIT %s""",
                (asker, limit),
            )
            return [{"thread_id": tid, "created_at": str(c), "updated_at": str(u)}
                    for tid, c, u in cur.fetchall()]
