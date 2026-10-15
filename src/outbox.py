"""The intro outbox (D-055): where a message the member approved is "sent".

This rebuild has no real messaging platform, so sending = one row here, in the same Postgres the
corpus lives in. Two implementations behind one small interface: the product writes to Postgres;
the self-checks and evals pass MemoryOutbox so a test run can never fill the real table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

DDL = """
CREATE TABLE IF NOT EXISTS intro_outbox (
    id              bigserial PRIMARY KEY,
    from_person_id  text        NOT NULL,
    to_person_id    text        NOT NULL,
    thread_id       text,
    message         text        NOT NULL,
    edited          boolean     NOT NULL DEFAULT false,
    sent_at         timestamptz NOT NULL DEFAULT now()
)"""


@dataclass
class Sent:
    from_person_id: str
    to_person_id: str
    thread_id: str | None
    message: str
    edited: bool
    sent_at: datetime


class PostgresOutbox:
    def __init__(self, conn):
        self.conn = conn
        with conn.cursor() as cur:
            cur.execute(DDL)
        conn.commit()

    def send(self, from_id: str, to_id: str, thread_id: str | None, message: str,
             edited: bool) -> Sent:
        with self.conn.cursor() as cur:
            cur.execute("INSERT INTO intro_outbox (from_person_id, to_person_id, thread_id, "
                        "message, edited) VALUES (%s, %s, %s, %s, %s) RETURNING sent_at",
                        (from_id, to_id, thread_id, message, edited))
            sent_at = cur.fetchone()[0]
        self.conn.commit()
        return Sent(from_id, to_id, thread_id, message, edited, sent_at)

    def last_sent(self, from_id: str, to_id: str) -> datetime | None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT max(sent_at) FROM intro_outbox WHERE from_person_id = %s "
                        "AND to_person_id = %s", (from_id, to_id))
            return cur.fetchone()[0]


@dataclass
class MemoryOutbox:
    sent: list[Sent] = field(default_factory=list)

    def send(self, from_id, to_id, thread_id, message, edited) -> Sent:
        row = Sent(from_id, to_id, thread_id, message, edited, datetime.now(timezone.utc))
        self.sent.append(row)
        return row

    def last_sent(self, from_id, to_id):
        times = [s.sent_at for s in self.sent if (s.from_person_id, s.to_person_id) == (from_id, to_id)]
        return max(times) if times else None
