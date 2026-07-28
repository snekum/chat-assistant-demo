"""Infra smoke check -- proves the SETUP works, WITHOUT the private corpus.

A fresh clone cannot run a real eval (the 268 real-person dossiers are gitignored), but it
CAN verify the plumbing this repo depends on. This script checks exactly that and nothing
more: deps import, Postgres+pgvector is reachable and the schema is creatable, and the local
embedder loads and produces a 768-d unit vector. It never reads data/raw and makes NO paid
API calls -- it only reports whether the ANTHROPIC_API_KEY the online lanes need is present.

Usage:  ./.venv/Scripts/python.exe scripts/smoke.py
Exit 0 = every hard check passed (a real run's infra is sound); non-zero = something broken.

First run downloads the ~0.5 GB nomic embedding model from HuggingFace and loads torch
(~15-20s); subsequent runs are fast.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_results: list[tuple[str, str, str]] = []


def record(check: str, status: str, detail: str = "") -> None:
    _results.append((check, status, detail))
    icon = {"PASS": "  ok ", "FAIL": "FAIL ", "SKIP": " -- "}[status]
    print(f"[{icon}] {check}" + (f"  ({detail})" if detail else ""))


def check_imports() -> None:
    """Hard: the packages every lane imports must be present."""
    missing = []
    for mod in ("numpy", "psycopg", "pgvector", "anthropic", "dotenv",
                "sentence_transformers", "torch"):
        try:
            __import__(mod)
        except Exception as e:  # noqa: BLE001
            missing.append(f"{mod}: {e.__class__.__name__}")
    if missing:
        record("deps import", FAIL, "; ".join(missing) + " -- run: pip install -r requirements.txt")
    else:
        record("deps import", PASS)


def check_db() -> None:
    """Hard: Postgres reachable + pgvector extension creatable + schema applies."""
    try:
        import store  # noqa: PLC0415  (path set above)
    except Exception as e:  # noqa: BLE001
        record("pgvector db", FAIL, f"could not import store: {e}")
        return
    try:
        conn = store.connect(init=True)  # creates the vector extension + persons/chunks tables
        ver = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        conn.close()
        if ver:
            record("pgvector db", PASS, f"extension vector {ver[0]}, schema created")
        else:
            record("pgvector db", FAIL, "connected but 'vector' extension missing")
    except Exception as e:  # noqa: BLE001
        record("pgvector db", FAIL,
               f"{e.__class__.__name__} -- is `docker compose up -d` running? DSN={store.DSN}")


def check_embedder() -> None:
    """Hard: the local embedder loads and returns a normalized 768-d vector."""
    try:
        import numpy as np  # noqa: PLC0415
        from embedder import NomicLocal  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        record("embedder", FAIL, f"import failed: {e}")
        return
    try:
        emb = NomicLocal(use_cache=False)  # don't pollute the real cache with a dummy string
        vec = emb.embed_query("smoke test query")
        norm = float(np.linalg.norm(vec))
        if vec.shape == (emb.dim,) and abs(norm - 1.0) < 1e-3:
            record("embedder", PASS, f"{emb.model_id}, dim={emb.dim}, |v|={norm:.4f}")
        else:
            record("embedder", FAIL, f"unexpected shape/norm: shape={vec.shape}, |v|={norm:.4f}")
    except Exception as e:  # noqa: BLE001
        record("embedder", FAIL, f"{e.__class__.__name__}: {e}")


def check_api_key() -> None:
    """Soft: the online lanes (generation + judge) need this; retrieval lanes don't."""
    try:
        from dotenv import load_dotenv  # noqa: PLC0415
        load_dotenv(ROOT / ".env")
    except Exception:  # noqa: BLE001
        pass
    if os.environ.get("ANTHROPIC_API_KEY"):
        record("ANTHROPIC_API_KEY", PASS, "online lanes available")
    else:
        record("ANTHROPIC_API_KEY", SKIP,
               "unset -- offline retrieval lanes still run; generation+judge will be skipped")


def main() -> int:
    print("Infra smoke check (no corpus, no paid API calls)\n" + "-" * 48)
    check_imports()
    check_db()
    check_embedder()
    check_api_key()

    hard_failed = [c for c, s, _ in _results if s == FAIL]
    print("-" * 48)
    if hard_failed:
        print(f"RESULT: {len(hard_failed)} check(s) failed: {', '.join(hard_failed)}")
        return 1
    print("RESULT: infra OK. With the private corpus in data/raw/, a full run can proceed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
