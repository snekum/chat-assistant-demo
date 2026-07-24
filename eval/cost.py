"""Per-stage cost model (ROADMAP Tier-1 item 4).

generate.py + judge.py capture raw token usage + wall-clock per API call and return it as a
plain dict -- with NO pricing dependency, so the generator (system-under-test) never imports
the eval harness. This module holds the PRICES and turns a usage dict into dollars, so the $
figures live in ONE dated place and can be re-derived from recorded `usage` at any time.

Prices are LOAD-BEARING numbers the owner will cite in interviews -- snapshotted, dated, TUNABLE.
"""
from __future__ import annotations

# Per-MILLION-token USD prices, snapshotted 2026-07-24 (Anthropic pricing table).
# Cache read = 0.10x input; cache write (5-min TTL, the default) = 1.25x input.
# TUNABLE(prices dated 2026-07-24. claude-sonnet-5 here is INTRO pricing ($2/$10 per Mtok) that
#   runs through 2026-08-31, then reverts to standard ($3/$15) -- a 50% jump in JUDGE cost.
#   Revisit when: the date passes 2026-08-31, or the generator/judge model is swapped. Symptom
#   wrong: recomputed $/question diverges from recorded `usage` x these rates.)
PRICING = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5":  {"input": 2.00, "output": 10.00},  # intro; 3.00/15.00 after 2026-08-31
}
PRICES_DATED = "2026-07-24"
_CACHE_READ_MULT = 0.10
_CACHE_WRITE_MULT = 1.25  # 5-min TTL default; 1h TTL would be 2.0 (we don't cache these calls)


def cost_usd(meta: dict) -> float | None:
    """Dollars for one API call, from its captured usage dict. None if the model isn't priced
    (so an unpriced model surfaces as a null cost rather than a silently-wrong number)."""
    p = PRICING.get(meta.get("model"))
    if p is None:
        return None
    micro = (meta["input_tokens"] * p["input"]
             + meta["output_tokens"] * p["output"]
             + meta["cache_read_tokens"] * p["input"] * _CACHE_READ_MULT
             + meta["cache_creation_tokens"] * p["input"] * _CACHE_WRITE_MULT)
    return round(micro / 1_000_000, 6)
