"""Cross-encoder reranking of member search (experiment arm; NOT wired into find_members).

Meaning search compares a question vector with section vectors computed in advance, so the model
never reads the question and a section together. A cross-encoder does: it takes (question,
section) as one input and scores how well that section answers the question. Sharper, but one
model run per pair at question time, so it only re-orders a shortlist -- hybrid search's top POOL
people. People past the pool keep their hybrid order. What a person is judged on (`unit`):

  sections  the PER_PERSON sections the chat would show for them, one model run each
  profile   their whole dossier in ONE model run -- needs a long-input model (m3); the only unit
            that can see a person cover two needs written in two different sections

ONE implementation: eval/recommend_eval.py's ce_* arms call rerank_people, and find_members would
call the same function if an arm ships.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable

import hybrid
import store

MODELS = {"minilm": "cross-encoder/ms-marco-MiniLM-L6-v2",     # 22M params, web-search trained
          "bge": "BAAI/bge-reranker-base",                    # 278M params
          "m3": "BAAI/bge-reranker-v2-m3"}                    # 568M params, long input
# TUNABLE(each model's input limit. minilm/bge: 512, sections average ~190 tokens so truncation is
#   rare. m3: 8192 holds a whole dossier (p50 ~2.8k, max ~5.3k tokens). Symptom wrong: a profile
#   or section loses its relevant tail -> split before scoring.)
MAX_TOKENS = {"minilm": 512, "bge": 512, "m3": 8192}
# TUNABLE(pairs per forward pass; whole profiles are ~10x longer, and the GPU here has 4 GB.
#   Symptom wrong: CUDA out of memory -> lower; GPU mostly idle between batches -> raise.)
BATCH = {"minilm": 32, "bge": 32, "m3": 2}
# Half precision for m3 on a GPU: 568M params in fp32 plus 8k-token activations crowd 4 GB.
HALF_ON_GPU = {"m3"}
# TUNABLE(rerank hybrid's top 50: holds 69% of the key's clearly-useful people (recall@50) at
#   150 model runs per search. Symptom wrong: useful people sit just past 50 -> widen; rerank
#   time over budget -> narrow.)
POOL = 50
# TUNABLE(the 3 sections find_members already shows per person, so the reranker judges the
#   evidence the model will see. Symptom wrong: a person's on-point section is not among their
#   3 -> score more sections, at 1 model run each.)
PER_PERSON = 3

AGG: dict[str, Callable[[list[float]], float]] = {
    "max": max,                                    # a person = their single best section
    "sum3": lambda s: sum(sorted(s, reverse=True)[:3]),   # rewards several relevant sections
}
_CITE = re.compile(r"\[cite:[^\]]*\]")
_LOADED: dict = {}
# Scores memoised per (model, question, text): arms that differ only in how a person's scores
# combine (max vs sum3) share one scoring pass. STATS counts fresh vs reused pairs so an eval run
# can say when its latency excludes scoring. A live search almost never repeats a pair.
# TUNABLE(memo cleared past 200k pairs, ~50 MB of keys; symptom wrong: memory growth in a
#   long-running service -> drop the memo there entirely.)
_MEMO: dict[tuple[str, str, str], float] = {}
MEMO_MAX = 200_000
STATS = {"scored": 0, "reused": 0}


def device() -> str:
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def model(name: str):
    """Loaded once per process, on the GPU when there is one. Sigmoid scores (0..1) so sum3
    adds comparable numbers."""
    if name not in _LOADED:
        import torch
        from sentence_transformers import CrossEncoder
        half = name in HALF_ON_GPU and torch.cuda.is_available()
        _LOADED[name] = CrossEncoder(
            MODELS[name], max_length=MAX_TOKENS[name], activation_fn=torch.nn.Sigmoid(),
            model_kwargs={"torch_dtype": torch.float16} if half else {})
    return _LOADED[name]


def dtype(name: str) -> str:
    import torch
    return "float16" if name in HALF_ON_GPU and torch.cuda.is_available() else "float32"


def _clean(text: str) -> str:
    return " ".join(_CITE.sub("", text).split())


def rerank_people(conn, qvec, query: str, scheme: str, model_name: str = "minilm",
                  agg: str = "max", keep: Callable[[str], bool] = lambda p: True,
                  top: int = 5, unit: str = "sections") -> tuple[list[str], dict[str, list[dict]]]:
    """-> (every kept person, best first; the `top` people's sections) -- same shape as
    hybrid.rank_people, so a caller can switch between them."""
    order, sections = hybrid.rank_people(conn, qvec, query, scheme, keep=keep, top=POOL,
                                         per_person=PER_PERSON)
    pool = order[:POOL]
    pairs, whose = [], []
    for p in pool:
        if unit == "profile":
            texts = [store.fetch_person_chunks(conn, p, "whole_doc")[0]["text"]]
        else:
            texts = [row["text"] for row in sections[p]]
        for t in texts:
            pairs.append((query, _clean(t)))
            whose.append(p)
    todo = [pr for pr in dict.fromkeys(pairs) if (model_name, *pr) not in _MEMO]
    if todo:
        if len(_MEMO) > MEMO_MAX:
            _MEMO.clear()
        fresh = model(model_name).predict(todo, batch_size=BATCH[model_name],
                                          show_progress_bar=False)
        _MEMO.update({(model_name, *pr): float(s) for pr, s in zip(todo, fresh)})
    STATS["scored"] += len(todo)
    STATS["reused"] += len(pairs) - len(todo)
    scores = [_MEMO[(model_name, *pr)] for pr in pairs]
    per: dict[str, list[float]] = defaultdict(list)
    for p, s in zip(whose, scores):
        per[p].append(float(s))
    reranked = sorted(pool, key=lambda p: -AGG[agg](per[p]))    # stable: ties keep hybrid order
    return reranked + order[POOL:], {p: sections[p] for p in reranked[:top]}
