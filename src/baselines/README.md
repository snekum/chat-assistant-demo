# Baselines

The chat product runs on `src/agent.py`. The two systems in this folder came before it. They
are kept, not deleted, because every recorded evaluation run was produced by one of them, and
a baseline you can no longer run is a number you can no longer check.

| System | Files | Measured by | Superseded by |
|---|---|---|---|
| Single-call RAG pipeline: retrieve top-k chunks, one generation call | `retrieve.py`, `generate.py` | `eval/run.py` | the coordinator (D-029, D-036) |
| Coordinator tool loop: hand-rolled agent loop, separate pinned answer writer, declared outcomes, one-call greeting path | `coordinator.py`, `provider.py`, `synthesize.py`, `session.py`, `greeting.py`, `tracing.py` | `eval/agent_run.py` | the LangChain agent (D-042) |

Why the product moved off the coordinator is recorded in `DECISIONS.md` (D-037, D-042): the
separate answer-writer call cost one model call per turn without improving groundedness, and the
loop, memory and provider plumbing are what an agent framework already provides.

Self-checks (no API calls), run from `src/`:

```
python -m baselines.coordinator
python -m baselines.greeting
```
