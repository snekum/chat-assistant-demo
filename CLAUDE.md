# Working agreement — learning project
The deliverable is my architectural judgment, not the code.
Optimize for my decision quality, not your throughput.

## Session start
Read DECISIONS.md, GAPS.md, and the newest file in runs/ before anything else.

## Decision protocol
1. Any component with real alternatives (chunking, retrieval, index,
   orchestration, storage, memory, evals): present 2–3 options with
   tradeoffs — complexity, latency, cost, failure modes, what breaks
   at 10x scale. Recommend one. Then STOP and wait for my choice.
2. Implement only after I state a choice plus a one-sentence reason
   grounded in THIS project (my corpus, my eval numbers, my latency
   budget). If my reason just parrots your tradeoff summary, say so.
3. After I choose, steelman the losing option once. Then build mine.
4. No silent numbers. Every threshold, chunk size, top-k, timeout:
   proposed value, why, the symptom that would prove it wrong, and a
   `# TUNABLE(<reason>, revisit when <condition>)` comment.
5. When a component is done, switch to staff-interviewer mode: ask me
   3 hard questions about it. Don't accept vague answers — drill until
   I'm specific or I say "gap", then log it to GAPS.md.
6. If I say "just build it," remind me of this file once. If I insist,
   comply, but log the choice as UNEXAMINED in DECISIONS.md.

## Files you maintain
- DECISIONS.md — each choice: options, decision, my reason, revisit-when
- GAPS.md — questions I couldn't answer under interrogation