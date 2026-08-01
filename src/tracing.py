"""Per-step traces (D-031): what the system DID while answering one query.

Day-one requirement, not a Phase-5 retrofit -- debugging an orchestrated workflow without
traces is archaeology, and the same records feed Phase 5's observability. Emitted as
OpenTelemetry spans because the format has a spec and an ecosystem: the viewer is then just an
exporter endpoint, so Phase 5's observability-substrate fork stays genuinely open instead of
being decided here by accident.

Confined ON PURPOSE (D-031): SDK + exporters only. No collector, no Docker, no backend beyond a
local viewer. Traces carry retrieved chunk text and real subject names, so hosted trace services
are ruled out on data-sensitivity grounds -- local-only is a hard constraint, not a preference.

THE FILE IS THE RECORD, THE UI IS A LENS. Every span is appended to a write-once JSONL artifact
alongside results.jsonl; a viewer reads the same spans over OTLP. Swapping or dropping the
viewer changes nothing that was measured.

Framework firewall (D-025 mitigation (a)): spans are emitted by plain functions here, never by
LangGraph internals, so a framework version bump cannot reshape a Phase-5 metric source.
LangGraph's own checkpoint/replay stays available for interactive debugging -- it just is not
the system of record.
"""
from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Sequence
from datetime import datetime, timezone

from opentelemetry import trace as _trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

# Bump when the emitted span shape changes (field names, what gets an attribute). Recorded as a
# resource attribute on every span and snapshotted into run config like PROMPT_CONTRACT_VERSION
# and the D-029 schema versions -- traces are about to be a metric source, so their shape is
# part of the instrument (D-021).
TRACE_SCHEMA_VERSION = "v1"

SERVICE_NAME = "chat-assistant"

# Local viewer's OTLP/HTTP endpoint -- Jaeger all-in-one (docker-compose service `jaeger`).
# Arize Phoenix was the first choice (LLM-native: token/cost views out of the box) but does not
# install on this box: its sqlean-py dependency has no Python 3.14 Windows wheel and needs MSVC
# build tools. Jaeger costs one container and shows the span waterfall; it knows nothing about
# LLMs, so token/cost views stay in the run artifacts where they already are. Swapping viewers
# is this one line, which was the whole point of emitting OTel rather than a homegrown format.
# TUNABLE(localhost:4318; symptom wrong: viewer shows nothing while trace.jsonl keeps filling
#         -> the viewer moved ports or is not running; the file artifact is unaffected)
VIEWER_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4318/v1/traces")

# --- Step vocabulary -----------------------------------------------------------------------

# Span names are the workflow's steps. Kept as a named list because three consumers read them:
# the trace file, the viewer, and the user-facing progress line (memo section 8 -- perceived
# latency: the user sees "searching profiles..." while the work happens).
STEP_PROGRESS: dict[str, str] = {
    "resolve": "looking up who you mean...",
    "route": "working out what you're asking for...",
    "retrieve_corpus": "searching profiles...",
    "retrieve_web": "checking recent news...",
    "synthesize": "writing the answer...",
}


def progress_message(step: str) -> str | None:
    """The user-facing line for a step, or None for steps the user should not see."""
    return STEP_PROGRESS.get(step)


# --- GenAI semantic conventions --------------------------------------------------------------

# OTel's agreed attribute names for LLM calls. This is the half that carries real vocabulary:
# a Phoenix/Jaeger/Grafana view understands these without configuration. The conventions are
# still marked experimental upstream, so they are pinned here by name and versioned with
# TRACE_SCHEMA_VERSION rather than tracked silently.
#
# NOTE: no dollars here, deliberately. generate.py records RAW token counts and leaves pricing
# to eval/cost.py so the system under test never imports the eval harness; spans keep that same
# split -- tokens in the span, cost computed downstream.
def record_llm_call(span, *, model: str, usage: dict, operation: str = "chat") -> None:
    """Attach GenAI-convention attributes from an existing usage_meta() dict (generate.py)."""
    span.set_attribute("gen_ai.system", "anthropic")
    span.set_attribute("gen_ai.operation.name", operation)
    span.set_attribute("gen_ai.request.model", model)
    span.set_attribute("gen_ai.usage.input_tokens", usage.get("input_tokens", 0))
    span.set_attribute("gen_ai.usage.output_tokens", usage.get("output_tokens", 0))
    if usage.get("cache_read_tokens"):
        span.set_attribute("gen_ai.usage.cache_read_input_tokens", usage["cache_read_tokens"])


# --- File exporter: the durable record -------------------------------------------------------


def _iso(ns: int | None) -> str | None:
    """OTel timestamps are nanoseconds since the epoch; run artifacts use ISO8601 UTC."""
    if ns is None:
        return None
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def span_to_dict(span: ReadableSpan) -> dict:
    """One span as a JSON-able row. Field names chosen so the file and the viewer agree."""
    ctx = span.get_span_context()
    parent = span.parent
    duration_ms = None
    if span.start_time is not None and span.end_time is not None:
        duration_ms = round((span.end_time - span.start_time) / 1e6, 3)
    return {
        "trace_id": f"0x{ctx.trace_id:032x}",
        "span_id": f"0x{ctx.span_id:016x}",
        "parent_id": f"0x{parent.span_id:016x}" if parent else None,
        "step": span.name,
        "start": _iso(span.start_time),
        "end": _iso(span.end_time),
        "duration_ms": duration_ms,
        "status": span.status.status_code.name,
        "attrs": dict(span.attributes or {}),
    }


class JsonlSpanExporter(SpanExporter):
    """Appends finished spans to a JSONL artifact. Write-once, same idiom as results.jsonl."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock, open(self.path, "a", encoding="utf-8", newline="\n") as fh:
            for span in spans:
                fh.write(json.dumps(span_to_dict(span), ensure_ascii=False) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:  # nothing held open between writes
        return None


class ProgressProcessor(SpanProcessor):
    """Fires a callback when a step STARTS, so the chat surface can show what is happening
    while it happens (memo section 8, perceived latency). Only attributes passed at span
    creation are visible here -- anything set later lands in the file, not the progress line."""

    def __init__(self, callback: Callable[[str, dict], None]):
        self.callback = callback

    def on_start(self, span, parent_context=None) -> None:
        msg = progress_message(span.name)
        if msg:
            self.callback(msg, dict(span.attributes or {}))

    def on_end(self, span: ReadableSpan) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


# --- Setup -----------------------------------------------------------------------------------

_PROVIDER: TracerProvider | None = None


def init_tracing(
    trace_path: str,
    *,
    to_viewer: bool = False,
    progress: Callable[[str, dict], None] | None = None,
) -> TracerProvider:
    """Wire up span export. Call once per process, before the graph runs.

    trace_path -- the durable JSONL artifact (runs/<id>/trace.jsonl in a real run).
    to_viewer  -- also ship spans to a LOCAL viewer over OTLP (Phoenix by default).
    progress   -- optional callback for user-facing step messages.

    SimpleSpanProcessor, not Batch: volume is a handful of spans per query, and a run artifact
    that is deterministic in ordering is worth more here than batching throughput.
    """
    global _PROVIDER
    resource = Resource.create(
        {
            "service.name": SERVICE_NAME,
            "trace.schema_version": TRACE_SCHEMA_VERSION,
        }
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(trace_path)))

    if to_viewer:
        # Imported lazily: a run that never opens a viewer should not need the OTLP exporter.
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        # BATCH here, unlike the file exporter, and the reason is measured not stylistic: with a
        # SimpleSpanProcessor the OTLP HTTP POST runs synchronously as each child span ends --
        # INSIDE the parent span. First demo run showed a `query` span of 210ms whose children
        # summed to under 1ms; the 210ms was export, not work. The defense pack's p50/p95 rows
        # would have inherited that. Batch moves export off the measured path.
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=VIEWER_ENDPOINT))
        )

    if progress is not None:
        provider.add_span_processor(ProgressProcessor(progress))

    _trace.set_tracer_provider(provider)
    _PROVIDER = provider
    return provider


def tracer(name: str = SERVICE_NAME):
    """The tracer every step uses. Safe to call before init_tracing (spans become no-ops)."""
    return _trace.get_tracer(name)


def shutdown() -> None:
    """Flush exporters at end of run."""
    if _PROVIDER is not None:
        _PROVIDER.shutdown()


if __name__ == "__main__":
    # Emits one realistic trace so the shape is visible before any of it is wired into the
    # graph: `python src/tracing.py [--viewer]`.
    import sys

    out = os.path.join(
        os.environ.get("TEMP", "."), "chat-assistant-demo-trace", "trace.jsonl"
    )
    if os.path.exists(out):
        os.remove(out)

    init_tracing(out, to_viewer="--viewer" in sys.argv, progress=lambda m, a: print(f"  ...{m}"))
    tr = tracer()

    with tr.start_as_current_span("query", attributes={"asker": "p042"}) as q:
        q.set_attribute("query", "Compare Jane Rivera and Alex Chen")
        with tr.start_as_current_span("resolve") as s:
            s.set_attribute("resolution.status", "resolved")
            s.set_attribute("resolution.person_ids", ["p101", "p204"])
        with tr.start_as_current_span("route") as s:
            s.set_attribute("router.stage", "rules")
            s.set_attribute("router.flow", "comparison")
            s.set_attribute("router.scope", "open")
            s.set_attribute("router.lane", "corpus")
        # Two person-scoped retrievals; in the graph these run in parallel, and the file is
        # what shows whether they actually overlapped.
        for pid in ("p101", "p204"):
            with tr.start_as_current_span("retrieve_corpus") as s:
                s.set_attribute("person_id", pid)
                s.set_attribute("retrieval.k", 5)
                s.set_attribute("retrieval.hits", 5)
        with tr.start_as_current_span("synthesize") as s:
            record_llm_call(
                s,
                model="claude-haiku-4-5",
                usage={"input_tokens": 4210, "output_tokens": 180, "cache_read_tokens": 0},
            )
            s.set_attribute("response_mode", "answer")
            s.set_attribute("citations", 2)

    shutdown()
    rows = [json.loads(line) for line in open(out, encoding="utf-8")]
    print(f"\n{len(rows)} spans -> {out}")
    by_id = {r["span_id"]: r for r in rows}
    for r in sorted(rows, key=lambda r: r["start"]):
        depth = 0
        p = r["parent_id"]
        while p in by_id:
            depth += 1
            p = by_id[p]["parent_id"]
        print(f"{'  ' * depth}{r['step']:<18} {r['duration_ms']:>7.2f} ms")
