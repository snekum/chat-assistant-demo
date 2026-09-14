"""Provider adapter (D-037): the one seam between this system and any LLM vendor.

The loop, tools, harness law, session store, and evals never import a vendor SDK -- they
speak to this Protocol. Today there is one implementation (Anthropic); the owner's D-037
requirement is the OPTION to swap in a local/open-source model later without touching the
loop, so the seam exists now while it is one file wide.

WIRE-FORMAT CHOICE, stated honestly: the neutral message shape used across the seam IS the
Anthropic Messages format (role + content blocks, tool_use/tool_result). Inventing a private
format today would mean writing a translation layer whose only consumer translates back to
the same shape. The cost lands where it belongs: a FUTURE adapter (OpenAI-compatible local
server, etc.) translates at its edge. If that ever feels wrong, the seam is still the only
place the format is spoken.

Transport-plane retries (429/5xx/connection) belong HERE, per the retry discipline: the
harness retries failures, the model handles results. The Anthropic SDK already retries
429/5xx with backoff (max_retries=2 default); we keep that and do not add a second layer.
# TUNABLE(SDK default max_retries=2 as the transport retry policy; symptom wrong: trace
#         spans show turns failing on blips a third retry would have absorbed -> raise on
#         the client, never per-call loops)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

# The coordinator's model. Closed-ish decisions over a small typed tool menu are a
# classification-shaped job; the cheapest tier gets it until measurement says otherwise.
# TUNABLE(Haiku 4.5 for the coordinator; symptom wrong: tool-choice accuracy on gold
#         materially below the gold's own authoring agreement -> bump tier before
#         redesigning prompts)
COORDINATOR_MODEL = "claude-haiku-4-5"

# The synthesizer keeps the model the f6/synth lineage was measured on.
SYNTH_MODEL = "claude-haiku-4-5"


@dataclass(frozen=True)
class ToolCallRequest:
    """One tool invocation the model asked for."""

    id: str
    name: str
    args: dict


@dataclass(frozen=True)
class ModelTurn:
    """What one model round returned, vendor-neutral."""

    text: str
    tool_calls: tuple[ToolCallRequest, ...]
    stop_reason: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw_content: list = field(default_factory=list, hash=False, compare=False)


class Provider(Protocol):
    """What the loop needs from a vendor. Two calls, nothing else."""

    def complete(self, system: str, messages: list[dict], tools: list[dict],
                 force_tool: bool, max_tokens: int) -> ModelTurn:
        """One non-streaming round (coordinator decisions)."""
        ...

    def stream_text(self, system: str, messages: list[dict], tools: list[dict] | None,
                    max_tokens: int, on_delta: Callable[[str], None] | None) -> ModelTurn:
        """One streaming text generation (the answer call). Streams deltas to `on_delta` and
        returns the completed turn.

        It returns a ModelTurn rather than yielding an iterator for one reason: USAGE. The
        defense pack owes an overhead row (tokens and latency of agency vs a direct retrieval
        call), and the answer call is half of that turn's cost -- a bare delta iterator would
        leave it unaccounted. Streaming still happens; the caller just gets the deltas through
        the callback instead of by iterating.
        """
        ...


class AnthropicProvider:
    """The Anthropic implementation. Everything vendor-specific lives in this class."""

    def __init__(self, model: str = COORDINATOR_MODEL, synth_model: str = SYNTH_MODEL):
        import anthropic  # imported here so unit tests with a fake provider need no SDK

        self.client = anthropic.Anthropic()
        self.model = model
        self.synth_model = synth_model

    def complete(self, system: str, messages: list[dict], tools: list[dict],
                 force_tool: bool, max_tokens: int = 1024) -> ModelTurn:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools,
            # force_tool=True means every round MUST end in a tool call -- with `respond`
            # in the menu this makes the declared outcome structural: there is no bare-text
            # path out of the loop for the harness to interpret.
            tool_choice={"type": "any"} if force_tool else {"type": "auto"},
        )
        calls = tuple(
            ToolCallRequest(id=b.id, name=b.name, args=dict(b.input))
            for b in resp.content if b.type == "tool_use"
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return ModelTurn(
            text=text,
            tool_calls=calls,
            stop_reason=resp.stop_reason or "",
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            raw_content=[b.to_dict() for b in resp.content],
        )

    def stream_text(self, system: str, messages: list[dict], tools: list[dict] | None = None,
                    max_tokens: int = 2048,
                    on_delta: Callable[[str], None] | None = None) -> ModelTurn:
        # WHY tools ARE SENT TO A CALL THAT MUST NOT USE THEM: the answer call replays the
        # turn's thread verbatim, and that history contains tool_use blocks -- the wire format
        # requires the tool definitions whenever it does. tool_choice "none" is what forbids
        # another round of calls. Sending the menu without "none" would let the answer call
        # start its own loop, outside harness law; omitting the menu is a 400.
        extra: dict = {}
        if tools:
            extra["tools"] = tools
            extra["tool_choice"] = {"type": "none"}

        with self.client.messages.stream(
            model=self.synth_model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            **extra,
        ) as stream:
            for delta in stream.text_stream:
                if on_delta is not None:
                    on_delta(delta)
            final = stream.get_final_message()

        return ModelTurn(
            text="".join(b.text for b in final.content if b.type == "text"),
            tool_calls=(),
            stop_reason=final.stop_reason or "",
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
            raw_content=[b.to_dict() for b in final.content],
        )
