"""A closed SSE payload ignores one BOM at the stream boundary, not in data."""

import json
from codecs import BOM_UTF8

import pytest

from headroom.ccr.response_handler import CCRResponseHandler, StreamingCCRHandler
from headroom.proxy.anthropic_wire import AnthropicSSEEnvelope


def _data_only(event):
    return b"data: " + json.dumps(event, ensure_ascii=False).encode() + b"\n\n"


@pytest.mark.parametrize(
    ("prefix", "between", "content"),
    [
        (BOM_UTF8, b"", "\ufefffirstsecond"),
        (BOM_UTF8 * 2, b"", "second"),
        (b"", BOM_UTF8, "\ufefffirst"),
    ],
)
def test_closed_openai_stream_only_ignores_one_initial_bom(prefix, between, content):
    first = _data_only({"choices": [{"index": 0, "delta": {"content": "\ufefffirst"}}]})
    second = _data_only(
        {"choices": [{"index": 0, "delta": {"content": "second"}, "finish_reason": "stop"}]}
    )
    handler = StreamingCCRHandler(CCRResponseHandler(), provider="openai")

    response = handler._parse_sse_stream(prefix + first + between + second)

    assert response["choices"][0]["message"]["content"] == content


@pytest.mark.parametrize(("prefix", "has_start"), [(BOM_UTF8, True), (BOM_UTF8 * 2, False)])
def test_anthropic_data_only_message_start_after_bom(prefix, has_start):
    start = _data_only(
        {
            "type": "message_start",
            "message": {
                "id": "msg_bom",
                "type": "message",
                "role": "assistant",
                "model": "local-replay",
                "content": [],
                "usage": {"input_tokens": 9, "output_tokens": 1},
            },
        }
    )

    envelope = AnthropicSSEEnvelope.parse(prefix + start + _data_only({"type": "message_stop"}))

    assert envelope.saw_message_start is has_start
    if has_start:
        assert envelope.message["id"] == "msg_bom"
