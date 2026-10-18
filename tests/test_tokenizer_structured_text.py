from __future__ import annotations

import copy

import pytest

from headroom.tokenizers.estimator import EstimatingTokenCounter
from headroom.tokenizers.tiktoken_counter import TiktokenCounter


@pytest.mark.parametrize("counter", [TiktokenCounter("gpt-4o"), EstimatingTokenCounter()])
def test_structured_text_fields_count_like_their_text_representation(counter) -> None:
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": {"message": "привет"}},
            {"text": 42},
            {"reasoningContent": {"reasoningText": {"text": ["a", "b"]}}},
            {"type": "thinking", "thinking": {"step": "inspect"}},
        ],
    }
    expected = {
        "role": "user",
        "content": [
            {"type": "text", "text": '{"message": "привет"}'},
            {"text": "42"},
            {"reasoningContent": {"reasoningText": {"text": '["a", "b"]'}}},
            {"type": "thinking", "thinking": '{"step": "inspect"}'},
        ],
    }
    original = copy.deepcopy(message)
    assert counter.count_messages([message]) == counter.count_messages([expected])
    assert message == original
