from __future__ import annotations

import pytest

from headroom.tokenizers.huggingface import HuggingFaceTokenizer


def test_single_message_uses_the_template_without_a_generation_prompt(
    tmp_path, monkeypatch
) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    engine = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({"[UNK]": 0, "<eos>": 1, "<assistant>": 2}, unk_token="[UNK]")
    )
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=engine,
        unk_token="[UNK]",
        eos_token="<eos>",
        additional_special_tokens=["<assistant>"],
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }}<eos>{% endfor %}"
        "{% if add_generation_prompt %}<assistant>{% endif %}"
    )
    tokenizer.save_pretrained(tmp_path)
    monkeypatch.setenv("HEADROOM_HF_TOKENIZER_ALLOWLIST", str(tmp_path))
    monkeypatch.setenv("HEADROOM_HF_TOKENIZER_LOAD_TIMEOUT_SECS", "0")
    counter = HuggingFaceTokenizer(str(tmp_path))
    message = {"role": "user", "content": ""}

    assert counter.count_message(message) == 1
    assert counter.count_messages([message]) == 2
