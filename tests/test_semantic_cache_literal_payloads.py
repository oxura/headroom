from __future__ import annotations

import copy

import pytest

from headroom.proxy.semantic_cache import SemanticCache


@pytest.mark.asyncio
@pytest.mark.parametrize("location", ["tool_input", "tool_schema", "response_schema"])
async def test_cache_control_inside_model_data_does_not_alias_responses(location: str) -> None:
    messages = [{"role": "user", "content": "Configure the application."}]
    fields = {}
    if location == "tool_input":
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call1",
                        "name": "configure",
                        "input": {"cache_control": "first"},
                    }
                ],
            }
        ]
    elif location == "tool_schema":
        fields = {
            "tools": [
                {
                    "name": "configure",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "object", "default": {"cache_control": "first"}}
                        },
                    },
                }
            ]
        }
    else:
        fields = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": {"const": {"cache_control": "first"}}},
            }
        }
    changed_messages, changed_fields = copy.deepcopy((messages, fields))
    if location == "tool_input":
        changed_messages[0]["content"][0]["input"]["cache_control"] = "second"
    elif location == "tool_schema":
        changed_fields["tools"][0]["input_schema"]["properties"]["value"]["default"][
            "cache_control"
        ] = "second"
    else:
        changed_fields["response_format"]["json_schema"]["schema"]["const"]["cache_control"] = (
            "second"
        )

    cache = SemanticCache()
    response = b'{"id":"first-response","choices":[{"message":{"content":"first"}}]}'
    await cache.set(messages, "fixture", response, {}, partition="caller", **fields)

    assert (
        await cache.get(changed_messages, "fixture", partition="caller", **changed_fields) is None
    )
    original = await cache.get(messages, "fixture", partition="caller", **fields)
    assert original is not None and original.response_body == response
