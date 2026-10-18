from __future__ import annotations

import copy

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app


@pytest.mark.parametrize("changed", [{"voice": "echo"}, {"format": "mp3"}])
def test_audio_configuration_partitions_response_cache(changed: dict[str, str]) -> None:
    config = ProxyConfig(
        optimize=False,
        cache_enabled=True,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    calls = 0
    with TestClient(create_app(config)) as client:

        async def upstream(method, url, headers, body, stream=False, **kwargs):
            nonlocal calls
            calls += 1
            return httpx.Response(
                200,
                json={
                    "id": f"chatcmpl_{calls}",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "answer"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                },
            )

        client.app.state.proxy._retry_request = upstream
        first = {
            "model": "gpt-4o-audio-preview",
            "messages": [{"role": "user", "content": "Say hello."}],
            "modalities": ["text", "audio"],
            "audio": {"voice": "alloy", "format": "wav"},
        }
        second = copy.deepcopy(first)
        second["audio"].update(changed)
        responses = [
            client.post(
                "/v1/chat/completions",
                headers={"authorization": "Bearer test-key"},
                json=body,
            )
            for body in (first, second, second)
        ]
        assert [response.status_code for response in responses] == [200, 200, 200]
        assert [response.json()["id"] for response in responses] == [
            "chatcmpl_1",
            "chatcmpl_2",
            "chatcmpl_2",
        ]
        assert calls == 2
