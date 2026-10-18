from __future__ import annotations

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.hooks import CompressionHooks
from headroom.proxy.server import ProxyConfig, create_app


class InPlaceHooks(CompressionHooks):
    def pre_compress(self, messages, ctx):
        messages[0]["content"] = "rewritten in place"
        return messages


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_in_place_hook_does_not_rekey_a_cached_response(provider: str) -> None:
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
        hooks=InPlaceHooks(),
    )
    calls = 0
    with TestClient(create_app(config)) as client:

        async def upstream(method, url, headers, body, stream=False, **kwargs):
            nonlocal calls
            calls += 1
            if provider == "anthropic":
                reply = {
                    "id": f"msg_{calls}",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5",
                    "content": [{"type": "text", "text": "answer"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 5, "output_tokens": 1},
                }
            else:
                reply = {
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
                }
            return httpx.Response(200, json=reply)

        client.app.state.proxy._retry_request = upstream
        anthropic = provider == "anthropic"
        path = "/v1/messages" if anthropic else "/v1/chat/completions"
        headers = (
            {"x-api-key": "test-key", "anthropic-version": "2023-06-01"}
            if anthropic
            else {"authorization": "Bearer test-key"}
        )
        body = {
            "model": "claude-sonnet-4-5" if anthropic else "gpt-4o",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "original input"}],
        }
        first = client.post(path, headers=headers, json=body)
        second = client.post(path, headers=headers, json=body)
        assert first.status_code == second.status_code == 200
        assert second.json()["id"] == first.json()["id"]
        assert calls == 1
