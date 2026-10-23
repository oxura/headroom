from __future__ import annotations

import asyncio
import os

import pytest

from headroom.proxy import image_isolation


@pytest.fixture(autouse=True)
def _reset_image_pool() -> None:
    image_isolation._reset_image_pool()
    try:
        yield
    finally:
        image_isolation._reset_image_pool()
        image_isolation._IMAGE_WORKER = image_isolation._compress_messages_worker


def _messages() -> list[dict[str, object]]:
    return [{"role": "user", "content": "image payload"}]


async def test_worker_sigsegv_fails_open_parent_survives(monkeypatch) -> None:
    messages = _messages()
    monkeypatch.setattr(image_isolation, "_IMAGE_WORKER", image_isolation._abort_worker)

    returned, result = await image_isolation.run_image_compression_isolated(
        messages,
        provider="openai",
        timeout=5.0,
    )

    assert returned is messages
    assert result is None

    monkeypatch.setattr(image_isolation, "_IMAGE_WORKER", image_isolation._success_worker)
    recovered, recovered_result = await image_isolation.run_image_compression_isolated(
        messages,
        provider="openai",
        timeout=5.0,
    )

    assert recovered[-1]["content"] == "compressed:openai"
    assert recovered_result is not None


async def test_worker_exception_fails_open(monkeypatch) -> None:
    messages = _messages()
    monkeypatch.setattr(image_isolation, "_IMAGE_WORKER", image_isolation._raise_worker)

    returned, result = await image_isolation.run_image_compression_isolated(
        messages,
        provider="anthropic",
        timeout=5.0,
    )

    assert returned is messages
    assert result is None


async def test_worker_timeout_fails_open(monkeypatch) -> None:
    messages = _messages()
    monkeypatch.setattr(image_isolation, "_IMAGE_WORKER", image_isolation._sleep_worker)

    returned, result = await image_isolation.run_image_compression_isolated(
        messages,
        provider="openai",
        timeout=0.05,
    )

    assert returned is messages
    assert result is None

    monkeypatch.setattr(image_isolation, "_IMAGE_WORKER", image_isolation._success_worker)
    recovered, recovered_result = await image_isolation.run_image_compression_isolated(
        messages,
        provider="openai",
        timeout=5.0,
    )

    assert recovered[-1]["content"] == "compressed:openai"
    assert recovered_result is not None


async def test_worker_timeout_kills_the_abandoned_worker(monkeypatch) -> None:
    # shutdown(wait=False) alone left the worker running the abandoned call to
    # the end while the next request cold-started a fresh worker beside it.
    messages = _messages()
    monkeypatch.setattr(image_isolation, "_IMAGE_WORKER", image_isolation._success_worker)
    await image_isolation.run_image_compression_isolated(messages, provider="openai", timeout=30.0)
    (worker,) = image_isolation._IMAGE_POOL._processes.values()

    monkeypatch.setattr(image_isolation, "_IMAGE_WORKER", image_isolation._hang_worker)
    returned, result = await image_isolation.run_image_compression_isolated(
        messages,
        provider="openai",
        timeout=0.5,
    )

    assert returned is messages
    assert result is None
    worker.join(timeout=5.0)
    assert not worker.is_alive()


async def test_pool_reset_preserves_queued_request_payloads(monkeypatch) -> None:
    monkeypatch.setattr(image_isolation, "_IMAGE_WORKER", image_isolation._hang_worker)
    loop = asyncio.get_running_loop()
    await asyncio.wait_for(
        loop.run_in_executor(image_isolation._image_pool(), os.getpid),
        timeout=30.0,
    )
    payloads = [[{"role": "user", "content": f"image payload {i}"}] for i in range(9)]

    results = await asyncio.gather(
        *(
            image_isolation.run_image_compression_isolated(
                messages,
                provider="openai",
                timeout=0.2 if i == 0 else 10.0,
            )
            for i, messages in enumerate(payloads)
        )
    )

    assert results == [(messages, None) for messages in payloads]


async def test_caller_cancellation_still_propagates(monkeypatch) -> None:
    monkeypatch.setattr(image_isolation, "_IMAGE_WORKER", image_isolation._hang_worker)
    loop = asyncio.get_running_loop()
    await asyncio.wait_for(
        loop.run_in_executor(image_isolation._image_pool(), os.getpid),
        timeout=30.0,
    )
    task = asyncio.create_task(
        image_isolation.run_image_compression_isolated(
            _messages(), provider="anthropic", timeout=10.0
        )
    )
    await asyncio.sleep(0)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
