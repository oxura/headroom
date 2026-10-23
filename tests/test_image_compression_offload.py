"""Image compression must stay off the event loop and behind the isolation runner.

The handlers no longer call `compressor.compress(...)` on the thread pool. They
delegate to `run_image_compression_isolated(...)`, which moves the native image
stack into a spawned subprocess so OpenCV crashes fail open without taking down
the proxy.
"""

from __future__ import annotations

import asyncio

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


async def test_image_isolation_keeps_event_loop_responsive() -> None:
    ticks = 0
    image_isolation._IMAGE_WORKER = image_isolation._sleep_worker

    async def _ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    tick_task = asyncio.create_task(_ticker())
    try:
        returned, result = await image_isolation.run_image_compression_isolated(
            [{"role": "user", "content": "image payload"}],
            provider="openai",
            timeout=1.0,
        )
    finally:
        tick_task.cancel()

    assert returned == [{"role": "user", "content": "image payload"}]
    assert result is None
    assert ticks >= 5
