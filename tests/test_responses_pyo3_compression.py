"""Rust binding tests for `/v1/responses` live-zone compression.

The default Python CLI runtime currently compresses Responses payloads
through CompressionUnit extraction plus ContentRouter. This module keeps
the lower-level PyO3 live-zone binding covered so Rust migration work
cannot silently break the exposed bridge.

The regressions cover byte-preserving passthrough, compression results,
non-fatal malformed input, and Python thread progress during native work.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest


def _ensure_binding():
    """Skip if the Rust extension hasn't been built (mirrors existing pattern)."""
    try:
        from headroom._core import compress_openai_responses_live_zone

        return compress_openai_responses_live_zone
    except ImportError:
        pytest.skip("headroom._core not built — run scripts/build_rust_extension.sh")


def test_native_work_allows_other_python_threads():
    _ensure_binding()
    source = textwrap.dedent(
        """
        import json
        import sys
        import threading
        from headroom import _core

        entered = threading.Event()
        active = False
        errors = []
        body = b'{"model":"gpt-4o-mini","input":[]}'

        def compress():
            global active
            active = True
            entered.set()
            try:
                for _ in range(10000):
                    result = _core.compress_openai_responses_live_zone(body, "payg", "gpt-4o-mini")
                    if result[0] != body or result[1]:
                        raise AssertionError("Passthrough bytes changed")
            except Exception as error:
                errors.append(repr(error))
            finally:
                active = False

        old_interval = sys.getswitchinterval()
        worker = threading.Thread(target=compress)
        try:
            sys.setswitchinterval(10)
            worker.start()
            if not entered.wait(15):
                raise RuntimeError("Native worker did not start")
            progress = active
            worker.join(15)
            if worker.is_alive():
                raise RuntimeError("Native worker did not finish")
        finally:
            sys.setswitchinterval(old_interval)
        print(json.dumps({"progress": progress, "errors": errors, "native": _core.__file__}))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    observed = json.loads(completed.stdout)
    assert observed["errors"] == []
    assert observed["progress"] is True, observed


class TestPassthroughCases:
    """Bodies the dispatcher cannot compress should be returned byte-for-byte
    with `modified=False`. Matches the Rust proxy's `Outcome::Passthrough`
    contract."""

    def test_not_json_passthrough(self):
        compress = _ensure_binding()
        body = b"this is not JSON at all"
        out, modified, _saved, _transforms, _reason = compress(body, "payg", "gpt-4o-mini")
        assert out == body
        assert modified is False

    def test_no_input_array_passthrough(self):
        compress = _ensure_binding()
        body = json.dumps({"model": "gpt-4o-mini"}).encode()
        out, modified, _saved, _transforms, _reason = compress(body, "payg", "gpt-4o-mini")
        assert out == body
        assert modified is False

    def test_empty_input_array_passthrough(self):
        compress = _ensure_binding()
        body = json.dumps({"model": "gpt-4o-mini", "input": []}).encode()
        out, modified, _saved, _transforms, _reason = compress(body, "payg", "gpt-4o-mini")
        assert out == body
        assert modified is False

    def test_no_eligible_items_passthrough(self):
        compress = _ensure_binding()
        # Single user message under the byte threshold — no compression
        # applies, but still valid input.
        body = json.dumps(
            {
                "model": "gpt-4o-mini",
                "input": [{"type": "message", "role": "user", "content": "hi"}],
            }
        ).encode()
        out, modified, _saved, _transforms, _reason = compress(body, "payg", "gpt-4o-mini")
        assert modified is False
        # Body should be byte-equal (passthrough, not re-serialized).
        assert out == body


class TestNoExceptionsLeak:
    """The binding's contract is `never raises` (matches the Rust proxy's
    `compress_openai_responses_request` passthrough-on-error semantics).
    Pin this so future maintainers don't accidentally introduce a
    raising path."""

    def test_garbage_bytes_no_raise(self):
        compress = _ensure_binding()
        out, modified, _saved, _transforms, _reason = compress(
            b"\xff\xfe\x00\xff", "payg", "gpt-4o-mini"
        )
        assert modified is False
        assert out == b"\xff\xfe\x00\xff"

    def test_empty_body_no_raise(self):
        compress = _ensure_binding()
        out, modified, _saved, _transforms, _reason = compress(b"", "payg", "gpt-4o-mini")
        assert modified is False
        assert out == b""


class TestTelemetryFields:
    """Direct binding consumers receive manifest token savings, applied
    strategies, and a machine-readable no-change reason."""

    def test_no_change_returns_zero_savings_and_empty_transforms(self):
        compress = _ensure_binding()
        body = json.dumps({"model": "gpt-4o-mini", "input": []}).encode()
        out, modified, saved, transforms, reason = compress(body, "payg", "gpt-4o-mini")
        assert modified is False
        assert out == body
        assert saved == 0
        assert transforms == []
        assert reason == "no_eligible_items"

    def test_large_local_shell_output_compresses_with_telemetry(self):
        """Compression preserves the call identity and a critical diagnostic."""
        compress = _ensure_binding()
        critical = "ERROR compile.rs:900 cannot find crate critical_dependency"
        log_body = "".join(
            critical + "\n"
            if i == 200
            else f"[2024-01-01 00:00:00] INFO compile.rs:42 building module foo_{i}\n"
            for i in range(400)
        )
        body = json.dumps(
            {
                "model": "gpt-4o",
                "input": [
                    {
                        "type": "local_shell_call_output",
                        "call_id": "c1",
                        "output": log_body,
                    }
                ],
            }
        ).encode()
        out, modified, saved, transforms, reason = compress(body, "payg", "gpt-4o")
        assert modified is True
        assert saved > 0
        assert reason is None
        new_doc = json.loads(out)
        assert new_doc["input"][0]["type"] == "local_shell_call_output"
        assert new_doc["input"][0]["call_id"] == "c1"
        assert critical in new_doc["input"][0]["output"]
