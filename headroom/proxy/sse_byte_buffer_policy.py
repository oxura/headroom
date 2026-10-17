"""Pure SSE byte-buffer parsing policy."""

from __future__ import annotations

import re

# SSE lines may end in LF, CRLF or CR. The default event name is "message";
# return ``None`` so callers can decide whether to apply that default.
_SSE_EVENT_TERMINATOR_RE = re.compile(rb"(?:\r\n|\n|\r(?!\n))(?:\r\n?|\n)")
SSE_EVENT_LINE_PREFIX = "event:"
SSE_DATA_LINE_PREFIX = "data:"


def find_sse_event_terminator(buf: bytearray) -> tuple[int, int] | None:
    """Return the earliest complete SSE event terminator in ``buf``."""
    if not buf:
        return None
    # An empty leading line also consumes the LF left by a split CRLF.
    if buf[0] in (10, 13):
        return 0, 2 if buf.startswith(b"\r\n") else 1
    match = _SSE_EVENT_TERMINATOR_RE.search(buf)
    if match is None:
        return None
    start = match.start()
    return start, match.end() - start


def parse_sse_events_from_byte_buffer(
    buf: bytearray,
) -> list[tuple[str | None, str]]:
    """Drain complete ``event:`` + ``data:`` events from a bytes buffer.

    Returns list of ``(event_name, data_str)`` tuples for complete events.
    Mutates ``buf`` in-place to leave only partial-event tail bytes.

    Operates on bytes; only decodes complete events as UTF-8 (raises if a
    *complete* event has invalid UTF-8, which is an upstream protocol bug).
    """
    events: list[tuple[str | None, str]] = []
    while True:
        terminator_match = find_sse_event_terminator(buf)
        if terminator_match is None:
            break
        idx, terminator_len = terminator_match
        event_bytes = bytes(buf[:idx])
        del buf[: idx + terminator_len]

        event_text = event_bytes.decode("utf-8")
        event_name: str | None = None
        data_lines: list[str] = []
        for line in event_text.splitlines():
            if not line:
                continue
            if line.startswith(":"):
                continue
            if line.startswith(SSE_EVENT_LINE_PREFIX):
                event_name = line[len(SSE_EVENT_LINE_PREFIX) :].lstrip()
            elif line.startswith(SSE_DATA_LINE_PREFIX):
                data_lines.append(line[len(SSE_DATA_LINE_PREFIX) :].lstrip())

        if data_lines:
            events.append((event_name, "\n".join(data_lines)))
    return events
