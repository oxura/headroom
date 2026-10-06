"""Signed savings must survive the public views and durable accounting paths."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta, timezone

import pytest

from headroom.proxy.persistent_metrics import PersistentMetricsState
from headroom.proxy.savings_tracker import SavingsTracker


def _record(tracker, compression, tool=0.0, *, tokens=10, timestamp=None):
    tracker.record_request(
        model="test-model",
        provider="test-provider",
        project="test-project",
        input_tokens=100,
        tokens_saved=tokens,
        estimated_savings_usd={
            "compression": compression,
            "tool_schema": tool,
            "compression_list": compression,
            "tool_schema_list": tool,
            "basis": "mix",
        },
        timestamp=timestamp,
    )


@pytest.mark.parametrize("tokens", [0, 10])
def test_losses_survive_public_views_restarts_and_subsequent_requests(tmp_path, tokens):
    path = tmp_path / "savings.json"
    tracker = SavingsTracker(path=str(path))
    for turn in range(1, 4):
        _record(tracker, -0.3, -0.2, tokens=tokens)
        tracker.flush()
        for current in (tracker, SavingsTracker(path=str(path))):
            snapshot = current.stats_preview()
            for section in ("lifetime", "display_session"):
                assert snapshot[section]["compression_savings_usd"] == pytest.approx(-0.5 * turn)
                assert snapshot[section]["compression_savings_list_usd"] == pytest.approx(
                    -0.5 * turn
                )
                assert snapshot[section]["tool_schema_savings_usd"] == pytest.approx(-0.2 * turn)
            assert snapshot["by_model"]["test-model"]["compression_savings_usd"] == pytest.approx(
                -0.5 * turn
            )
            assert snapshot["projects"]["test-project"]["compression_savings_usd"] == pytest.approx(
                -0.5 * turn
            )
            history = current.history_response()
            assert len(history["history"]) == turn
            for series in history["series"].values():
                assert sum(row["compression_savings_usd_delta"] for row in series) == pytest.approx(
                    -0.5 * turn
                )
                assert sum(row["tool_schema_savings_usd_delta"] for row in series) == pytest.approx(
                    -0.2 * turn
                )
        tracker = SavingsTracker(path=str(path))


@pytest.mark.parametrize("loss", [-0.4, -1.5])
def test_rollups_and_csv_keep_losses_even_when_total_stays_positive(tmp_path, loss):
    tracker = SavingsTracker(path=str(tmp_path / "s.json"))
    start = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
    for index, amount in enumerate((1.0, loss, 0.25)):
        _record(tracker, amount, timestamp=start + timedelta(days=index))
    for current in (tracker, SavingsTracker(path=tracker.storage_path)):
        rows = current.history_response()["series"]["daily"]
        assert [row["compression_savings_usd_delta"] for row in rows] == pytest.approx(
            [1.0, loss, 0.25]
        )
        for row, amount in zip(rows, (1.0, loss, 0.25), strict=True):
            for breakdown, name in (("by_model", "test-model"), ("by_provider", "test-provider")):
                assert row[breakdown][name]["compression_savings_usd_delta"] == pytest.approx(
                    amount
                )
        exported = list(csv.DictReader(io.StringIO(current.export_csv("daily"))))
        assert [float(row["compression_savings_usd_delta"]) for row in exported] == pytest.approx(
            [1.0, loss, 0.25]
        )


@pytest.mark.parametrize("current", [-0.5, 0.0, 0.5])
def test_current_signed_total_wins_over_an_older_larger_checkpoint(tmp_path, current):
    path = tmp_path / "s.json"
    tracker = SavingsTracker(path=str(path))
    _record(tracker, 1.0)
    raw = json.loads(path.read_text())
    # Older releases could persist a monetary-only loss without a checkpoint.
    raw["lifetime"]["compression_savings_usd"] = current
    raw["lifetime"]["tool_schema_savings_usd"] = current
    raw["history"][-1]["tool_schema_savings_usd"] = 1.0
    path.write_text(json.dumps(raw), encoding="utf-8", newline="\n")
    lifetime = SavingsTracker(path=str(path)).snapshot()["lifetime"]
    assert lifetime["compression_savings_usd"] == current
    assert lifetime["tool_schema_savings_usd"] == current


@pytest.mark.parametrize("invalid", [None, "invalid", float("nan"), float("inf"), -float("inf")])
def test_invalid_lifetime_savings_fall_back_to_signed_history(tmp_path, invalid):
    path = tmp_path / "s.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 6,
                "lifetime": {
                    "compression_savings_usd": invalid,
                    "tool_schema_savings_usd": invalid,
                },
                "history": [
                    {
                        "timestamp": "2026-10-06T00:00:00Z",
                        "total_tokens_saved": 10,
                        "compression_savings_usd": -0.5,
                        "tool_schema_savings_usd": -0.2,
                    }
                ],
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    lifetime = SavingsTracker(path=str(path)).snapshot()["lifetime"]
    assert lifetime["compression_savings_usd"] == -0.5
    assert lifetime["tool_schema_savings_usd"] == -0.2


def test_legacy_history_still_repairs_monotonic_totals(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 0,
                "lifetime": {"tokens_saved": 1, "compression_savings_usd": 0.001},
                "history": [["2026-10-06T00:00:00Z", 30, 0.03]],
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    lifetime = SavingsTracker(path=str(path)).snapshot()["lifetime"]
    assert lifetime["compression_savings_usd"] == 0.03
    assert lifetime["compression_savings_list_usd"] == 0.03


def test_persistent_metrics_preserve_signed_effective_and_list_savings():
    metrics = PersistentMetricsState()
    for index, amount in enumerate((0.25, -0.75, -0.5), start=1):
        metrics.record_request(
            provider="test",
            stack="test",
            model="test",
            compression_savings_usd=amount,
            compression_savings_list_usd=amount,
            savings_basis="mix",
        )
        metrics = PersistentMetricsState(metrics.to_dict())
        expected = (0.25, -0.5, -1.0)[index - 1]
        assert metrics.to_dict()["cost"]["compression_savings_usd"] == expected
        assert metrics.to_dict()["cost"]["compression_savings_list_usd"] == expected


@pytest.mark.parametrize("invalid", [None, "invalid", float("nan"), float("inf"), -float("inf")])
def test_persistent_metrics_reject_nonfinite_savings_and_keep_other_floors(invalid):
    metrics = PersistentMetricsState()
    metrics.record_request(
        provider="test",
        stack="test",
        model="test",
        input_tokens=-10,
        tokens_saved=-10,
        input_usd=-1.0,
        cache_savings_usd=-1.0,
        compression_savings_usd=invalid,
        compression_savings_list_usd=invalid,
    )
    result = PersistentMetricsState(metrics.to_dict()).to_dict()
    assert result["tokens"]["input"] == result["tokens"]["saved"] == 0
    assert result["cost"]["input_usd"] == result["cost"]["cache_savings_usd"] == 0
    assert result["cost"]["compression_savings_usd"] == 0
    assert result["cost"]["compression_savings_list_usd"] == 0


def test_unpriced_legacy_negative_metrics_seed_list_column():
    result = PersistentMetricsState({"cost": {"compression_savings_usd": -0.5}}).to_dict()
    assert result["cost"]["compression_savings_usd"] == -0.5
    assert result["cost"]["compression_savings_list_usd"] == -0.5
    assert result["cost"]["savings_basis"] == "list"


def test_zero_net_monetary_change_keeps_disjoint_tool_history(tmp_path):
    tracker = SavingsTracker(path=str(tmp_path / "s.json"))
    _record(tracker, 0.5, -0.5, tokens=0)
    row = tracker.history_response()["series"]["daily"][0]
    assert row["compression_savings_usd_delta"] == 0
    assert row["tool_schema_savings_usd_delta"] == -0.5
    assert tracker.snapshot()["history"][0]["tool_schema_savings_usd"] == -0.5


@pytest.mark.parametrize("tool", [0.2, -0.2])
def test_compression_only_recorder_preserves_tool_cumulative_total(tmp_path, tool):
    tracker = SavingsTracker(path=str(tmp_path / "s.json"))
    _record(tracker, 0.3, tool)
    tracker.record_compression_savings(model="test-model", tokens_saved=10)
    for current in (tracker, SavingsTracker(path=tracker.storage_path)):
        assert current.snapshot()["lifetime"]["tool_schema_savings_usd"] == tool
        assert (
            current.history_response()["series"]["daily"][0]["tool_schema_savings_usd_delta"]
            == tool
        )


@pytest.mark.parametrize("first", [0.5, -0.5])
@pytest.mark.parametrize("sparse", ["mapping", "tuple"])
def test_old_sparse_checkpoints_do_not_invent_a_savings_reversal(tmp_path, first, sparse):
    path = tmp_path / "s.json"
    omitted = (
        {"timestamp": "2026-10-06T02:00:00Z", "total_tokens_saved": 20}
        if sparse == "mapping"
        else ["2026-10-06T02:00:00Z", 20]
    )
    raw = {
        "schema_version": 6,
        "history": [
            omitted,  # Intentionally out of order: carry-forward must follow time.
            {
                "timestamp": "2026-10-06T01:00:00Z",
                "total_tokens_saved": 10,
                "compression_savings_usd": first,
                "tool_schema_savings_usd": first,
            },
            {
                "timestamp": "2026-10-06T03:00:00Z",
                "total_tokens_saved": 30,
                "compression_savings_usd": 0.0,
                "tool_schema_savings_usd": 0.0,
            },
        ],
    }
    path.write_text(json.dumps(raw), encoding="utf-8", newline="\n")
    tracker = SavingsTracker(path=str(path))
    rows = tracker.history_response()["series"]["hourly"]
    assert [r["compression_savings_usd_delta"] for r in rows] == [first, 0, -first]
    assert [r["tool_schema_savings_usd_delta"] for r in rows] == [first, 0, -first]
