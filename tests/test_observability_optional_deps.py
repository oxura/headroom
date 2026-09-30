"""Configuration status must survive missing optional OTEL exporters."""

from __future__ import annotations

import builtins
from typing import Any
from unittest.mock import Mock

import pytest

from headroom.observability import metrics, tracing


@pytest.fixture
def without_otel_exporters(monkeypatch: pytest.MonkeyPatch):
    original_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith(("opentelemetry.exporter", "opentelemetry.sdk")):
            raise ImportError(f"Missing optional dependency: {name}")
        return original_import(name, *args, **kwargs)

    metrics.reset_otel_metrics()
    tracing.reset_headroom_tracing()
    monkeypatch.setattr(builtins, "__import__", blocked_import)
    yield
    metrics.reset_otel_metrics()
    tracing.reset_headroom_tracing()


@pytest.mark.parametrize("previous_provider", ["none", "normal", "raises"])
def test_metrics_keeps_explicit_config_without_exporters(
    without_otel_exporters, monkeypatch: pytest.MonkeyPatch, previous_provider: str
) -> None:
    initial_status = metrics.get_otel_metrics_status()
    provider = Mock()
    if previous_provider == "raises":
        provider.shutdown.side_effect = RuntimeError("provider shutdown failed")
    if previous_provider != "none":
        monkeypatch.setattr(metrics, "_owned_meter_provider", provider)
    config = metrics.OTelMetricsConfig(
        enabled=True,
        service_name="explicit-metrics",
        exporter="console",
        resource_attributes={"deployment.environment.name": "test"},
    )

    result = metrics.configure_otel_metrics(config)

    assert metrics.get_otel_metrics_status() == config.status()
    assert metrics._owned_meter_provider is None
    assert result is metrics.get_otel_metrics()
    if previous_provider != "none":
        provider.shutdown.assert_called_once_with()
    metrics.reset_otel_metrics()
    assert metrics.get_otel_metrics_status() == initial_status


@pytest.mark.parametrize("previous_provider", ["none", "normal", "raises"])
def test_tracing_keeps_explicit_config_without_exporters(
    without_otel_exporters, monkeypatch: pytest.MonkeyPatch, previous_provider: str
) -> None:
    initial_status = tracing.get_langfuse_tracing_status()
    provider = Mock()
    if previous_provider == "raises":
        provider.shutdown.side_effect = RuntimeError("provider shutdown failed")
    if previous_provider != "none":
        monkeypatch.setattr(tracing, "_owned_tracer_provider", provider)
    config = tracing.LangfuseTracingConfig(
        enabled=True,
        service_name="explicit-tracing",
        public_key="synthetic-public",
        secret_key="synthetic-secret",
        base_url="http://127.0.0.1:9999",
    )

    result = tracing.configure_langfuse_tracing(config)

    assert tracing.get_langfuse_tracing_status() == config.status()
    assert tracing._owned_tracer_provider is None
    assert result is tracing.get_headroom_tracer()
    assert "synthetic-secret" not in str(tracing.get_langfuse_tracing_status())
    if previous_provider != "none":
        provider.shutdown.assert_called_once_with()
    tracing.reset_headroom_tracing()
    assert tracing.get_langfuse_tracing_status() == initial_status
