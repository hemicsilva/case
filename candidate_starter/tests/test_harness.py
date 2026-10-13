"""Testes unitários para as métricas do Pilar 3 — Evaluation Harness."""
import pytest
from candidate_starter.harness import (
    compute_precision_at_k,
    compute_router_metrics,
    compute_savings,
)


def test_compute_router_metrics_accuracy_and_matrix():
    y_true = ["FAST_PATH", "FAST_PATH", "AGENT", "AGENT"]
    y_pred = ["FAST_PATH", "AGENT", "AGENT", "AGENT"]
    labels = ["FAST_PATH", "AGENT"]

    metrics = compute_router_metrics(y_true, y_pred, labels)

    assert metrics["accuracy"] == 0.75
    cm = metrics["confusion_matrix"]
    assert cm["FAST_PATH"]["FAST_PATH"] == 1
    assert cm["FAST_PATH"]["AGENT"] == 1
    assert cm["AGENT"]["FAST_PATH"] == 0
    assert cm["AGENT"]["AGENT"] == 2


def test_compute_router_metrics_empty():
    labels = ["FAST_PATH", "AGENT"]
    metrics = compute_router_metrics([], [], labels)

    assert metrics["accuracy"] == 0.0
    assert metrics["confusion_matrix"]["FAST_PATH"]["FAST_PATH"] == 0


def test_compute_precision_at_k():
    hits = [1, 1, 0, 1, 0]  # 3 acertos em 5
    precision = compute_precision_at_k(hits)
    assert precision == 0.60

    assert compute_precision_at_k([]) == 0.0


def test_compute_savings():
    smart_cost = 0.20
    baseline_cost = 1.00
    smart_lat = 100.0
    baseline_lat = 2000.0

    savings = compute_savings(smart_cost, smart_lat, baseline_cost, baseline_lat)

    assert pytest.approx(savings["cost_savings_pct"], 0.01) == 80.0
    assert pytest.approx(savings["latency_savings_pct"], 0.01) == 95.0


def test_compute_savings_zero_division_guard():
    savings = compute_savings(0.0, 0.0, 0.0, 0.0)
    assert savings["cost_savings_pct"] == 0.0
    assert savings["latency_savings_pct"] == 0.0