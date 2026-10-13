"""Testes unitários para o Pilar 1 — QueryRouter."""
import pytest
from candidate_starter.router import QueryRouter
from common.schemas import RouteResult


def test_router_raises_runtime_error_when_predict_before_fit():
    router = QueryRouter()
    with pytest.raises(RuntimeError, match="Chame fit() antes de predict()"):
        router.predict("Qual o meu saldo?")


def test_router_raises_value_error_with_empty_training_data():
    router = QueryRouter()
    with pytest.raises(ValueError):
        router.fit([], [])


def test_router_predict_schema_and_types():
    texts = [
        "Bom dia, tudo bem?",
        "Qual o horário de atendimento?",
        "Quero consultar meu saldo",
        "Qual a fatura do cartão?",
    ]
    labels = ["FAST_PATH", "FAST_PATH", "AGENT", "AGENT"]

    router = QueryRouter().fit(texts, labels)
    result = router.predict("Quero saber meu saldo disponível")

    assert isinstance(result, RouteResult)
    assert result.route in ["FAST_PATH", "AGENT"]
    assert isinstance(result.latency_ms, float)
    assert result.latency_ms >= 0.0
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0


def test_router_predict_empty_query_fallback():
    texts = ["Olá", "Saldo"]
    labels = ["FAST_PATH", "AGENT"]
    router = QueryRouter().fit(texts, labels)

    result = router.predict("")
    assert result.route == "FAST_PATH"
    assert result.confidence == 1.0
    assert result.latency_ms >= 0.0