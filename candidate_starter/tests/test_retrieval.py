"""Testes unitários para o Pilar 2 — ToolRetriever."""
import pytest
from candidate_starter.retrieval import ToolRetriever
from common.schemas import RetrievalResult, Tool


@pytest.fixture
def sample_tools():
    return [
        Tool(
            name="consultar_saldo",
            description="Consulta o saldo disponível da conta corrente ou poupança",
            category="financeiro",
            parameters={},
        ),
        Tool(
            name="consultar_fatura",
            description="Consulta o valor e vencimento da fatura do cartão",
            category="cartao",
            parameters={},
        ),
        Tool(
            name="alterar_endereco",
            description="Atualiza o endereço residencial cadastrado no banco",
            category="cadastro",
            parameters={},
        ),
    ]


def test_retriever_raises_runtime_error_when_search_before_fit():
    retriever = ToolRetriever()
    with pytest.raises(RuntimeError, match="Chame fit() antes de search()"):
        retriever.search("Qual meu saldo?", k=2)


def test_retriever_search_schema_and_ordering(sample_tools):
    retriever = ToolRetriever().fit(sample_tools)
    result = retriever.search("Quero ver meu saldo em conta", k=2)

    assert isinstance(result, RetrievalResult)
    assert isinstance(result.latency_ms, float)
    assert result.latency_ms >= 0.0
    assert len(result.matches) == 2

    # Verifica se os scores estão ordenados de forma decrescente
    scores = [match.score for match in result.matches]
    assert scores == sorted(scores, reverse=True)

    # A ferramenta mais relevante esperada deve ser consultar_saldo
    assert result.matches[0].name == "consultar_saldo"


def test_retriever_search_respects_k_parameter(sample_tools):
    retriever = ToolRetriever().fit(sample_tools)
    result_k1 = retriever.search("fatura cartão", k=1)
    result_k3 = retriever.search("fatura cartão", k=3)

    assert len(result_k1.matches) == 1
    assert len(result_k3.matches) == 3


def test_retriever_empty_query(sample_tools):
    retriever = ToolRetriever().fit(sample_tools)
    result = retriever.search("", k=2)

    assert len(result.matches) == 0
    assert result.latency_ms >= 0.0