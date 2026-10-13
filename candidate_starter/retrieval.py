"""Pilar 2 — Seleção de Tools Relevantes.

Recuperação de tools combinando TF-IDF ponderado por campos e
penalização por complexidade/prolixidade de assinatura de API.
"""
import time
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common.interfaces import BaseToolRetriever
from common.schemas import RetrievalResult, Tool, ToolMatch


class ToolRetriever(BaseToolRetriever):
    def __init__(self, brevity_penalty_factor: float = 1.40) -> None:
        self._tools: List[Tool] = []
        self._fitted = False
        self._penalty_factor = brevity_penalty_factor

        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True
        )
        self._tool_matrix = None
        self._length_penalties: np.ndarray = np.array([])

    def _tool_to_document(self, tool: Tool) -> str:
        """Monta o documento indexável ponderando o nome canónico da interface."""
        name_tokens = tool.name.replace("_", " ")
        # Dá peso proeminente ao nome (verbo + entidade) em relação à descrição
        return f"{name_tokens} {name_tokens} {name_tokens} {tool.category} {tool.description}"

    def fit(self, tools: List[Tool]) -> "ToolRetriever":
        """Indexa o catálogo calculando os vetores TF-IDF e os fatores de penalização."""
        self._tools = list(tools)
        if not self._tools:
            self._fitted = True
            return self

        corpus = [self._tool_to_document(t) for t in self._tools]
        self._tool_matrix = self._vectorizer.fit_transform(corpus)

        # Penalização puramente matemática: interfaces atómicas/canónicas
        # (ex: 'consultar_saldo' -> 2 tokens) recebem penalidade mínima.
        # Interfaces compostas/prolixas (ex: 5-8 tokens) sofrem decaimento.
        penalties = [
            float(len(t.name.split("_"))) ** self._penalty_factor
            for t in self._tools
        ]
        self._length_penalties = np.array(penalties, dtype=float)
        self._fitted = True
        return self

    def search(self, query: str, k: int = 2) -> RetrievalResult:
        """Executa a busca vetorial e retorna as top-k tools ordenadas."""
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de search().")

        start = time.perf_counter()

        if not self._tools or self._tool_matrix is None or not query.strip():
            latency_ms = (time.perf_counter() - start) * 1000.0
            return RetrievalResult(matches=[], latency_ms=latency_ms)

        query_vec = self._vectorizer.transform([query])
        raw_scores = cosine_similarity(query_vec, self._tool_matrix)[0]

        # Ajusta scores ponderando relevância vs prolixidade da tool
        calibrated_scores = raw_scores / self._length_penalties
        top_indices = np.argsort(calibrated_scores)[::-1][:k]

        matches: List[ToolMatch] = [
            ToolMatch(
                name=self._tools[idx].name,
                score=float(calibrated_scores[idx]),
            )
            for idx in top_indices
        ]

        latency_ms = (time.perf_counter() - start) * 1000.0
        return RetrievalResult(matches=matches, latency_ms=latency_ms)