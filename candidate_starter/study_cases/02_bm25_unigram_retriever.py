"""Pilar 2 — Seleção de Tools Relevantes via BM25Okapi.

Implementação com o algoritmo probabilístico BM25 para recuperação textual,
incorporando tokenização normalizada e calibração por granularidade de interface.
"""
import re
import time
import unicodedata
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi

from common.interfaces import BaseToolRetriever
from common.schemas import RetrievalResult, Tool, ToolMatch


def tokenize(text: str) -> List[str]:
    """Normaliza texto (minúsculas, remoção de acentos/pontuação) e extrai tokens."""
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    text = text.lower()
    return re.findall(r"\b\w+\b", text)


class ToolRetriever(BaseToolRetriever):
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        brevity_penalty_factor: float = 1.2,
    ) -> None:
        self._tools: List[Tool] = []
        self._fitted = False
        self._k1 = k1
        self._b = b
        self._brevity_penalty_factor = brevity_penalty_factor
        self._bm25: BM25Okapi = None
        self._length_penalties: np.ndarray = np.array([])

    def _tool_to_tokens(self, tool: Tool) -> List[str]:
        """Gera a lista de tokens do documento dando ênfase à assinatura da interface."""
        name_clean = tool.name.replace("_", " ")
        # Dá peso triplicado ao identificador canónico da ferramenta
        combined_text = f"{name_clean} {name_clean} {name_clean} {tool.category} {tool.description}"
        return tokenize(combined_text)

    def fit(self, tools: List[Tool]) -> "ToolRetriever":
        """Indexa o catálogo construindo o índice BM25 invertido."""
        self._tools = list(tools)
        if not self._tools:
            self._fitted = True
            return self

        tokenized_corpus = [self._tool_to_tokens(t) for t in self._tools]
        self._bm25 = BM25Okapi(tokenized_corpus, k1=self._k1, b=self._b)

        penalties = [
            float(len(t.name.split("_"))) ** self._brevity_penalty_factor
            for t in self._tools
        ]
        self._length_penalties = np.array(penalties, dtype=float)
        self._fitted = True
        return self

    def search(self, query: str, k: int = 2) -> RetrievalResult:
        """Calcula as pontuações BM25 para a query e retorna as top-k tools."""
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de search().")

        start = time.perf_counter()

        query_tokens = tokenize(query)
        if not self._tools or self._bm25 is None or not query_tokens:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return RetrievalResult(matches=[], latency_ms=latency_ms)

        raw_scores = np.array(self._bm25.get_scores(query_tokens))

        # Calibra com a penalidade matemática de prolixidade
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