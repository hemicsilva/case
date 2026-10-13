"""Pilar 2 — Seleção de Tools Relevantes via Bi-Encoder.

Utiliza Sentence Transformers pré-treinado multilíngue para gerar
embeddings semânticos densos das ferramentas e das queries dos usuários.
"""
import time
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from common.interfaces import BaseToolRetriever
from common.schemas import RetrievalResult, Tool, ToolMatch


class ToolRetriever(BaseToolRetriever):
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        brevity_penalty_factor: float = 1.15,
    ) -> None:
        self._tools: List[Tool] = []
        self._fitted = False
        self._brevity_penalty_factor = brevity_penalty_factor
        self._model = SentenceTransformer(model_name)
        self._tool_embeddings: np.ndarray = np.array([])
        self._length_penalties: np.ndarray = np.array([])

    def _tool_to_text(self, tool: Tool) -> str:
        """Gera a descrição semântica da ferramenta enfatizando a intenção nuclear."""
        name_clean = tool.name.replace("_", " ")
        # Dá ênfase ao nome canônico combinado com a categoria e descrição
        return f"{name_clean}. {tool.category}: {tool.description}"

    def fit(self, tools: List[Tool]) -> "ToolRetriever":
        """Gera e armazena os embeddings densos de todas as tools do catálogo."""
        self._tools = list(tools)
        if not self._tools:
            self._fitted = True
            return self

        corpus = [self._tool_to_text(t) for t in self._tools]

        # Gera embeddings normalizados (norma L2 = 1.0)
        self._tool_embeddings = self._model.encode(
            corpus,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Penalização suave para coibir identificadores artificiais prolixos
        # mantendo a prioridade em APIs atômicas canônicas
        penalties = [
            float(len(t.name.split("_"))) ** self._brevity_penalty_factor
            for t in self._tools
        ]
        self._length_penalties = np.array(penalties, dtype=float)

        self._fitted = True
        return self

    def search(self, query: str, k: int = 2) -> RetrievalResult:
        """Calcula a similaridade semântica densa e retorna o top-k."""
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de search().")

        start = time.perf_counter()

        if not self._tools or len(self._tool_embeddings) == 0 or not query.strip():
            latency_ms = (time.perf_counter() - start) * 1000.0
            return RetrievalResult(matches=[], latency_ms=latency_ms)

        # Embedding denso da consulta
        query_embedding = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Similaridade de cossenos direta entre vetores normalizados
        raw_scores = cosine_similarity(query_embedding, self._tool_embeddings)[0]

        # Ponderação por granularidade de interface
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