"""Pilar 1 — Router de Queries.

Classificador supervisionado para distinguir entre:
  - 'FAST_PATH': queries informativas/FAQ e saudações (resposta local sem LLM).
  - 'AGENT': queries transacionais que requerem execução de ferramentas.
"""
import time
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from common.interfaces import BaseRouter
from common.schemas import RouteResult


class QueryRouter(BaseRouter):
    def __init__(self) -> None:
        self._fitted = False
        # Pipeline combinando representação por n-gramas e regressão logística calibrada
        self._pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                        strip_accents="unicode",
                        lowercase=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        solver="liblinear",
                        random_state=42,
                    ),
                ),
            ]
        )

    def fit(self, texts: List[str], labels: List[str]) -> "QueryRouter":
        """Treina o router com o dataset supervisionado."""
        if not texts or not labels:
            raise ValueError("Os parâmetros texts e labels não podem ser vazios.")

        self._pipeline.fit(texts, labels)
        self._fitted = True
        return self

    def predict(self, query: str) -> RouteResult:
        """Classifica a query, medindo a latência e a probabilidade de confiança."""
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de predict().")

        start = time.perf_counter()

        if not query or not query.strip():
            latency_ms = (time.perf_counter() - start) * 1000.0
            return RouteResult(
                route="FAST_PATH",
                latency_ms=latency_ms,
                confidence=1.0,
            )

        # Predição de rota e probabilidade associada
        route = str(self._pipeline.predict([query])[0])
        probabilities = self._pipeline.predict_proba([query])[0]
        confidence = float(np.max(probabilities))

        latency_ms = (time.perf_counter() - start) * 1000.0
        return RouteResult(
            route=route,
            latency_ms=latency_ms,
            confidence=confidence,
        )