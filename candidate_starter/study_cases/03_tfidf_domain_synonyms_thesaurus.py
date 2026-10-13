"""Pilar 2 — Seleção de Tools Relevantes.

Indexação via TF-IDF ponderada com normalização léxica de sinónimos de domínio,
resolução de intenções de actualização cadastral e atenuação de prolixidade.
"""
import re
import time
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common.interfaces import BaseToolRetriever
from common.schemas import RetrievalResult, Tool, ToolMatch

PORTUGUESE_STOPWORDS = [
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com", "nao",
    "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como", "mas",
    "ao", "ele", "das", "seu", "sua", "ou", "quando", "muito", "nos", "ja",
    "eu", "meu", "minha", "me", "te", "esse", "essa", "este", "esta", "pra",
    "voce", "voces", "quero", "preciso", "gostaria", "favor"
]

SYNONYM_REPLACEMENTS = [
    # Reembolso / Estorno
    (r"\b(dinheiro de volta|devolve meu dinheiro|devolucao)\b", "estorno estornar"),
    (r"\b(contestar|contestacao|estorno|estornar)\b", "estornar contestacao"),
    # Endereço e CEP
    (r"\b(cep|residencia|morada|mudei de casa)\b", "endereco"),
    # Consulta de saldo / Saldo disponível
    (r"\b(quanto eu tenho|quanto tenho|tenho disponivel|valor disponivel)\b", "saldo conta"),
    # Atualização de e-mail / cadastro
    (r"\be-mail\b", "email"),
    (r"\b(mudar|mudei|mudanca|trocar|troquei|troca)\b", "alterar atualizar"),
]


def normalize_banking_text(text: str) -> str:
    """Normaliza grafias e alinha variantes morfológicas com termos canónicos."""
    lower_text = text.lower()
    for pattern, replacement in SYNONYM_REPLACEMENTS:
        lower_text = re.sub(pattern, replacement, lower_text)
    return lower_text


class ToolRetriever(BaseToolRetriever):
    def __init__(self) -> None:
        self._tools: List[Tool] = []
        self._fitted = False
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
            stop_words=PORTUGUESE_STOPWORDS,
        )
        self._tool_matrix = None
        self._penalties: np.ndarray = np.array([])

    def _tool_to_text(self, tool: Tool) -> str:
        """Gera a representação textual indexável da ferramenta com ênfase no identificador."""
        name_clean = tool.name.replace("_", " ")
        doc_raw = f"{name_clean} " * 5 + f"{tool.category} {tool.description}"
        return normalize_banking_text(doc_raw)

    def fit(self, tools: List[Tool]) -> "ToolRetriever":
        """Indexa o catálogo calculando os pesos e as penalizações de prolixidade."""
        self._tools = list(tools)
        if not self._tools:
            self._fitted = True
            return self

        corpus = [self._tool_to_text(tool) for tool in self._tools]
        self._tool_matrix = self._vectorizer.fit_transform(corpus)

        penalties = []
        for tool in self._tools:
            num_tokens = len(tool.name.split("_"))
            # Penalização suave para atenuar ferramentas clonadas excessivamente longas
            penalty = float(num_tokens) ** 1.3
            penalties.append(penalty)

        self._penalties = np.array(penalties, dtype=float)
        self._fitted = True
        return self

    def search(self, query: str, k: int = 2) -> RetrievalResult:
        """Retorna as top-k ferramentas candidatas ordenadas por relevância."""
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de search().")

        start = time.perf_counter()

        if not self._tools or self._tool_matrix is None or not query.strip():
            latency_ms = (time.perf_counter() - start) * 1000.0
            return RetrievalResult(matches=[], latency_ms=latency_ms)

        normalized_query = normalize_banking_text(query)
        query_vec = self._vectorizer.transform([normalized_query])
        raw_scores = cosine_similarity(query_vec, self._tool_matrix)[0]

        calibrated_scores = raw_scores / self._penalties
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