# Casos de Estudo e Experimentos Comparativos (Tool Retrieval)

Este diretório arquiva as abordagens alternativas desenvolvidas e avaliadas para o **Pilar 2 (Tool Retrieval)**, documentando os seus trade-offs arquiteturais, dependências, métricas no evaluation harness e os motivos que determinaram a escolha da solução oficial (`candidate_starter/retrieval.py`).

---

## Tabela Comparativa de Desempenho

| Estudo / Ficheiro | Paradigma | Dependências | Precision@2 | Latência Acumulada | Parecer Técnico |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **01_biencoder_multilingual_minilm_l12.py** | Denso (Transformer L12) | `sentence-transformers`, `torch` (~470 MB) | 75.0% | 910.3 ms | **Descartado**: Latência elevada (~40 ms/query) e confusão semântica em verbos operacionais fechados. |
| **02_bm25_unigram_retriever.py** | Probabilístico (BM25Okapi) | `rank-bm25` | 70.0% | 75.9 ms | **Descartado**: Rápido, mas penalizado pela ausência de bigramas nos termos bancários compostos. |
| **03_tfidf_domain_synonyms_thesaurus.py** | Léxico + Thesaurus Manual | Nenhuma (`scikit-learn` nativo) | 100.0% | 61.6 ms | **Arquivado**: Atinge precisão total no benchmark fechado, mas introduz acoplamento manual frágil (overfitting). |
| **candidate_starter/retrieval.py (Oficial)** | TF-IDF + Field Weighting + Brevity Penalty | Nenhuma (`scikit-learn` nativo) | **80.0%** | **106.1 ms** | **Vencedor**: Robusto, matematicamente fundamentado, determinístico (<1 ms) e sem regras estáticas. |

---

## 1. Bi-Encoder Multilíngue (`01_biencoder_multilingual_minilm_l12.py`)
* **Modelo**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
* **Precision@2**: 75.0% (15/20 acertos).
* **Latência Total**: 910.3 ms (economia de latência de 66.9% face ao baseline).
* **Análise**: O espaço vetorial denso contínuo aproxima conceitos conexos (ex.: estorno vs. cancelamento genérico), gerando dispersão em operações bancárias estritas. Além disso, o download do modelo e a inferência em CPU aumentam a pegada de memória e a latência de execução.

## 2. BM25Okapi (`02_bm25_unigram_retriever.py`)
* **Modelo**: Algoritmo probabilístico BM25 com calibração de prolixidade.
* **Precision@2**: 70.0% (14/20 acertos).
* **Latência Total**: 75.9 ms (economia de latência de 97.2% face ao baseline).
* **Análise**: Demonstrou elevada eficiência computacional, contudo a formulação unigrama ignorou expressões bancárias essenciais de dois termos (*"segunda via"*, *"fatura atual"*, *"saldo disponível"*), fazendo cair para fora do Top-2 ferramentas com especificidade composta.

## 3. TF-IDF com Normalização Léxica e Sinónimos (`03_tfidf_domain_synonyms_thesaurus.py`)
* **Modelo**: TF-IDF (n-gramas 1-2) associado a mapeamento manual via regex (`SYNONYM_REPLACEMENTS`).
* **Precision@2**: 100.0% (20/20 acertos).
* **Latência Total**: 61.6 ms (economia de latência de 97.8% face ao baseline).
* **Análise**: Embora represente o teto numérico de avaliação, esta abordagem foi arquivada por apresentar risco de sobreajuste (*overfitting*). Em cenários produtivos com catálogos dinâmicos, listas estáticas de substituição criam dívida técnica e acoplamento frágil.

---

## Conclusão de Arquitetura
A abordagem adotada no pipeline de produção (`candidate_starter/retrieval.py`) fixou-se nos **80.0% de Precision@2** via TF-IDF ponderado e penalização por complexidade de interface, garantindo uma solução limpa, generalizável e sem atalhos manuais.