# Relatório Técnico de Engenharia — Intent Router & Tool Retrieval

## 1. Visão Geral da Arquitetura

O projeto implementa uma camada determinística e estatística de triagem, roteamento e recuperação prévia à invocação de Large Language Models (LLMs). O objetivo primordial da arquitetura é resolver dois dos maiores gargalos operacionais no ciclo de vida de agentes de inteligência artificial em produção: **latência de resposta ponta a ponta** e **custo financeiro acumulado por inferência**.

A solução estrutura-se em três pilares modulares e complementares:

```
                  ┌───────────────────────────────┐
                  │       Requisição Usuário       │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                     [ Pilar 1: Query Router ]
                     (TF-IDF + LogisticRegression)
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
        Rota: FAST_PATH                     Rota: AGENT
                 │                               │
                 ▼                               ▼
       ┌──────────────────┐          [ Pilar 2: Tool Retriever ]
       │ Resposta Local / │          (TF-IDF + Field Weighting +
       │ Template / FAQ   │           Brevity Penalty: Top-2 Tools)
       └──────────────────┘                      │
                 │                               ▼
                 │                   ┌───────────────────────┐
                 │                   │  LLM Especialista +   │
                 │                   │   Execução de Tool    │
                 │                   └───────────┬───────────┘
                 │                               │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                 [ Pilar 3: Evaluation Harness ]
                 (Métricas, Matriz de Confusão,
                  Precision@K, Savings de Custo/Latência)
```

1. **Pilar 1 — Query Router (`candidate_starter/router.py`):** Classificador estatístico supervisionado responsável por inferir se a requisição é puramente informativa/FAQ ou saudação (`FAST_PATH`), passível de resolução determinística local sem custo de token, ou transacional (`AGENT`), demandando o acionamento de ferramentas especializadas e raciocínio contextual.
2. **Pilar 2 — Tool Retriever (`candidate_starter/retrieval.py`):** Mecanismo de busca e ranqueamento vetorial em espaço esparso (*Vector Space Model*) calibrado com ponderação de metadados (*Field Weighting*) e penalização matemática de complexidade de interface (*Brevity Penalty*), responsável por selecionar as $K=2$ ferramentas mais prováveis para o agente contextualizar.
3. **Pilar 3 — Evaluation Harness & Benchmarking (`candidate_starter/harness.py`):** Arcabouço determinístico de avaliação contínua responsável por calcular métricas de roteamento (Acurácia, Matriz de Confusão), recuperação de ferramentas (Precision@K) e mensurar economias de custo (\$ USD) e latência (ms) frente ao baseline monolítico (*Always LLM*).

---

## 2. Decisões Técnicas e Análise de Engenharia

### Pilar 1: Query Router (`candidate_starter/router.py`)

* **Modelo Selecionado:** Pipeline linear composto por `TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode", lowercase=True)` associado a um classificador supervisionado `LogisticRegression(C=1.0, solver="liblinear", random_state=42)`.
* **Fundamentação e Justificativa de Engenharia:**
  * **Eliminação de Regras Heurísticas Frágeis:** Não recorre a regex ou dicionários manuais rígidos de palavras-chave, garantindo que o modelo generalize estatisticamente para variações de grafia e conjugações.
  * **Modelagem de Expressões Compostas:** O suporte a n-gramas de palavras $(1, 2)$ permite diferenciar com precisão comandos compostos de ação transacional (*"meu saldo"*, *"minha fatura"*, *"bloquear cartão"*) de termos institucionais correlatos (*"qual o horário"*, *"como funciona"*).
  * **Calibração de Confiança:** Permite a extração nativa de probabilidades via `predict_proba`, viabilizando o monitoramento operacional de certeza e o estabelecimento de *confidence thresholds* com rotas de *fallback* para supervisão humana.
  * **Eficiência Operacional:** Apresenta tempo médio de inferência de **~1.19 ms** em CPU pura, sem necessidade de alocação de GPUs ou carregamento de modelos fundacionais em memória.

---

### Pilar 2: Tool Retrieval (`candidate_starter/retrieval.py`)

* **Abordagem Oficial Selecionada:** TF-IDF com ponderação de campos (*Field Weighting*) e penalização matemática por prolixidade de assinatura (*Brevity Penalty*):
  $$\text{score} = \frac{\text{similaridade\_cosseno}}{\text{len}(\text{name.split('\_')})^\alpha}, \quad \text{com } \alpha = 1.25$$
* **Comportamento da Penalização de Prolixidade:** A função coíbe que ferramentas com identificadores extensos ou redundantes superem primitivas atômicas essenciais da API, priorizando interfaces modulares canônicas.
* **Avaliação de Abordagens e Análise de Trade-offs:**
  1. **Bi-Encoders Semânticos Densos (`paraphrase-multilingual-MiniLM-L12-v2`):**
     * *Resultado:* 75.0% de Precision@2 e latência acumulada de **910.3 ms** (~40 ms por inferência).
     * *Trade-off:* Adicionou sobrecarga volumosa de dependências externas (`torch`, `sentence-transformers`, ~470 MB de pesos). Em espaço vetorial denso contínuo, verbos de domínio bancário estrito sofreram dispersão semântica (ex.: aproximando cancelamentos de compras de operações de boletos).
  2. **Modelo Probabilístico BM25Okapi (`rank-bm25`):**
     * *Resultado:* 70.0% de Precision@2 e latência acumulada de **75.9 ms**.
     * *Trade-off:* Extremamente rápido em contagens discretas, porém a formulação unigrama pura desconsiderou bigramas cruciais de domínio bancário (*"segunda via"*, *"fatura atual"*, *"saldo disponível"*), rebaixando ferramentas canônicas fora do Top-2.
  3. **TF-IDF com Expansão de Sinônimos / Thesaurus (`re.sub`):**
     * *Resultado:* 100.0% de Precision@2 e latência acumulada de **61.6 ms**.
     * *Trade-off:* Embora atinja pontuação perfeita no dataset fechado de teste, apoia-se em mapeamentos explícitos que representam *overfitting* funcional. Em sistemas produtivos em expansão, dicionários estáticos codificados no repositório geram débito técnico severo e alta manutenção.
  4. **TF-IDF Ponderado Oficial (Decisão Final):**
     * *Resultado:* **80.0% de Precision@2** e latência sub-milissegundo ($<1\text{ ms}$).
     * *Veredito:* Solução matematicamente equilibrada, determinística, aderente às dependências base (`scikit-learn` nativo) e totalmente livre de atalhos frágeis.
* **Análise dos Casos Residuais (4/20):**
  * As 4 consultas em que a ferramenta canônica não figurou no Top-2 refletem redundâncias conceituais no catálogo (por exemplo, a query *"Fiz uma compra que não reconheço, quero contestar"* retornou `contestar_compra_cartao` e `contestar_compra_desconhecida`, enquanto o gabarito fixava a primitiva genérica `estornar_transacao`).
  * Em produção, tal divergência é mitigada através de governança formal de APIs e alinhamento taxonômico entre engenharia e produto.

---

### Pilar 3: Evaluation Harness (`candidate_starter/harness.py`)

O módulo de avaliação foi concebido para calcular métricas formais de forma auditável e determinística:
* **Acurácia e Matriz de Confusão:** Computa a distribuição bidimensional exata entre rotas preditas e observadas.
* **Precision@K:** Razão de acertos onde a ferramenta correta exigida pela intenção está contida entre os $K$ primeiros candidatos sugeridos.
* **Economia Relativa (% Savings):**
  $$\text{cost\_savings\_pct} = \frac{\text{custo\_baseline} - \text{custo\_smart}}{\text{custo\_baseline}} \times 100$$
  $$\text{latency\_savings\_pct} = \frac{\text{latência\_baseline} - \text{latência\_smart}}{\text{latência\_baseline}} \times 100$$

---

## 3. Resultados Consolidados do Benchmark

Os resultados a seguir refletem a execução oficial do pipeline (`python run_case.py`):

| Dimensão Avaliada | Métrica Obtida | Baseline (Always LLM) | Impacto / Eficiência |
| :--- | :--- | :--- | :--- |
| **Acurácia do Router** | **100.0%** (30/30) | — | Decisão binária sem falso positivos ou falsos negativos |
| **Matriz de Confusão** | `FAST_PATH`: 10/10<br>`AGENT`: 20/20 | — | Separação linear perfeita entre FAQ e chamadas de ferramentas |
| **Precision@2 (Retriever)** | **80.0%** (16/20) | — | 80% das ferramentas corretas presentes no Top-2 |
| **Custo Financeiro Total** | **$0.20003** | **$0.90000** | **Economia de 77.8% em custos de tokens** |
| **Latência Acumulada** | **106.1 ms** | **2742.1 ms** | **Economia de 96.1% em tempo de processamento** (~25x mais rápido) |

O relatório estruturado é persistido automaticamente em `reports/candidate_report.json`.

---

## 4. Estrutura do Repositório

```text
.
├── candidate_starter/
│   ├── __init__.py
│   ├── harness.py              # Implementação das métricas do Pilar 3
│   ├── retrieval.py            # Solução oficial do Retriever (Pilar 2 - 80%)
│   └── router.py               # Solução supervisionada do Router (Pilar 1 - 100%)
├── common/
│   ├── data_loader.py          # Utilitários de carregamento de datasets
│   ├── interfaces.py           # Contratos abstratos (BaseRouter, BaseToolRetriever)
│   ├── mock_llm.py             # Simuladores de custo e latência de LLMs
│   └── schemas.py              # Dataclasses de domínio (Tool, RouteResult, RetrievalResult)
├── data/
│   ├── eval_dataset.json       # Dataset oficial de validação e avaliação
│   ├── router_training_data.json # Dataset de treino supervisionado do Router
│   └── tools_registry.json     # Catálogo de ferramentas bancárias
├── reports/
│   └── candidate_report.json   # Relatório gerado pela execução do harness
├── study_cases/                # Arquivamento e documentação de experimentos prévios
│   ├── 01_biencoder_multilingual_minilm_l12.py
│   ├── 02_bm25_unigram_retriever.py
│   ├── 03_tfidf_domain_synonyms_thesaurus.py
│   └── README.md
├── tests/                      # Suite de testes unitários automatizados
│   ├── test_harness.py
│   ├── test_retrieval.py
│   └── test_router.py
├── requirements.txt
├── run_case.py                 # Ponto de entrada oficial de execução do benchmark
└── README.md                   # Documentação técnica de engenharia
```

---

## 5. Instruções de Reprodução e Execução

### Pré-requisitos
* Python 3.10 ou superior instalado.
* Instalação das dependências do projeto:
```bash
pip install -r requirements.txt
```

### Execução do Benchmark Oficial
Para rodar a orquestração ponta a ponta, gerar o relatório técnico no terminal e salvar o artefato `reports/candidate_report.json`:
```bash
python run_case.py
```

### Execução dos Testes Automatizados
Para executar a suite de testes unitários e validar regressões em todos os pilares:
```bash
pytest tests/ -v
```