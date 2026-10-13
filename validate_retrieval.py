import json
from pathlib import Path
from common.schemas import Tool
from candidate_starter.retrieval import ToolRetriever

def run_retrieval_validation():
    # 1. Carregar tools_registry.json
    tools_path = Path("data/tools_registry.json")
    with open(tools_path, "r", encoding="utf-8") as f:
        raw_tools = json.load(f)
    
    tools = [
        Tool(name=t["name"], description=t["description"], category=t["category"])
        for t in raw_tools
    ]

    # 2. Inicializar e indexar o Retriever
    retriever = ToolRetriever()
    retriever.fit(tools)

    # 3. Carregar eval_dataset.json
    eval_path = Path("data/eval_dataset.json")
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # 4. Avaliar Precision@K (apenas para casos onde uma tool é esperada)
    agent_evals = [item for item in eval_data if item.get("expected_tool") is not None]
    
    hits_at_1 = 0
    hits_at_2 = 0
    failures = []

    for item in agent_evals:
        query = item["query"]
        expected = item["expected_tool"]
        
        # Executa a busca com top-2
        result = retriever.search(query, k=2)
        retrieved_names = [m.name for m in result.matches]

        hit_1 = len(retrieved_names) > 0 and retrieved_names[0] == expected
        hit_2 = expected in retrieved_names

        if hit_1:
            hits_at_1 += 1
        if hit_2:
            hits_at_2 += 1
        else:
            failures.append({
                "query": query,
                "expected": expected,
                "got": retrieved_names
            })

    total = len(agent_evals)
    precision_at_1 = hits_at_1 / total
    precision_at_2 = hits_at_2 / total

    print("=" * 60)
    print(f"Total de queries de ferramentas avaliadas: {total}")
    print(f"Precision@1: {precision_at_1:.2%} ({hits_at_1}/{total})")
    print(f"Precision@2 (K=2): {precision_at_2:.2%} ({hits_at_2}/{total})")
    print("=" * 60)

    if failures:
        print("\nCasos onde a tool esperada NÃO esteve no top-2:")
        for fail in failures:
            print(f"- Query: '{fail['query']}'")
            print(f"  Esperado: {fail['expected']}")
            print(f"  Retornado: {fail['got']}\n")
    else:
        print("\nTodas as tools esperadas foram encontradas no top-2!")

if __name__ == "__main__":
    run_retrieval_validation()