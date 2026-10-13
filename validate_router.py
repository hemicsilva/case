import json
from pathlib import Path
from candidate_starter.router import QueryRouter

def run_router_validation():
    # 1. Carregar router_training_data.json
    train_path = Path("data/router_training_data.json")
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    texts = [item["text"] for item in train_data]
    labels = [item["label"] for item in train_data]

    # 2. Treinar o Router
    router = QueryRouter()
    router.fit(texts, labels)

    # 3. Carregar eval_dataset.json
    eval_path = Path("data/eval_dataset.json")
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # 4. Avaliar acertos de rota
    hits = 0
    total = len(eval_data)
    latencies = []
    failures = []

    for item in eval_data:
        query = item["query"]
        expected_route = item["expected_route"]

        result = router.predict(query)
        latencies.append(result.latency_ms)

        if result.route == expected_route:
            hits += 1
        else:
            failures.append({
                "query": query,
                "expected": expected_route,
                "got": result.route,
                "confidence": result.confidence,
            })

    accuracy = hits / total
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    print("=" * 60)
    print(f"Total de queries avaliadas no Router: {total}")
    print(f"Acurácia de Roteamento: {accuracy:.2%} ({hits}/{total})")
    print(f"Latência Média por Inferência: {avg_latency:.4f} ms")
    print("=" * 60)

    if failures:
        print("\nFalhas de Roteamento encontradas:")
        for fail in failures:
            print(f"- Query: '{fail['query']}'")
            print(f"  Esperado: {fail['expected']} | Retornado: {fail['got']} (Confiança: {fail['confidence']:.2%})\n")
    else:
        print("\nTodas as decisões de rota correspondem exatamente ao ground truth!")

if __name__ == "__main__":
    run_router_validation()