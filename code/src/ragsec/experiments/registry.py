METHOD_REGISTRY: dict[str, str] = {
    "no_rag": "No RAG baseline",
    "dense": "Vanilla Dense RAG",
    "bm25": "BM25 RAG",
    "hybrid": "Hybrid RAG",
    "behavior": "Security-Behavior RAG",
    "contrastive": "Contrastive RAG",
    "behavior_contrastive": "Behavior + Contrastive RAG",
}
