"""Vector search is great at meaning, weak with exact terms.
keyword search is great with exact terms, weak with synonyms.
"""

def _min_max_normalize(scores: dict) -> dict: 
    """Rescale a dict of scores to the [0,1] range using min-max normalization.
    Vector and keyword scores live on completely diff scales. We cant
    average them raw. Normalizing both to [0,1] first puts them on equal 
    footing so weights mean something.
    """
    if not scores:
        return {} 
    vals = list(scores.values()) 
    lo, hi = min(vals), max(vals) 
    if hi == lo: 
        return {k: 1.0 for k in scores} 
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()} 

def hybrid_search_weighted(
    query: str, 
    vector_results: list,
    keyword_results: list, 
    vector_weight: float = 0.6, 
    keyword_weight: float = 0.4,
    top_k: int = 5) -> list:
    """Combined vector + keyword results via weighted average of normalized scores"""
    vec_raw = {r["chunk_index"]: r["score"] for r in vector_results} 
    key_raw = {r["chunk_index"]: r["score"] for r in keyword_results}
    vec_norm = _min_max_normalize(vec_raw) 
    key_norm = _min_max_normalize(key_raw) 

    all_indices = set(vec_norm.keys()) | set(keyword_results_keys := key_norm.keys()) 
    combined = [] 
    
    for idx in all_indices: 
        v = vec_norm.get(idx, 0.0) 
        b = key_norm.get(idx, 0.0)   
        final = vector_weight * v + keyword_weight * b 
        
        # Fixed: corrected "next" to "text" and added "r" to the conditional check
        text = next((r["text"] for r in vector_results if r["chunk_index"] == idx),
                    next((r["text"] for r in keyword_results if r["chunk_index"] == idx), "")
                   )
        combined.append({
            "chunk_index": idx, 
            "text": text, 
            "score": final, 
            "search_type": "hybrid",
        })
        
    # Fixed: Added sorting and returning the top_k results
    return sorted(combined, key=lambda x: x["score"], reverse=True)[:top_k]


def hybrid_search_rrf(
    vector_results: list, 
    keyword_results: list, 
    top_k: int = 5, 
    k: int = 60) -> list:
    """RRF score = sum over methods of 1/(k+rank). `k` (typically 60) 
    prevents top-ranked items from dominating. This method doesn't need normalization 
    bcz it works on ranks, not raw scores.""" 
    
    rrf_scores: dict = {}  # Fixed: changed from rff_scores to rrf_scores
    
    # Add RRF contributions from each result list 
    for results in (vector_results, keyword_results): 
        for rank, r in enumerate(results, start=1):
            idx = r["chunk_index"] 
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank) 
     
    # Sort by RRF score descending 
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    all_results = {r["chunk_index"]: r for r in vector_results + keyword_results}
    
    return [
        {"chunk_index": idx, "text": all_results[idx]["text"], "score": score, "search_type": "hybrid_rff"} 
        for idx, score in ranked
    ]

# --- Demo: run both searches and combine them. ---
if 'embedding_service' in globals() and embedding_service is not None:
    query = "What are the security and encryption features of Orbital?"
    vec_res = vector_index.search(query, embedding_service, top_k=5)
    key_res = keyword_index.search(query, top_k=5)

    print(f"Query: {query}\n")
    print(f"--- Vector-only top-3 ---")
    for r in vec_res[:3]:
        print(f"  [score {r['score']:.3f}] {r['text'][:100]}...")

    print(f"\n--- Keyword-only top-3 ---")
    for r in key_res[:3]:
        print(f"  [score {r['score']:.2f}] {r['text'][:100]}...")

    print(f"\n--- Hybrid (weighted) top-3 ---")
    hybrid_res = hybrid_search_weighted(query, vec_res, key_res, top_k=3)
    for r in hybrid_res:
        print(f"  [score {r['score']:.3f}] {r['text'][:100]}...")

    print(f"\n--- Hybrid (RRF) top-3 ---")
    rrf_res = hybrid_search_rrf(vec_res, key_res, top_k=3)
    for r in rrf_res:
        print(f"  [score {r['score']:.4f}] {r['text'][:100]}...")
else:
    print("SKIPPED (embedding service not available).")

























