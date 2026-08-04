from typing import List
from src.utils.schemas import RetrievalChunk
from src.retrieval.vector_search import vector_search

def multi_hop_search(query: str, top_k: int = 5) -> List[RetrievalChunk]:
    """
    Perform multi-hop retrieval by decomposing the query into sub-queries,
    retrieving results for each, and combining them.
    
    Args:
        query: Input query string
        top_k: Number of top results to return
        
    Returns:
        List of RetrievalChunk objects sorted by relevance
    """
    # Decompose query into sub-queries (mock implementation for MVP)
    # In reality, this would use the LLM via Ollama to decompose the query
    sub_queries = [query]  # For MVP, assume single sub-query
    
    all_chunks = []
    for sub_query in sub_queries:
        # Retrieve results for each sub-query using vector search
        chunks = vector_search(sub_query, top_k=top_k)
        all_chunks.extend(chunks)
    
    # Sort all chunks by score and return top_k
    all_chunks.sort(key=lambda x: x.score, reverse=True)
    return all_chunks[:top_k]