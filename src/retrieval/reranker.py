from typing import List
from sentence_transformers import CrossEncoder
from src.utils.schemas import RetrievalChunk

# Load the cross-encoder model
# Defaulting to the one specified in CLAUDE.md
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, chunks: List[RetrievalChunk], top_k: int = 5) -> List[RetrievalChunk]:
    """
    Rerank retrieved chunks using a cross-encoder model.
    
    Args:
        query: The original user query.
        chunks: A list of RetrievalChunk objects to rerank.
        top_k: Number of reranked results to return.
        
    Returns:
        List of RetrievalChunk objects sorted by reranker score.
    """
    if not chunks:
        return []

    # Prepare pairs for the cross-encoder: [[query, text1], [query, text2],...]
    pairs = [[query, chunk.text] for chunk in chunks]
    
    # Predict scores
    scores = reranker.predict(pairs)
    
    # Update chunks with new scores and sort
    for i, score in enumerate(scores):
        chunks[i].score = float(score)
        chunks[i].source_method = "reranker"
    
    chunks.sort(key=lambda x: x.score, reverse=True)
    
    return chunks[:top_k]