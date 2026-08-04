from typing import List
from rank_bm25 import BM25Okapi
import json
from src.utils.schemas import RetrievalChunk
import config  # Assumes config.py handles data paths

def load_corpus() -> List[dict]:
    """
    Load the pre-processed corpus from the data directory.
    
    Returns:
        List of document dicts with 'text' and 'metadata' fields
    """
    with open(config.PROCESSED_DATA_PATH, "r") as f:
        return json.load(f)

def bm25_search(query: str, top_k: int = 5) -> List[RetrievalChunk]:
    """
    Perform BM25 keyword search on the corpus for the given query.
    
    Args:
        query: Input query string
        top_k: Number of top results to return
        
    Returns:
        List of RetrievalChunk objects sorted by BM25 relevance score
    """
    # Load corpus
    corpus = load_corpus()
    
    # Tokenize documents
    tokenized_docs = [doc["text"].split() for doc in corpus]
    
    # Build BM25 index
    bm25 = BM25Okapi(tokenized_docs)
    
    # Tokenize query
    tokenized_query = query.split()
    
    # Get scores
    scores = bm25.get_scores(tokenized_query)
    
    # Get top-k results
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    # Format results into RetrievalChunk objects
    chunks = []
    for idx in top_indices:
        doc = corpus[idx]
        chunk = RetrievalChunk(
            chunk_id=doc.get("chunk_id", f"doc_{idx}"),
            text=doc["text"],
            score=float(scores[idx]),
            source_method="bm25",
            metadata=doc.get("metadata", {})
        )
        chunks.append(chunk)
    
    return chunks