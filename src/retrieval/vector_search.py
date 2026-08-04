from typing import List
from sentence_transformers import SentenceTransformer
import chromadb
from src.utils.schemas import RetrievalChunk
import config  # Assumes config.py handles model paths and ChromaDB setup

# Load models and client
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # CPU-friendly
client = chromadb.PersistentClient(path=config.VECTOR_STORE_PATH)  # Path from configs/
index = client.get_or_create_collection("rag_chunks")

def vector_search(query: str, top_k: int = 5) -> List[RetrievalChunk]:
    """
    Perform vector similarity search on ChromaDB for the given query.
    
    Args:
        query: Input query string
        top_k: Number of top results to return
        
    Returns:
        List of RetrievalChunk objects sorted by relevance score
    """
    # Generate query embedding
    query_embedding = embedding_model.encode(query, convert_to_tensor=False)
    
    # Search ChromaDB
    results = index.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Format results into RetrievalChunk objects
    chunks = []
    for i in range(len(results["ids"][0])):
        chunk_id = results["ids"][0][i]
        text = results["metadatas"][0][i].get("text", "")
        score = results["distances"][0][i]
        source_method = "vector"
        metadata = results["metadatas"][0][i]
        
        chunks.append(RetrievalChunk(
            chunk_id=chunk_id,
            text=text,
            score=score,
            source_method=source_method,
            metadata=metadata
        ))
    
    return sorted(chunks, key=lambda x: x.score, reverse=True)