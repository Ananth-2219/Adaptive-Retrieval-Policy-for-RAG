from pydantic import BaseModel
from typing import Dict, Any

class RetrievalChunk(BaseModel):
    chunk_id: str
    text: str
    score: float  # method-specific relevance score, NOT normalized across methods
    source_method: str  # "vector" | "bm25" | "hybrid" | "multihop"
    metadata: Dict[str, Any]  # doc_id, source dataset, any hop info for multi-hop

class RouterResult(BaseModel):
    query: str
    predicted_intent: str  # "factual" | "comparative" | "definitional" | "multihop"
    confidence: float
    pipeline_used: str