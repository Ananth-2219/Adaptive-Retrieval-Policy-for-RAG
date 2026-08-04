import unittest
from src.retrieval.vector_search import vector_search
from src.retrieval.bm25_search import bm25_search
from src.retrieval.hybrid_rrf import hybrid_rrf
from src.retrieval.multi_hop import multi_hop_search
from src.retrieval.reranker import rerank
from src.utils.schemas import RetrievalChunk
import config  # Assumes config handles test data paths

class TestRetrievalPipelines(unittest.TestCase):
    def setUp(self):
        # Load a small test dataset (e.g., 50 SQuAD questions)
        self.test_queries = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms",
            "Who wrote the play 'Romeo and Juliet'?"
        ]
        self.top_k = 3  # Test with top 3 results

    def test_vector_search(self):
        for query in self.test_queries:
            results = vector_search(query, self.top_k)
            self.assertEqual(len(results), self.top_k)
            for chunk in results:
                self.assertIn("vector", chunk.source_method)
                self.assertIsInstance(chunk.score, float)

    def test_bm25_search(self):
        for query in self.test_queries:
            results = bm25_search(query, self.top_k)
            self.assertEqual(len(results), self.top_k)
            for chunk in results:
                self.assertIn("bm25", chunk.source_method)
                self.assertIsInstance(chunk.score, float)

    def test_hybrid_rrf(self):
        for query in self.test_queries:
            results = hybrid_rrf(query, self.top_k)
            self.assertEqual(len(results), self.top_k)
            for chunk in results:
                self.assertIn("hybrid", chunk.source_method)
                self.assertIsInstance(chunk.score, float)

    def test_multi_hop(self):
        for query in self.test_queries:
            results = multi_hop_search(query, self.top_k)
            self.assertEqual(len(results), self.top_k)
            for chunk in results:
                self.assertIn("vector", chunk.source_method)  # Currently uses vector search
                self.assertIsInstance(chunk.score, float)

    def test_reranker(self):
        # Test with hybrid RRF results
        query = "Explain quantum computing"
        hybrid_results = hybrid_rrf(query, self.top_k)
        reranked = rerank(query, hybrid_results, self.top_k)
        self.assertEqual(len(reranked), self.top_k)
        for chunk in reranked:
            self.assertIn("reranker", chunk.source_method)
            self.assertIsInstance(chunk.score, float)

if __name__ == '__main__':
    unittest.main()