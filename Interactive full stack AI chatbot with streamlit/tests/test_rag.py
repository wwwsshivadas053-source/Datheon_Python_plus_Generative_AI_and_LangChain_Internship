"""
Unit tests for RAGEngine.
"""

import unittest
from core.rag_engine import RAGEngine


class TestRAGEngine(unittest.TestCase):

    def setUp(self):
        self.rag = RAGEngine(chunk_size=300, chunk_overlap=50)

    def test_initial_state(self):
        self.assertFalse(self.rag.is_indexed())
        self.assertEqual(self.rag.get_document_count(), 0)
        self.assertEqual(self.rag.get_total_chunks(), 0)

    def test_text_ingestion_and_indexing(self):
        sample_doc = (
            "Streamlit is an open-source Python framework that allows developers to create "
            "interactive web applications for machine learning and data science quickly. "
            "It features session state management and native streaming chat interfaces.\n\n"
            "Retrieval-Augmented Generation (RAG) grounds language model outputs in verifiable "
            "external documentation, preventing hallucinations and ensuring domain accuracy."
        )

        chunks_count = self.rag.ingest_file("sample.txt", sample_doc.encode("utf-8"))
        self.assertGreater(chunks_count, 0)
        self.assertTrue(self.rag.is_indexed())
        self.assertEqual(self.rag.get_document_count(), 1)
        self.assertEqual(self.rag.get_total_chunks(), chunks_count)

    def test_search_relevancy(self):
        doc1 = "Python programming language is versatile and used for artificial intelligence, web development, and data analysis."
        doc2 = "Marine biology studies organisms in oceans and saltwater environments such as coral reefs and deep sea trenches."

        self.rag.ingest_file("python.txt", doc1.encode("utf-8"))
        self.rag.ingest_file("marine.txt", doc2.encode("utf-8"))

        # Search for Python query
        results = self.rag.search("Python AI programming", top_k=2)
        self.assertGreater(len(results), 0)
        top_match = results[0]
        self.assertEqual(top_match.chunk.source_name, "python.txt")
        self.assertGreater(top_match.score, 0.0)

        # Search for marine biology query
        marine_results = self.rag.search("coral reefs ocean biology", top_k=2)
        self.assertGreater(len(marine_results), 0)
        self.assertEqual(marine_results[0].chunk.source_name, "marine.txt")

    def test_build_rag_context(self):
        doc = "Project Nexus deploys on port 8501 using Streamlit and sublinear TF-IDF retrieval."
        self.rag.ingest_file("nexus.txt", doc.encode("utf-8"))

        context, sources = self.rag.build_rag_context("What port does Nexus use?", top_k=1)
        self.assertIn("nexus.txt", context)
        self.assertIn("8501", context)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["source"], "nexus.txt")

    def test_clear_engine(self):
        doc = "Testing clear functionality."
        self.rag.ingest_file("test.txt", doc.encode("utf-8"))
        self.assertTrue(self.rag.is_indexed())

        self.rag.clear()
        self.assertFalse(self.rag.is_indexed())
        self.assertEqual(self.rag.get_document_count(), 0)
        self.assertEqual(self.rag.get_total_chunks(), 0)


if __name__ == "__main__":
    unittest.main()

