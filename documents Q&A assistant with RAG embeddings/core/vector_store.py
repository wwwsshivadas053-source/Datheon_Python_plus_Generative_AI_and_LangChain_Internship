import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

from config import CHROMA_DIR, FAISS_DIR, Config


class VectorStoreService:
    """
    Unified Vector Store Manager supporting both ChromaDB and FAISS.
    Allows seamless switching, persistence, similarity search with scores,
    and index inspection.
    """

    def __init__(self, store_type: str = "faiss", embeddings: Optional[Embeddings] = None):
        self.store_type = store_type.lower()
        self.embeddings = embeddings
        self.chroma_instance: Optional[Chroma] = None
        self.faiss_instance: Optional[FAISS] = None
        self.indexed_files: Dict[str, Dict[str, Any]] = {}
        self._init_store()

    def set_embeddings(self, embeddings: Embeddings):
        """Update active embeddings and re-initialize vector stores."""
        self.embeddings = embeddings
        self._init_store()

    def _init_store(self):
        """Initializes or loads existing ChromaDB or FAISS indices."""
        if not self.embeddings:
            return

        if self.store_type == "chroma":
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            self.chroma_instance = Chroma(
                collection_name=Config.COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=str(CHROMA_DIR)
            )

        elif self.store_type == "faiss":
            FAISS_DIR.mkdir(parents=True, exist_ok=True)
            faiss_file = FAISS_DIR / "index.faiss"
            if faiss_file.exists():
                try:
                    self.faiss_instance = FAISS.load_local(
                        folder_path=str(FAISS_DIR),
                        embeddings=self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    # Verify dimension compatibility
                    test_dim = len(self.embeddings.embed_query("dim_test"))
                    if self.faiss_instance.index.d != test_dim:
                        print(f"[VectorStoreService] Reinitializing FAISS due to dimension change: {self.faiss_instance.index.d} -> {test_dim}")
                        self.faiss_instance = None
                        shutil.rmtree(str(FAISS_DIR), ignore_errors=True)
                        FAISS_DIR.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    print(f"[VectorStoreService] Note: Initializing new FAISS index ({e})")
                    self.faiss_instance = None
            else:
                self.faiss_instance = None

    def switch_store(self, new_store_type: str, all_current_documents: Optional[List[Document]] = None):
        """Switches between ChromaDB and FAISS on the fly."""
        new_store_type = new_store_type.lower()
        if new_store_type not in ["faiss", "chroma"]:
            raise ValueError(f"Unknown store type '{new_store_type}'. Choose 'faiss' or 'chroma'.")

        self.store_type = new_store_type
        self._init_store()

        if all_current_documents:
            self.add_documents(all_current_documents)

    def add_documents(self, documents: List[Document]) -> int:
        """Adds chunked documents to the active vector store."""
        if not documents:
            return 0

        # Track indexed documents metadata
        for doc in documents:
            file_name = doc.metadata.get("file_name", "unknown")
            page = doc.metadata.get("page", 1)
            if file_name not in self.indexed_files:
                self.indexed_files[file_name] = {
                    "file_name": file_name,
                    "pages": set(),
                    "total_chunks": 0
                }
            self.indexed_files[file_name]["pages"].add(page)
            self.indexed_files[file_name]["total_chunks"] += 1

        if self.store_type == "chroma":
            if not self.chroma_instance:
                self._init_store()
            self.chroma_instance.add_documents(documents)
            return len(documents)

        elif self.store_type == "faiss":
            if self.faiss_instance is None:
                self.faiss_instance = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
            else:
                self.faiss_instance.add_documents(documents)

            # Persist FAISS index to disk
            FAISS_DIR.mkdir(parents=True, exist_ok=True)
            self.faiss_instance.save_local(str(FAISS_DIR))
            return len(documents)

        return 0

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Retrieves top-k most relevant chunks along with distance/similarity scores.
        Converts distance metrics into normalized relevance scores [0.0 - 1.0].
        """
        results: List[Tuple[Document, float]] = []

        if self.store_type == "chroma" and self.chroma_instance:
            raw_results = self.chroma_instance.similarity_search_with_score(query, k=k)
            for doc, distance in raw_results:
                # Chroma returns L2 squared or cosine distance; normalize to [0, 1]
                relevance = round(1.0 / (1.0 + float(distance)), 4)
                results.append((doc, relevance))

        elif self.store_type == "faiss" and self.faiss_instance:
            raw_results = self.faiss_instance.similarity_search_with_score(query, k=k)
            for doc, distance in raw_results:
                # FAISS L2 distance -> similarity score
                relevance = round(1.0 / (1.0 + float(distance)), 4)
                results.append((doc, relevance))

        return results

    def clear(self):
        """Clears all stored indices and reset tracker."""
        self.indexed_files.clear()

        # Reset Chroma
        try:
            if self.chroma_instance:
                self.chroma_instance.delete_collection()
            if CHROMA_DIR.exists():
                shutil.rmtree(str(CHROMA_DIR), ignore_errors=True)
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[VectorStoreService] Chroma clear note: {e}")

        # Reset FAISS
        try:
            self.faiss_instance = None
            if FAISS_DIR.exists():
                shutil.rmtree(str(FAISS_DIR), ignore_errors=True)
            FAISS_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[VectorStoreService] FAISS clear note: {e}")

        self._init_store()

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics about current vector store and indexed files."""
        total_chunks = sum(item["total_chunks"] for item in self.indexed_files.values())
        files_summary = [
            {
                "file_name": k,
                "pages_indexed": len(v["pages"]),
                "chunk_count": v["total_chunks"]
            }
            for k, v in self.indexed_files.items()
        ]

        return {
            "active_store": self.store_type.upper(),
            "total_documents": len(self.indexed_files),
            "total_chunks": total_chunks,
            "documents": files_summary
        }
