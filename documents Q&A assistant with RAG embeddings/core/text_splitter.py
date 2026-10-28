from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class TextSplitterService:
    """
    Service for recursively splitting documents into semantically coherent chunks
    while preserving page citations and metadata for attribution.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
            length_function=len,
            is_separator_regex=False
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of page-level Documents into smaller chunks.
        Augments metadata with unique chunk_id and sequential chunk_index.
        """
        raw_chunks = self.splitter.split_documents(documents)
        enriched_chunks: List[Document] = []

        for idx, chunk in enumerate(raw_chunks):
            # Clean up whitespace
            content = chunk.page_content.strip()
            if not content:
                continue

            metadata = dict(chunk.metadata)
            page_num = metadata.get("page", 1)
            file_name = metadata.get("file_name", "unknown")
            metadata["chunk_index"] = idx
            metadata["chunk_id"] = f"{file_name}_p{page_num}_c{idx}"
            metadata["chunk_char_length"] = len(content)

            enriched_chunks.append(Document(page_content=content, metadata=metadata))

        return enriched_chunks
