import os
import re
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.vector_store import VectorStoreService


SYSTEM_PROMPT = """You are a senior AI Document Research Assistant operating within a high-precision Retrieval-Augmented Generation (RAG) system.

Your core directives:
1. Groundedness: Answer the question STRICTLY and SOLELY using the provided context passages below.
2. Anti-Hallucination: If the provided context does not contain enough information to formulate an accurate answer, you MUST state clearly:
   "I cannot find the answer to this question in the provided document context."
   Do NOT extrapolate, guess, or incorporate external knowledge not in the text.
3. Citation Requirement: Whenever you state a key fact, cite the document name and page number using format: [Doc: {{file_name}}, Page: {{page}}].
4. Tone & Clarity: Deliver clear, structured, well-formatted markdown responses with bullet points where appropriate.

Context Passages:
{context}

Chat History:
{chat_history}
"""


class RAGPipeline:
    """
    End-to-End Retrieval-Augmented Generation (RAG) Pipeline.
    Enforces strict context grounding, hallucination prevention,
    and structured source page attribution.
    """

    def __init__(
        self,
        vector_service: VectorStoreService,
        llm_provider: str = "gemini",
        google_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        temperature: float = 0.1,
        top_k: int = 4
    ):
        self.vector_service = vector_service
        self.llm_provider = llm_provider.lower()
        self.google_api_key = (google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
        self.openai_api_key = (openai_api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        self.temperature = temperature
        self.top_k = top_k
        self.chat_history: List[Dict[str, str]] = []

    def update_llm_settings(
        self,
        provider: Optional[str] = None,
        google_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None
    ):
        """Update runtime hyperparameters and credentials."""
        if provider:
            self.llm_provider = provider.lower().strip()
        if google_key is not None:
            self.google_api_key = google_key.strip()
        if openai_key is not None:
            self.openai_api_key = openai_key.strip()
        if temperature is not None:
            self.temperature = float(temperature)
        if top_k is not None:
            self.top_k = int(top_k)

    def _format_context(self, docs_with_scores: List[Tuple[Document, float]]) -> str:
        """Formats retrieved chunks with metadata headers for prompt ingestion."""
        formatted_chunks = []
        for idx, (doc, score) in enumerate(docs_with_scores, 1):
            fname = doc.metadata.get("file_name", "Unknown File")
            page = doc.metadata.get("page", 1)
            chunk_id = doc.metadata.get("chunk_id", f"chunk_{idx}")
            header = f"--- [Passage {idx} | Document: {fname} | Page: {page} | Relevance: {score:.2f} | ID: {chunk_id}] ---"
            formatted_chunks.append(f"{header}\n{doc.page_content.strip()}")
        return "\n\n".join(formatted_chunks)

    def _format_history(self) -> str:
        """Formats last few conversation turns for dialogue continuity."""
        if not self.chat_history:
            return "No prior conversational context."
        recent = self.chat_history[-6:]
        return "\n".join([f"{item['role'].capitalize()}: {item['content']}" for item in recent])

    def _extract_offline_answer(self, query: str, docs_with_scores: List[Tuple[Document, float]]) -> str:
        """
        Extractive answer synthesizer used in local offline mode (no API keys configured).
        Ranks relevant sentences and presents grounded extracts with page citations.
        """
        if not docs_with_scores:
            return "I cannot find the answer to this question in the provided document context."

        query_terms = set(re.findall(r"\w+", query.lower()))
        best_sentences = []

        for doc, score in docs_with_scores:
            fname = doc.metadata.get("file_name", "Document")
            page = doc.metadata.get("page", 1)
            # Split into rough sentences
            sentences = re.split(r"(?<=[.?!])\s+", doc.page_content.strip())
            for sent in sentences:
                sent_clean = sent.strip()
                if len(sent_clean) < 20:
                    continue
                words = set(re.findall(r"\w+", sent_clean.lower()))
                overlap = len(words.intersection(query_terms))
                if overlap > 0:
                    best_sentences.append({
                        "sentence": sent_clean,
                        "overlap": overlap,
                        "file_name": fname,
                        "page": page,
                        "score": score
                    })

        best_sentences.sort(key=lambda x: (x["overlap"], x["score"]), reverse=True)
        top_sentences = best_sentences[:4]

        if not top_sentences:
            top_doc, top_score = docs_with_scores[0]
            fname = top_doc.metadata.get("file_name", "Document")
            page = top_doc.metadata.get("page", 1)
            snippet = top_doc.page_content[:300].strip()
            return (
                f"**Most Relevant Match:**\n\n> \"{snippet}...\"\n\n"
                f"*(Source: **{fname}**, Page **{page}** | Relevance: {top_score:.2f})*\n\n"
                f"> 💡 *Tip: Connect a Google Gemini or OpenAI API key in the top settings to enable full generative synthesis.*"
            )

        answer_parts = ["Based on the loaded document context:\n"]
        for item in top_sentences:
            answer_parts.append(
                f"- \"{item['sentence']}\" **[Doc: {item['file_name']}, Page: {item['page']}]**"
            )

        answer_parts.append("\n\n> 💡 *Operating in Local Extractive Mode. Enter a Google Gemini or OpenAI API key above for generative narrative answers.*")
        return "\n".join(answer_parts)

    def query(self, question: str) -> Dict[str, Any]:
        """
        Executes end-to-end RAG query:
        1. Retrieval of top-k chunks with similarity scores.
        2. Strict context grounding check.
        3. LLM generation or offline synthesis.
        4. Structured citation compilation.
        """
        cleaned_question = question.strip()
        if not cleaned_question:
            return {
                "answer": "Please provide a valid question.",
                "citations": [],
                "retrieved_chunks": 0
            }

        # Step 1: Retrieval
        docs_with_scores = self.vector_service.similarity_search_with_score(
            cleaned_question, k=self.top_k
        )

        if not docs_with_scores:
            return {
                "answer": "I cannot answer this question. No documents have been indexed yet. Please upload a PDF or text document first.",
                "citations": [],
                "retrieved_chunks": 0
            }

        # Format citations
        citations = []
        for doc, score in docs_with_scores:
            citations.append({
                "file_name": doc.metadata.get("file_name", "Unknown"),
                "page": doc.metadata.get("page", 1),
                "total_pages": doc.metadata.get("total_pages", 1),
                "relevance_score": score,
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "snippet": doc.page_content.strip()
            })

        # Step 2: Generation
        answer = ""
        context_str = self._format_context(docs_with_scores)
        history_str = self._format_history()

        # Check for Gemini
        if self.llm_provider == "gemini" and self.google_api_key:
            from langchain_google_genai import ChatGoogleGenerativeAI
            candidate_models = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-1.5-flash", "gemini-2.0-flash"]
            gemini_success = False

            for model_name in candidate_models:
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=model_name,
                        google_api_key=self.google_api_key,
                        temperature=self.temperature
                    )
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", SYSTEM_PROMPT),
                        ("user", "{question}")
                    ])
                    chain = prompt_template | llm | StrOutputParser()
                    answer = chain.invoke({
                        "context": context_str,
                        "chat_history": history_str,
                        "question": cleaned_question
                    })
                    gemini_success = True
                    break
                except Exception as e:
                    print(f"[RAGPipeline] Model {model_name} invocation attempt error: {e}")
                    continue

            if not gemini_success:
                print("[RAGPipeline] All Gemini generation models failed. Using extractive mode.")
                answer = self._extract_offline_answer(cleaned_question, docs_with_scores)

        # Check for OpenAI
        elif self.llm_provider == "openai" and self.openai_api_key:
            try:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    openai_api_key=self.openai_api_key,
                    temperature=self.temperature
                )
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", SYSTEM_PROMPT),
                    ("user", "{question}")
                ])
                chain = prompt_template | llm | StrOutputParser()
                answer = chain.invoke({
                    "context": context_str,
                    "chat_history": history_str,
                    "question": cleaned_question
                })
            except Exception as e:
                print(f"[RAGPipeline] OpenAI LLM error: {e}. Falling back to extractive mode.")
                answer = self._extract_offline_answer(cleaned_question, docs_with_scores)

        else:
            # Local Extractive Synthesis
            answer = self._extract_offline_answer(cleaned_question, docs_with_scores)

        # Update chat history
        self.chat_history.append({"role": "user", "content": cleaned_question})
        self.chat_history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": len(docs_with_scores),
            "vector_store": self.vector_service.store_type.upper(),
            "model_provider": self.llm_provider
        }

    def clear_history(self):
        """Clears conversational memory."""
        self.chat_history.clear()
