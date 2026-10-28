import os
import shutil
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from config import Config, UPLOAD_DIR, SAMPLE_DOCS_DIR
from core.document_loader import DocumentLoaderService
from core.text_splitter import TextSplitterService
from core.embedding_manager import EmbeddingManager
from core.vector_store import VectorStoreService
from core.rag_pipeline import RAGPipeline


app = Flask(__name__)
app.config.from_object(Config)

# Global in-memory storage of loaded raw Documents for re-indexing across stores
loaded_documents_registry = []

# Initialize Core Services
current_embeddings = EmbeddingManager.get_embeddings(
    provider=Config.DEFAULT_LLM_PROVIDER,
    google_api_key=Config.GOOGLE_API_KEY,
    openai_api_key=Config.OPENAI_API_KEY
)

vector_service = VectorStoreService(
    store_type=Config.DEFAULT_VECTOR_STORE,
    embeddings=current_embeddings
)

text_splitter = TextSplitterService(
    chunk_size=Config.CHUNK_SIZE,
    chunk_overlap=Config.CHUNK_OVERLAP
)

rag_pipeline = RAGPipeline(
    vector_service=vector_service,
    llm_provider=Config.DEFAULT_LLM_PROVIDER,
    google_api_key=Config.GOOGLE_API_KEY,
    openai_api_key=Config.OPENAI_API_KEY,
    top_k=Config.TOP_K
)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Renders the main RAG Assistant interface."""
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def get_status():
    """Returns active system status, store type, model configuration, and indexed documents."""
    stats = vector_service.get_stats()
    stats.update({
        "llm_provider": rag_pipeline.llm_provider,
        "has_google_key": bool(rag_pipeline.google_api_key),
        "has_openai_key": bool(rag_pipeline.openai_api_key),
        "chunk_size": text_splitter.chunk_size,
        "chunk_overlap": text_splitter.chunk_overlap,
        "top_k": rag_pipeline.top_k,
        "active_store": vector_service.store_type.upper()
    })
    return jsonify({"success": True, "data": stats})


@app.route("/api/upload", methods=["POST"])
def upload_document():
    """
    Handles PDF and text document uploads.
    Extracts pages, applies recursive character chunking,
    generates vector embeddings, and indexes into the vector database.
    """
    if "files" not in request.files:
        return jsonify({"success": False, "error": "No file field found in request."}), 400

    files = request.files.getlist("files")
    if not files or files[0].filename == "":
        return jsonify({"success": False, "error": "No files selected."}), 400

    uploaded_summary = []

    for file in files:
        if file and allowed_file(file.filename):
            safe_name = secure_filename(file.filename)
            save_path = UPLOAD_DIR / safe_name
            file.save(str(save_path))

            try:
                # 1. Parse pages
                page_docs = DocumentLoaderService.load_document(str(save_path))
                if not page_docs:
                    continue

                # 2. Chunk recursively
                chunks = text_splitter.split_documents(page_docs)

                # 3. Add to vector store
                vector_service.add_documents(chunks)
                loaded_documents_registry.extend(chunks)

                uploaded_summary.append({
                    "file_name": safe_name,
                    "pages": len(page_docs),
                    "chunks": len(chunks),
                    "total_characters": sum(len(d.page_content) for d in page_docs)
                })
            except Exception as e:
                return jsonify({"success": False, "error": f"Error indexing {safe_name}: {str(e)}"}), 500
        else:
            return jsonify({"success": False, "error": f"Invalid format for '{file.filename}'. Allowed: PDF, TXT, MD, DOCX."}), 400

    return jsonify({
        "success": True,
        "message": f"Successfully processed and indexed {len(uploaded_summary)} document(s).",
        "documents": uploaded_summary,
        "system_stats": vector_service.get_stats()
    })


@app.route("/api/load-sample", methods=["POST"])
def load_sample():
    """Preloads the built-in research paper PDF for zero-friction testing."""
    try:
        sample_pdf = SAMPLE_DOCS_DIR / "ai_rag_overview.pdf"
        if not sample_pdf.exists():
            from scripts.generate_sample_pdf import create_sample_pdf
            create_sample_pdf(str(sample_pdf))

        target_path = UPLOAD_DIR / "ai_rag_overview.pdf"
        shutil.copy(str(sample_pdf), str(target_path))

        page_docs = DocumentLoaderService.load_document(str(target_path))
        chunks = text_splitter.split_documents(page_docs)
        vector_service.add_documents(chunks)
        loaded_documents_registry.extend(chunks)

        return jsonify({
            "success": True,
            "message": "Sample PDF (AI & RAG Architecture Overview) loaded and indexed successfully!",
            "document": {
                "file_name": "ai_rag_overview.pdf",
                "pages": len(page_docs),
                "chunks": len(chunks)
            },
            "system_stats": vector_service.get_stats()
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Error loading sample: {str(e)}"}), 500


@app.route("/api/query", methods=["POST"])
def query_assistant():
    """
    Retrieval-Augmented Generation Query Endpoint.
    Retrieves top-k relevant chunks with similarity metrics,
    enforces strict context grounding, and outputs answer with exact page citations.
    """
    data = request.get_json() or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"success": False, "error": "Query question is required."}), 400

    try:
        response = rag_pipeline.query(question)
        return jsonify({
            "success": True,
            "data": response
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"RAG Query processing error: {str(e)}"}), 500


@app.route("/api/switch-store", methods=["POST"])
def switch_store():
    """Switches active vector store between ChromaDB and FAISS on demand."""
    data = request.get_json() or {}
    new_store = data.get("store_type", "").lower()

    if new_store not in ["faiss", "chroma"]:
        return jsonify({"success": False, "error": "Store must be 'faiss' or 'chroma'."}), 400

    try:
        vector_service.switch_store(new_store, all_current_documents=loaded_documents_registry)
        return jsonify({
            "success": True,
            "message": f"Successfully switched vector store to {new_store.upper()}.",
            "stats": vector_service.get_stats()
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to switch vector store: {str(e)}"}), 500


@app.route("/api/settings", methods=["POST"])
def update_settings():
    """Updates API keys, active model provider, chunking hyperparameters, and retrieval k."""
    data = request.get_json() or {}

    provider = data.get("llm_provider")
    google_key = data.get("google_api_key")
    openai_key = data.get("openai_api_key")
    top_k = data.get("top_k")
    temperature = data.get("temperature")
    chunk_size = data.get("chunk_size")
    chunk_overlap = data.get("chunk_overlap")

    if chunk_size and chunk_overlap:
        text_splitter.chunk_size = int(chunk_size)
        text_splitter.chunk_overlap = int(chunk_overlap)
        text_splitter.__init__(chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))

    # Update embeddings if provider changed or keys provided
    if provider or google_key or openai_key:
        new_embeddings = EmbeddingManager.get_embeddings(
            provider=provider or rag_pipeline.llm_provider,
            google_api_key=google_key or rag_pipeline.google_api_key,
            openai_api_key=openai_key or rag_pipeline.openai_api_key
        )
        vector_service.set_embeddings(new_embeddings)
        if loaded_documents_registry:
            vector_service.clear()
            vector_service.add_documents(loaded_documents_registry)

    rag_pipeline.update_llm_settings(
        provider=provider,
        google_key=google_key,
        openai_key=openai_key,
        temperature=temperature,
        top_k=top_k
    )

    return jsonify({
        "success": True,
        "message": "Settings updated successfully.",
        "status": {
            "llm_provider": rag_pipeline.llm_provider,
            "has_google_key": bool(rag_pipeline.google_api_key),
            "has_openai_key": bool(rag_pipeline.openai_api_key),
            "top_k": rag_pipeline.top_k,
            "chunk_size": text_splitter.chunk_size,
            "chunk_overlap": text_splitter.chunk_overlap
        }
    })


@app.route("/api/clear", methods=["POST"])
def clear_all():
    """Clears all vector stores, uploaded documents, and dialogue memory."""
    global loaded_documents_registry
    loaded_documents_registry = []
    vector_service.clear()
    rag_pipeline.clear_history()

    # Clear uploaded files
    for item in UPLOAD_DIR.iterdir():
        if item.is_file():
            try:
                item.unlink()
            except Exception:
                pass

    return jsonify({
        "success": True,
        "message": "Vector databases, uploaded files, and chat history cleared successfully."
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print("\n" + "=" * 55)
    print("Document Q&A Assistant with RAG & Vector Embeddings")
    print(f"Server running at: http://127.0.0.1:{port}")
    print(f"Active Vector Store: {vector_service.store_type.upper()}")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False)
