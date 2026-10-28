import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env or .env.example
env_path = Path(__file__).resolve().parent / ".env"
env_example_path = Path(__file__).resolve().parent / ".env.example"

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
elif env_example_path.exists():
    load_dotenv(dotenv_path=env_example_path)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chromadb"
FAISS_DIR = DATA_DIR / "faiss_index"
SAMPLE_DOCS_DIR = BASE_DIR / "sample_docs"

# Ensure runtime directories exist
for path in [DATA_DIR, UPLOAD_DIR, CHROMA_DIR, FAISS_DIR, SAMPLE_DOCS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

class Config:
    # Flask Settings
    SECRET_KEY = os.getenv("SECRET_KEY", "rag-assistant-secret-key-2026").strip()
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32 MB max upload
    ALLOWED_EXTENSIONS = {"pdf", "txt", "docx", "md"}

    # Vector Store Settings
    DEFAULT_VECTOR_STORE = os.getenv("DEFAULT_VECTOR_STORE", "faiss").strip().lower()
    COLLECTION_NAME = "document_qa_collection"

    # Chunking Hyperparameters (Recommended for technical & general PDF RAG)
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
    TOP_K = int(os.getenv("TOP_K", 4))

    # Model Settings
    # Supported: "gemini" (Google Gemini), "openai" (OpenAI), "local" (TF-IDF / Cosine fallback)
    DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "gemini").strip().lower()
    
    # API Keys (clean leading/trailing whitespace)
    GOOGLE_API_KEY = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
