"""
UI Components Module.
Reusable Streamlit components: Sidebar, App Header, Chat Renderers,
Telemetry cards, RAG controls, and Export handlers.
"""

from typing import Dict, Any, List, Optional
import os
import streamlit as st

from core.config import PROVIDER_MODELS, PERSONAS, DEFAULT_CONFIG
from core.rag_engine import RAGEngine
from core.memory_manager import ConversationMemoryManager


def inject_custom_css(css_path: str = "ui/styles.css") -> None:
    """Inject external CSS stylesheet into Streamlit."""
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def render_header(active_mode: str, rag_active: bool = False) -> None:
    """Render the top banner with application title, status badges, and architecture info."""
    badge_type = "badge-success" if rag_active else "badge-primary"
    mode_label = "RAG Mode Active 📚" if rag_active else active_mode

    st.markdown(
        f"""
        <div class="app-header">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <h1 class="app-title">
                        <span>⚡</span> {DEFAULT_CONFIG['app_title']}
                        <span class="badge {badge_type}">{mode_label}</span>
                    </h1>
                    <p class="app-subtitle">
                        {DEFAULT_CONFIG['app_subtitle']} • <small>v{DEFAULT_CONFIG['app_version']}</small>
                    </p>
                </div>
                <div style="display: flex; gap: 0.5rem; margin-top: 0.5rem;">
                    <span class="badge badge-primary">LangChain Core</span>
                    <span class="badge badge-success">Streaming Token Engine</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_welcome_screen() -> Optional[str]:
    """Render attractive welcome prompt cards for empty conversation state."""
    st.markdown(
        """
        <div class="welcome-card">
            <div class="welcome-icon">💬</div>
            <div class="welcome-heading">Welcome to Full-Stack AI Nexus</div>
            <div class="welcome-text">
                An enterprise conversational workspace featuring real-time token streaming,
                dynamic session memory, and in-memory document RAG.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("##### 💡 Try an example prompt to get started:")
    col1, col2 = st.columns(2)

    quick_prompt = None
    with col1:
        if st.button("🚀 Explain Streamlit session state architecture", use_container_width=True):
            quick_prompt = "Explain how Streamlit session_state works under the hood and best practices for managing state."
        if st.button("🐍 Write a Python streaming generator", use_container_width=True):
            quick_prompt = "Write a production-grade Python streaming token generator with type annotations and error handling."

    with col2:
        if st.button("🔍 How does RAG vector search work?", use_container_width=True):
            quick_prompt = "Explain the step-by-step pipeline of Retrieval-Augmented Generation (RAG) from chunking to cosine similarity."
        if st.button("📐 Design an AI agent tool pipeline", use_container_width=True):
            quick_prompt = "How do AI agents plan and execute tools? Provide an architectural diagram in markdown."

    return quick_prompt


def render_sidebar(
    memory: ConversationMemoryManager,
    rag_engine: RAGEngine
) -> Dict[str, Any]:
    """
    Renders the configuration sidebar and returns selected hyperparameters and settings.
    """
    settings: Dict[str, Any] = {}

    with st.sidebar:
        st.markdown("### ⚙️ Pipeline Configuration")

        # 1. Operational Mode
        mode = st.radio(
            "Application Mode",
            options=["Standard Conversational AI", "RAG (Document Knowledge Base)", "Agent Tools & Actions"],
            index=0,
            help="Choose between vanilla chat, document-grounded RAG, or tool execution."
        )
        settings["mode"] = mode

        st.markdown("---")

        # 2. LLM Provider & Model
        st.markdown("#### 🤖 Model Provider")
        providers = list(PROVIDER_MODELS.keys())
        selected_provider = st.selectbox(
            "Provider",
            options=providers,
            index=0,
            help="Select your model provider or use Demo Mode for instant keyless testing."
        )
        settings["provider"] = selected_provider

        models = PROVIDER_MODELS[selected_provider]
        model_names = [m.name for m in models]
        selected_model_idx = st.selectbox(
            "Model",
            options=range(len(model_names)),
            format_func=lambda i: model_names[i],
            help="Select the specific LLM checkpoint."
        )
        selected_model_obj = models[selected_model_idx]
        settings["model_id"] = selected_model_obj.id
        settings["model_name"] = selected_model_obj.name

        # API Key input if not Demo Mode
        api_key_input = ""
        if "Demo Mode" not in selected_provider:
            env_var_name = ""
            if "Gemini" in selected_provider:
                env_var_name = "GEMINI_API_KEY"
            elif "OpenAI" in selected_provider:
                env_var_name = "OPENAI_API_KEY"
            elif "Groq" in selected_provider:
                env_var_name = "GROQ_API_KEY"
            elif "Anthropic" in selected_provider:
                env_var_name = "ANTHROPIC_API_KEY"

            default_val = os.getenv(env_var_name, "")
            api_key_input = st.text_input(
                f"{selected_provider} API Key",
                value=default_val,
                type="password",
                help=f"Loaded automatically from ${env_var_name} or enter here manually."
            )
            if not api_key_input:
                st.info(f"💡 No key provided. App will fallback to Demo mode.")
        settings["api_key"] = api_key_input

        st.markdown("---")

        # 3. Persona & System Instructions
        st.markdown("#### 🎭 Persona & System Prompt")
        persona_choice = st.selectbox(
            "Persona Preset",
            options=list(PERSONAS.keys()),
            index=0
        )
        default_prompt = PERSONAS[persona_choice]

        with st.expander("Custom Prompt Editor", expanded=False):
            system_prompt = st.text_area(
                "System Instructions",
                value=default_prompt,
                height=110,
                help="Instructions that dictate the persona, tone, and formatting constraints."
            )
        settings["system_prompt"] = system_prompt

        # 4. Hyperparameters
        with st.expander("🎛️ Hyperparameters", expanded=False):
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.5,
                value=DEFAULT_CONFIG["default_temperature"],
                step=0.05,
                help="Controls randomness: 0.0 is deterministic, 1.0 is creative."
            )
            max_tokens = st.slider(
                "Max Tokens",
                min_value=256,
                max_value=4096,
                value=DEFAULT_CONFIG["default_max_tokens"],
                step=128,
                help="Maximum number of response tokens to generate."
            )
            memory_turns = st.slider(
                "Memory Window (Turns)",
                min_value=1,
                max_value=25,
                value=DEFAULT_CONFIG["max_history_turns"],
                step=1,
                help="Number of conversational turns retained in sliding window context."
            )
        settings["temperature"] = temperature
        settings["max_tokens"] = max_tokens
        settings["memory_turns"] = memory_turns

        # 5. RAG Document Ingestion Section (shown if RAG mode is active)
        if "RAG" in mode:
            st.markdown("---")
            st.markdown("#### 📚 Knowledge Base (RAG)")
            uploaded_files = st.file_uploader(
                "Upload Documents",
                type=["pdf", "docx", "txt", "md", "csv"],
                accept_multiple_files=True,
                help="Upload documents to ground the chatbot's answers in your own private knowledge base."
            )

            if uploaded_files:
                if st.button("📥 Process & Index Documents", use_container_width=True, type="primary"):
                    total_chunks_added = 0
                    with st.spinner("Parsing and vectorizing documents..."):
                        for up_file in uploaded_files:
                            file_bytes = up_file.read()
                            chunks_count = rag_engine.ingest_file(up_file.name, file_bytes)
                            total_chunks_added += chunks_count
                    st.success(f"Indexed {len(uploaded_files)} files ({total_chunks_added} chunks)!")
                    st.rerun()

            if rag_engine.is_indexed():
                st.markdown(
                    f"**Indexed:** `{rag_engine.get_document_count()}` doc(s) • `{rag_engine.get_total_chunks()}` chunks"
                )
                if st.button("🗑️ Clear Vector Index", use_container_width=True):
                    rag_engine.clear()
                    st.info("Vector index cleared.")
                    st.rerun()

        st.markdown("---")

        # 6. Session Telemetry & Conversation Controls
        st.markdown("#### 📊 Session Telemetry")
        stats = memory.get_statistics()
        c1, c2, c3 = st.columns(3)
        c1.metric("Messages", stats["total_messages"])
        c2.metric("Tokens ~", stats["estimated_tokens"])
        c3.metric("Avg Latency", f"{stats['avg_latency_seconds']}s")

        st.markdown("#### 💾 Session Management")
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            st.download_button(
                label="📄 Markdown",
                data=memory.export_markdown(),
                file_name="chat_transcript.md",
                mime="text/markdown",
                use_container_width=True,
                disabled=len(memory.messages) == 0
            )
        with export_col2:
            st.download_button(
                label="📦 JSON",
                data=memory.export_json(),
                file_name="chat_history.json",
                mime="application/json",
                use_container_width=True,
                disabled=len(memory.messages) == 0
            )

        if st.button("🧹 Clear Chat History", use_container_width=True):
            memory.clear()
            st.rerun()

    return settings


def render_message_with_metadata(msg: Dict[str, Any]) -> None:
    """Render a single conversation message with sources and latency telemetry."""
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    metadata = msg.get("metadata", {})

    with st.chat_message(role):
        st.markdown(content)

        # Render retrieved RAG sources if present
        sources = metadata.get("sources", [])
        if sources:
            with st.expander(f"📚 Retrieved Knowledge Sources ({len(sources)})", expanded=False):
                for idx, src in enumerate(sources, 1):
                    doc_name = src.get("source", "Document")
                    page = f" • Page {src.get('page')}" if src.get("page") else ""
                    score = f" • Match: {int(src.get('score', 0) * 100)}%"
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div class="source-title">[{idx}] {doc_name}{page}{score}</div>
                            <div class="source-snippet">"{src.get('snippet', '')}"</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        # Render latency if recorded
        if "latency_seconds" in metadata:
            latency = metadata["latency_seconds"]
            st.caption(f"⏱️ Generation time: `{latency}s`")

