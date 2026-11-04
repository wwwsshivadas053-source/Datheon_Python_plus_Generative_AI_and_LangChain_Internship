"""
Interactive Full-Stack AI Chatbot with Streamlit.
Task Deliverable: Enterprise web-based conversational AI featuring streaming token output,
multi-turn conversation memory, document RAG pipeline, and modular session state management.
"""

import time
import os
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()

from core.config import DEFAULT_CONFIG
from core.memory_manager import ConversationMemoryManager
from core.rag_engine import RAGEngine
from core.llm_manager import LLMManager
from core.agent_tools import AgentToolbox
from ui.components import (
    inject_custom_css,
    render_header,
    render_welcome_screen,
    render_sidebar,
    render_message_with_metadata
)

# ---------------------------------------------------------
# Streamlit Application Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title=f"{DEFAULT_CONFIG['app_title']} | Streamlit AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject modern UI styling
inject_custom_css("ui/styles.css")


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationMemoryManager(
        max_history_turns=DEFAULT_CONFIG["max_history_turns"]
    )

if "rag_engine" not in st.session_state:
    st.session_state["rag_engine"] = RAGEngine(
        chunk_size=DEFAULT_CONFIG["chunk_size"],
        chunk_overlap=DEFAULT_CONFIG["chunk_overlap"]
    )

if "toolbox" not in st.session_state:
    st.session_state["toolbox"] = AgentToolbox()

if "quick_prompt" not in st.session_state:
    st.session_state["quick_prompt"] = None


# ---------------------------------------------------------
# Render Sidebar & Fetch Controls
# ---------------------------------------------------------
settings = render_sidebar(
    memory=st.session_state["memory"],
    rag_engine=st.session_state["rag_engine"]
)

# Synchronize memory settings with sidebar values
st.session_state["memory"].max_history_turns = settings["memory_turns"]
st.session_state["memory"].set_system_prompt(settings["system_prompt"])

# ---------------------------------------------------------
# Render Header
# ---------------------------------------------------------
is_rag_active = "RAG" in settings["mode"] and st.session_state["rag_engine"].is_indexed()
render_header(active_mode=settings["mode"], rag_active=is_rag_active)


# ---------------------------------------------------------
# Display Conversation History
# ---------------------------------------------------------
messages = st.session_state["memory"].get_messages()

quick_choice = None
if len(messages) == 0:
    quick_choice = render_welcome_screen()
else:
    for msg in messages:
        render_message_with_metadata(msg)


# ---------------------------------------------------------
# Handle User Input (Input Bar or Quick Suggestion)
# ---------------------------------------------------------
user_prompt = st.chat_input(
    "Type your query, ask for code, or explore your uploaded documents..."
)

# Detect if user clicked a quick suggestion button
active_prompt = user_prompt or quick_choice

if active_prompt:
    # 1. Render User Message immediately
    with st.chat_message("user"):
        st.markdown(active_prompt)

    # 2. Save User Message into memory state
    st.session_state["memory"].add_user_message(active_prompt)

    # 3. Prepare Context and Tools
    rag_context = ""
    retrieved_sources = []

    # If RAG is enabled and documents are indexed, retrieve relevant chunks
    if "RAG" in settings["mode"] and st.session_state["rag_engine"].is_indexed():
        with st.status("🔍 Searching knowledge base...", expanded=False) as status:
            rag_context, retrieved_sources = st.session_state["rag_engine"].build_rag_context(
                active_prompt,
                top_k=DEFAULT_CONFIG["default_top_k"]
            )
            if retrieved_sources:
                status.update(
                    label=f"Found {len(retrieved_sources)} relevant excerpts",
                    state="complete"
                )
            else:
                status.update(label="No strong document match found", state="complete")

    # If Agent Mode is selected, evaluate if a tool should be executed
    tool_prefix = ""
    if "Agent" in settings["mode"]:
        prompt_lower = active_prompt.lower()
        if any(w in prompt_lower for w in ["calculate", "compute", "what is", "+", "-", "*", "/", "^"]):
            # Extract potential math expression
            expr_candidate = active_prompt
            for trigger in ["calculate", "compute", "what is"]:
                if trigger in prompt_lower:
                    expr_candidate = active_prompt.split(trigger, 1)[-1].strip(" ?:")
            calc_res = st.session_state["toolbox"].calculate(expr_candidate)
            if not calc_res.startswith("Calculation error") and not calc_res.startswith("Error:"):
                tool_prefix = f"🧮 **Tool Output [Calculator]:** `{expr_candidate} = {calc_res}`\n\n"

        elif "time" in prompt_lower or "date" in prompt_lower or "today" in prompt_lower:
            sys_time = st.session_state["toolbox"].get_system_time()
            tool_prefix = f"🕒 **Tool Output [System Clock]:** Current time is `{sys_time}`\n\n"

    # 4. Stream Assistant Output
    with st.chat_message("assistant"):
        start_time = time.perf_counter()

        # Instantiate LLM Manager with current configuration
        llm_mgr = LLMManager(
            provider=settings["provider"],
            model_name=settings["model_id"],
            temperature=settings["temperature"],
            max_tokens=settings["max_tokens"],
            api_key=settings["api_key"]
        )

        # Retrieve windowed LangChain messages
        lc_messages = st.session_state["memory"].get_langchain_messages(include_system=True)

        # Display tool pre-output if triggered
        if tool_prefix:
            st.markdown(tool_prefix)

        # Stream response tokens natively using st.write_stream
        stream_generator = llm_mgr.stream_response(
            messages=lc_messages,
            rag_context=rag_context
        )
        assistant_reply = st.write_stream(stream_generator)
        latency = round(time.perf_counter() - start_time, 2)

        # Combine with tool prefix if any
        full_assistant_content = (tool_prefix + assistant_reply) if tool_prefix else assistant_reply

        # Render retrieved sources expander if present
        if retrieved_sources:
            with st.expander(f"📚 Retrieved Knowledge Sources ({len(retrieved_sources)})", expanded=False):
                for idx, src in enumerate(retrieved_sources, 1):
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

        # Render generation latency
        st.caption(f"⏱️ Generation time: `{latency}s`")

    # 5. Record Assistant response in conversation memory
    st.session_state["memory"].add_assistant_message(
        content=full_assistant_content,
        sources=retrieved_sources if retrieved_sources else None,
        latency_seconds=latency
    )

    # Rerun to update sidebar metrics & telemetry cleanly
    st.rerun()

