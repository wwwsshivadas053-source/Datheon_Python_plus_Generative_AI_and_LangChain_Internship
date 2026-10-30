"""
app.py
======
Flask web server for the Multi-Tool Autonomous AI Agent UI.

Endpoints:
  GET  /              → Serve the main chat UI
  POST /api/chat      → Run a query through the agent, return JSON
  GET  /api/stream    → SSE stream of agent reasoning steps
  POST /api/clear     → Clear conversation memory
  GET  /api/status    → Agent health check
"""

import os
import json
import queue
import threading
import traceback
from datetime import datetime

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

from langchain_core.callbacks import BaseCallbackHandler

from agent.agent_builder import build_agent_executor
from tools import get_all_tools


def clean_output_text(output_val) -> str:
    """Format agent output cleanly into a string regardless of LLM return structure."""
    if output_val is None:
        return "No answer returned."
    if isinstance(output_val, str):
        return output_val
    if isinstance(output_val, list):
        parts = []
        for item in output_val:
            if isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(output_val, dict):
        return output_val.get("text", str(output_val))
    return str(output_val)


def format_tool_input(tool_input) -> str:
    """Extract clean string representation of tool input."""
    if isinstance(tool_input, dict):
        if len(tool_input) == 1:
            return str(next(iter(tool_input.values())))
        return json.dumps(tool_input, ensure_ascii=False)
    return str(tool_input)


class LiveStreamCallback(BaseCallbackHandler):
    """Callback handler to emit real-time status updates via SSE."""
    def __init__(self, q):
        self.q = q

    def on_agent_action(self, action, **kwargs):
        tool = getattr(action, "tool", "tool")
        inp = format_tool_input(getattr(action, "tool_input", "")).replace("\n", " ").strip()[:50]
        self.q.put(("progress", {"text": f"⚡ Running {tool}: {inp}"}))

    def on_tool_end(self, output, **kwargs):
        self.q.put(("progress", {"text": "🧠 Synthesizing observations..."}))


# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
app.secret_key = os.urandom(24)

# ── Global agent (lazy-loaded) ───────────────────────────────────────────────
_executor = None
_executor_lock = threading.Lock()
_agent_error = None


def get_executor():
    global _executor, _agent_error
    with _executor_lock:
        if _executor is None:
            try:
                _executor = build_agent_executor()
                _agent_error = None
            except Exception as e:
                _agent_error = str(e)
                raise
    return _executor


# ── SSE helper ───────────────────────────────────────────────────────────────
def sse_event(data: dict, event: str = "message") -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    provider = os.getenv("LLM_PROVIDER", "google").upper()
    model = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
    search = os.getenv("SEARCH_PROVIDER", "duckduckgo")
    return jsonify({
        "status": "ok" if api_key and api_key != "your_google_gemini_api_key_here" else "missing_key",
        "provider": provider,
        "model": model,
        "search": search,
        "tools": ["web_search", "python_calculator", "file_reader"],
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """Run a query through the agent and return full result as JSON."""
    data = request.get_json()
    query = (data or {}).get("query", "").strip()

    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    try:
        executor = get_executor()
    except Exception as e:
        return jsonify({"error": str(e), "type": "init_error"}), 500

    try:
        result = executor.invoke({"input": query})
    except Exception as e:
        return jsonify({"error": str(e), "type": "agent_error"}), 500

    # Parse intermediate steps
    steps = []
    for action, observation in result.get("intermediate_steps", []):
        import re
        log = getattr(action, "log", "") or ""
        thought = ""
        if log:
            lines = log.strip().splitlines()
            thought_lines = []
            for line in lines:
                if line.lower().startswith("action:"):
                    break
                cleaned = re.sub(r"^thought:\s*", "", line, flags=re.IGNORECASE).strip()
                if cleaned:
                    thought_lines.append(cleaned)
            thought = " ".join(thought_lines)

        tool_name = getattr(action, "tool", "unknown")
        tool_input = format_tool_input(getattr(action, "tool_input", ""))
        if not thought:
            thought = f"Invoked {tool_name} to gather required data."

        steps.append({
            "thought": thought,
            "tool": tool_name,
            "input": tool_input,
            "observation": str(observation),
        })

    return jsonify({
        "answer": clean_output_text(result.get("output")),
        "steps": steps,
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/stream")
def stream():
    """SSE endpoint — streams agent reasoning steps in real time."""
    query = request.args.get("q", "").strip()
    if not query:
        return Response("data: {}\n\n", mimetype="text/event-stream")

    event_queue = queue.Queue()

    def run_agent():
        try:
            executor = get_executor()
        except Exception as e:
            event_queue.put(("error_event", {"message": str(e)}))
            event_queue.put(("done", {}))
            return

        import re

        try:
            callback = LiveStreamCallback(event_queue)
            result = executor.invoke({"input": query}, config={"callbacks": [callback]})
        except Exception as e:
            event_queue.put(("error_event", {"message": str(e)}))
            event_queue.put(("done", {}))
            return

        for action, observation in result.get("intermediate_steps", []):
            log = getattr(action, "log", "")
            thought = ""
            if log:
                lines = log.strip().splitlines()
                thought_lines = []
                for line in lines:
                    if line.lower().startswith("action:"):
                        break
                    cleaned = re.sub(r"^thought:\s*", "", line, flags=re.IGNORECASE).strip()
                    if cleaned:
                        thought_lines.append(cleaned)
                thought = " ".join(thought_lines)

            tool_name = getattr(action, "tool", "unknown")
            tool_input = format_tool_input(getattr(action, "tool_input", ""))
            if not thought:
                thought = f"Invoked {tool_name} to gather required data."

            event_queue.put(("step", {
                "thought": thought,
                "tool": tool_name,
                "input": tool_input[:300],
                "observation": str(observation)[:500],
            }))

        final_answer = clean_output_text(result.get("output"))
        event_queue.put(("answer", {
            "text": final_answer,
            "timestamp": datetime.now().isoformat(),
        }))
        event_queue.put(("done", {}))

    thread = threading.Thread(target=run_agent, daemon=True)
    thread.start()

    def generate():
        yield sse_event({"status": "thinking"}, "thinking")
        while True:
            try:
                event_type, payload = event_queue.get(timeout=120)
                yield sse_event(payload, event_type)
                if event_type == "done":
                    break
            except queue.Empty:
                yield sse_event({"message": "timeout"}, "error")
                break

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/clear", methods=["POST"])
def clear_memory():
    """Clear the agent's conversation memory and reload executor."""
    global _executor
    with _executor_lock:
        if _executor and hasattr(_executor, "memory") and _executor.memory:
            _executor.memory.clear()
        _executor = None
    return jsonify({"status": "cleared"})


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"\n  [AI Agent] Multi-Tool AI Agent UI")
    print(f"  [WEB] Open in browser: http://localhost:{port}")
    print(f"  [LLM] Provider: {os.getenv('LLM_PROVIDER','google').upper()} / {os.getenv('LLM_MODEL','gemini-2.0-flash')}")
    print(f"  [INFO] Press Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
