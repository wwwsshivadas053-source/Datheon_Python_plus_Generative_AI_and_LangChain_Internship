"""
Conversation Memory Manager.
Handles multi-turn dialogue state, LangChain message conversion,
sliding-window context trimming, and history serialization/export.
"""

from typing import List, Dict, Any, Optional
import json
from datetime import datetime
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage
)


class ConversationMemoryManager:
    """
    Manages session-state conversation history with windowed pruning,
    metadata tracking, and serialization.
    """

    def __init__(self, max_history_turns: int = 10):
        self.max_history_turns = max_history_turns
        self.messages: List[Dict[str, Any]] = []
        self.system_prompt: Optional[str] = None
        self.created_at = datetime.now().isoformat()

    def set_system_prompt(self, system_prompt: str) -> None:
        """Update or set the guiding system prompt."""
        self.system_prompt = system_prompt

    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a user query."""
        self.messages.append({
            "role": "user",
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        })

    def add_assistant_message(
        self,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        latency_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record an assistant response with optional sources and performance metrics."""
        meta = metadata or {}
        if sources:
            meta["sources"] = sources
        if latency_seconds is not None:
            meta["latency_seconds"] = round(latency_seconds, 2)

        self.messages.append({
            "role": "assistant",
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": meta
        })

    def get_messages(self) -> List[Dict[str, Any]]:
        """Return the complete conversational history."""
        return self.messages

    def clear(self) -> None:
        """Reset conversation memory."""
        self.messages.clear()

    def get_langchain_messages(self, include_system: bool = True) -> List[BaseMessage]:
        """
        Convert stored messages into LangChain BaseMessage objects,
        applying a sliding-window over the last N conversational turns.
        """
        lc_messages: List[BaseMessage] = []

        if include_system and self.system_prompt:
            lc_messages.append(SystemMessage(content=self.system_prompt))

        # Sliding window: max_history_turns * 2 (user + assistant pairs)
        window_size = self.max_history_turns * 2
        active_slice = self.messages[-window_size:] if len(self.messages) > window_size else self.messages

        for msg in active_slice:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            elif role == "system":
                lc_messages.append(SystemMessage(content=content))

        return lc_messages

    def get_context_string(self, max_turns: Optional[int] = None) -> str:
        """Format recent dialogue turns into a plain-text prompt context string."""
        turns = max_turns or self.max_history_turns
        window = self.messages[-(turns * 2):] if len(self.messages) > (turns * 2) else self.messages
        context_lines = []
        for msg in window:
            speaker = "Human" if msg["role"] == "user" else "Assistant"
            context_lines.append(f"{speaker}: {msg['content']}")
        return "\n".join(context_lines)

    def get_statistics(self) -> Dict[str, Any]:
        """Calculate conversation statistics and engagement metrics."""
        user_msgs = [m for m in self.messages if m["role"] == "user"]
        bot_msgs = [m for m in self.messages if m["role"] == "assistant"]

        total_words = sum(len(m.get("content", "").split()) for m in self.messages)
        latencies = [
            m["metadata"]["latency_seconds"]
            for m in bot_msgs
            if "latency_seconds" in m.get("metadata", {})
        ]
        avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

        return {
            "total_messages": len(self.messages),
            "user_messages": len(user_msgs),
            "assistant_messages": len(bot_msgs),
            "total_words": total_words,
            "estimated_tokens": int(total_words * 1.33),
            "avg_latency_seconds": avg_latency
        }

    def export_json(self) -> str:
        """Export history as a structured JSON string."""
        export_payload = {
            "created_at": self.created_at,
            "exported_at": datetime.now().isoformat(),
            "system_prompt": self.system_prompt,
            "statistics": self.get_statistics(),
            "messages": self.messages
        }
        return json.dumps(export_payload, indent=2, ensure_ascii=False)

    def export_markdown(self) -> str:
        """Export history formatted as rich Markdown."""
        lines = [
            "# Conversation Transcript",
            f"*Exported at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "---",
            ""
        ]

        if self.system_prompt:
            lines.extend([
                "### ⚙️ System Instructions",
                f"> {self.system_prompt}",
                "",
                "---",
                ""
            ])

        for idx, msg in enumerate(self.messages, 1):
            role_label = "🧑 **User**" if msg["role"] == "user" else "🤖 **Assistant**"
            ts = msg.get("timestamp", "")
            time_str = f" `{ts[11:19]}`" if len(ts) >= 19 else ""
            lines.append(f"### {idx}. {role_label}{time_str}")
            lines.append(msg["content"])

            meta = msg.get("metadata", {})
            if meta.get("sources"):
                lines.append("\n**Retrieved Sources:**")
                for s in meta["sources"]:
                    src_title = s.get("source", "Document")
                    page = f" (page {s.get('page')})" if s.get("page") else ""
                    lines.append(f"- *{src_title}{page}*: {s.get('snippet', '')[:120]}...")

            if meta.get("latency_seconds"):
                lines.append(f"\n*Response latency: {meta['latency_seconds']}s*")

            lines.append("\n---\n")

        return "\n".join(lines)

