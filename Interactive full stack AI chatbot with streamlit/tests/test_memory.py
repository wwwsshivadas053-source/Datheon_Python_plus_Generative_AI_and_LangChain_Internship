"""
Unit tests for ConversationMemoryManager.
"""

import unittest
from core.memory_manager import ConversationMemoryManager
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class TestConversationMemoryManager(unittest.TestCase):

    def setUp(self):
        self.mem = ConversationMemoryManager(max_history_turns=3)

    def test_initial_state(self):
        self.assertEqual(len(self.mem.get_messages()), 0)
        stats = self.mem.get_statistics()
        self.assertEqual(stats["total_messages"], 0)
        self.assertEqual(stats["avg_latency_seconds"], 0.0)

    def test_add_and_retrieve_messages(self):
        self.mem.add_user_message("Hello world")
        self.mem.add_assistant_message("Hi there!", latency_seconds=0.42)

        messages = self.mem.get_messages()
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "Hello world")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["metadata"]["latency_seconds"], 0.42)

    def test_sliding_window_pruning(self):
        # max_history_turns = 3, so window size is 6 messages (3 turns)
        for i in range(10):
            self.mem.add_user_message(f"Query {i}")
            self.mem.add_assistant_message(f"Answer {i}")

        self.assertEqual(len(self.mem.get_messages()), 20)

        lc_msgs = self.mem.get_langchain_messages(include_system=False)
        self.assertEqual(len(lc_msgs), 6)  # 3 turns = 6 messages
        self.assertIsInstance(lc_msgs[-1], AIMessage)
        self.assertEqual(lc_msgs[-1].content, "Answer 9")

    def test_system_prompt_inclusion(self):
        self.mem.set_system_prompt("You are a helpful coding assistant.")
        self.mem.add_user_message("Test")

        lc_msgs = self.mem.get_langchain_messages(include_system=True)
        self.assertEqual(len(lc_msgs), 2)
        self.assertIsInstance(lc_msgs[0], SystemMessage)
        self.assertEqual(lc_msgs[0].content, "You are a helpful coding assistant.")

    def test_statistics_and_export(self):
        self.mem.add_user_message("Explain quantum computing")
        self.mem.add_assistant_message(
            "Quantum computing uses qubits.",
            latency_seconds=1.2,
            sources=[{"source": "paper.pdf", "page": 1, "score": 0.95, "snippet": "qubits..."}]
        )

        stats = self.mem.get_statistics()
        self.assertEqual(stats["total_messages"], 2)
        self.assertEqual(stats["user_messages"], 1)
        self.assertEqual(stats["assistant_messages"], 1)
        self.assertGreater(stats["estimated_tokens"], 0)
        self.assertEqual(stats["avg_latency_seconds"], 1.2)

        json_export = self.mem.export_json()
        self.assertIn("quantum computing", json_export)
        self.assertIn("paper.pdf", json_export)

        md_export = self.mem.export_markdown()
        self.assertIn("# Conversation Transcript", md_export)
        self.assertIn("Explain quantum computing", md_export)
        self.assertIn("paper.pdf", md_export)

    def test_clear_memory(self):
        self.mem.add_user_message("Query")
        self.mem.clear()
        self.assertEqual(len(self.mem.get_messages()), 0)


if __name__ == "__main__":
    unittest.main()

