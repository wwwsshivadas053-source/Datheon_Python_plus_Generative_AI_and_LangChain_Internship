"""
Unit tests for LLMManager and streaming generators.
"""

import unittest
from core.llm_manager import LLMManager
from core.agent_tools import AgentToolbox
from langchain_core.messages import HumanMessage, SystemMessage


class TestLLMManager(unittest.TestCase):

    def test_demo_streaming_generator(self):
        llm = LLMManager(provider="Demo Mode (No API Key Required)", model_name="demo-fast")
        self.assertTrue(llm.is_api_key_configured())

        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello world")
        ]

        chunks = list(llm.stream_response(messages))
        self.assertGreater(len(chunks), 0)
        full_output = "".join(chunks)
        self.assertIn("Full-Stack AI Nexus", full_output)

    def test_demo_streaming_with_rag_context(self):
        llm = LLMManager(provider="Demo Mode (No API Key Required)", model_name="demo-fast")

        messages = [
            HumanMessage(content="What are the system specifications?")
        ]
        rag_context = (
            "--- RETRIEVED DOCUMENTATION CONTEXT ---\n"
            "[Source Document 1: spec.pdf | Match Score: 95%]\n"
            "Nexus operates on port 8501 with 50,000 maximum chunks.\n"
            "--- END OF RETRIEVED CONTEXT ---"
        )

        chunks = list(llm.stream_response(messages, rag_context=rag_context))
        self.assertGreater(len(chunks), 0)
        full_output = "".join(chunks)
        self.assertIn("Nexus operates on port 8501", full_output)

    def test_agent_toolbox(self):
        # Test calculator
        calc_res = AgentToolbox.calculate("12 * 12 + 6")
        self.assertEqual(calc_res, "150")

        # Test calculator restricted expressions
        unsafe_res = AgentToolbox.calculate("__import__('os').system('ls')")
        self.assertIn("Restricted", unsafe_res)

        # Test system time
        sys_time = AgentToolbox.get_system_time()
        self.assertIsInstance(sys_time, str)
        self.assertGreater(len(sys_time), 10)


if __name__ == "__main__":
    unittest.main()

