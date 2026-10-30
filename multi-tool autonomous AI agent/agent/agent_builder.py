"""
agent/agent_builder.py
======================
Builds and returns a LangChain tool-calling agent executor.
Fast, reliable, and native function calling for Gemini, GPT, and Claude.
"""

import os
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate

from agent.llm_factory import create_llm
from tools import get_all_tools

SYSTEM_PROMPT = """You are an autonomous AI agent equipped with tools for web search, Python math calculation, and reading local files.

Guidelines:
- For real-time facts, current news, or live data: use `web_search`.
- For math computations, equations, or algebra: use `python_calculator`.
- For local files or directories: use `file_reader`.
- If you can answer directly from common knowledge, do so directly and clearly.
- Provide comprehensive, well-formatted final answers."""


def build_agent_executor() -> AgentExecutor:
    """Build a fast, robust tool-calling AgentExecutor."""
    llm = create_llm()
    tools = get_all_tools()
    verbose = os.getenv("VERBOSE", "true").lower() == "true"
    max_iterations = int(os.getenv("MAX_ITERATIONS", "5"))

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        output_key="output",
        k=5,
        return_messages=True,
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=verbose,
        max_iterations=max_iterations,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        early_stopping_method="generate",
        max_execution_time=60,
    )

    return executor
