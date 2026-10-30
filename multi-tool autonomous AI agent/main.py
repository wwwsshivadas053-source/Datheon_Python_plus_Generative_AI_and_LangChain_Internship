"""
main.py
=======
Entry point for the Multi-Tool Autonomous AI Agent.

Run:
    python main.py

Interactive console loop that accepts natural-language queries,
routes them through the LangChain ReAct agent, and displays the
full reasoning chain (Thought → Action → Observation → Final Answer).
"""

import os
import sys

# ── Load .env before anything else ──────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

# ── Project imports ──────────────────────────────────────────────────────────
from agent.agent_builder import build_agent_executor
from tools import get_all_tools
from utils.display import (
    print_banner,
    print_section,
    display_intermediate_steps,
    display_final_answer,
    display_error,
    print_help,
    print_examples,
    print_tools_list,
    cprint,
    Fore,
    Style,
)


# ── Constants ────────────────────────────────────────────────────────────────
EXIT_COMMANDS  = {"/exit", "/quit", "/q", "exit", "quit", "bye", "goodbye"}
CLEAR_COMMANDS = {"/clear", "/reset", "/new"}
HELP_COMMANDS  = {"/help", "/h", "help", "?"}
TOOL_COMMANDS  = {"/tools", "/tool"}
HIST_COMMANDS  = {"/history", "/hist"}
EXAM_COMMANDS  = {"/examples", "/example", "/ex"}


# ── Run a single query ───────────────────────────────────────────────────────
def run_query(executor, query: str) -> None:
    """
    Send a query to the agent executor and display the full response.
    Handles errors gracefully.
    """
    print_section(f"Query: {query}", icon="❓")

    try:
        result = executor.invoke({"input": query})
    except KeyboardInterrupt:
        cprint("\n  ⚠️  Query interrupted by user.", Fore.YELLOW)
        return
    except Exception as exc:
        display_error(str(exc))
        if os.getenv("DEBUG", "false").lower() == "true":
            import traceback
            traceback.print_exc()
        return

    # Display reasoning chain
    steps = result.get("intermediate_steps", [])
    if steps:
        display_intermediate_steps(steps)

    # Display final answer
    answer = result.get("output", "No answer returned.")
    display_final_answer(answer)


# ── Main REPL loop ───────────────────────────────────────────────────────────
def main() -> None:
    """Main interactive loop for the AI agent console."""
    print_banner()

    # Build agent (validates API keys + loads tools)
    cprint("  ⏳ Initialising agent ...", Fore.YELLOW)
    try:
        executor = build_agent_executor()
    except (EnvironmentError, ImportError) as exc:
        display_error(str(exc))
        cprint("\n  Refer to .env.example for configuration instructions.", Fore.YELLOW)
        sys.exit(1)

    tools = get_all_tools()
    provider = os.getenv("LLM_PROVIDER", "google").upper()
    model    = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    cprint(f"  ✅ Agent ready  │  LLM: {provider} / {model}  │  Tools: {len(tools)}", Fore.GREEN, Style.BRIGHT)
    cprint("  Type /help for available commands or /examples to see sample queries.", Fore.WHITE)
    print()

    session_query_count = 0

    while True:
        try:
            user_input = input(
                f"{Fore.CYAN if hasattr(Fore, 'CYAN') else ''}  🤖 You > "
                f"{Style.RESET_ALL if hasattr(Style, 'RESET_ALL') else ''}"
            ).strip()
        except (KeyboardInterrupt, EOFError):
            cprint("\n\n  👋 Goodbye! Session ended.", Fore.CYAN, Style.BRIGHT)
            break

        if not user_input:
            continue

        lower = user_input.lower()

        # ── Special commands ────────────────────────────────────────────────
        if lower in EXIT_COMMANDS:
            cprint(f"\n  👋 Goodbye! Processed {session_query_count} queries this session.", Fore.CYAN, Style.BRIGHT)
            break

        elif lower in HELP_COMMANDS:
            print_help()

        elif lower in TOOL_COMMANDS:
            print_tools_list(tools)

        elif lower in EXAM_COMMANDS:
            print_examples()

        elif lower in CLEAR_COMMANDS:
            if hasattr(executor, "memory") and executor.memory:
                executor.memory.clear()
            cprint("  🗑️  Memory cleared. Starting a fresh conversation.", Fore.YELLOW)

        elif lower in HIST_COMMANDS:
            if hasattr(executor, "memory") and executor.memory:
                history = executor.memory.load_memory_variables({}).get("chat_history", [])
                if history:
                    cprint("\n  📜 Conversation History:", Fore.YELLOW, Style.BRIGHT)
                    for msg in history:
                        role = "You" if msg.type == "human" else "Agent"
                        content = str(msg.content)[:200]
                        cprint(f"  [{role}] {content}", Fore.WHITE)
                else:
                    cprint("  No conversation history yet.", Fore.WHITE)
            else:
                cprint("  Memory is not enabled.", Fore.WHITE)

        else:
            # ── Run the agent ───────────────────────────────────────────────
            session_query_count += 1
            run_query(executor, user_input)


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

