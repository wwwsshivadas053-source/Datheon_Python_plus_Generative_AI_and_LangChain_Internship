"""
demo.py
=======
Automated demo script — runs 5 pre-defined queries to showcase
all three tools without requiring interactive input.

Usage:
    python demo.py

Useful for:
  - CI/CD smoke tests
  - Internship demonstrations
  - Verifying your API key setup
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()

from agent.agent_builder import build_agent_executor
from utils.display import (
    print_banner,
    print_section,
    display_intermediate_steps,
    display_final_answer,
    display_error,
    cprint,
    Fore,
    Style,
)

# ── Demo queries covering all three tools ────────────────────────────────────
DEMO_QUERIES = [
    # Tool: python_calculator
    {
        "label": "Calculator — Compound Interest",
        "query": (
            "Calculate the compound interest on a principal of $10,000 "
            "invested at an annual rate of 7% compounded monthly for 5 years. "
            "Use the formula: A = P * (1 + r/n)^(n*t)"
        ),
    },
    # Tool: python_calculator — symbolic math
    {
        "label": "Calculator — Symbolic Algebra",
        "query": "Solve the quadratic equation x^2 - 7x + 12 = 0 and verify the roots.",
    },
    # Tool: web_search
    {
        "label": "Web Search — Current Events",
        "query": "What are the top 3 latest advancements in artificial intelligence in 2025?",
    },
    # Tool: web_search + python_calculator
    {
        "label": "Multi-Tool — Search + Calculation",
        "query": (
            "Search for the approximate speed of light in km/s, "
            "then calculate how many seconds light takes to travel from "
            "the Earth to the Moon (average distance: 384,400 km)."
        ),
    },
    # Tool: file_reader
    {
        "label": "File Reader — Project Files",
        "query": "List the contents of the current project directory.",
    },
]


def run_demo() -> None:
    print_banner()
    cprint("  🎬 AUTOMATED DEMO MODE — showcasing all agent tools\n", Fore.YELLOW, Style.BRIGHT)

    cprint("  ⏳ Initialising agent ...", Fore.YELLOW)
    try:
        executor = build_agent_executor()
    except (EnvironmentError, ImportError) as exc:
        display_error(str(exc))
        cprint("  Refer to .env.example for setup instructions.", Fore.YELLOW)
        sys.exit(1)

    provider = os.getenv("LLM_PROVIDER", "google").upper()
    model    = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    cprint(f"  ✅ Agent ready  │  {provider} / {model}\n", Fore.GREEN, Style.BRIGHT)

    for i, demo in enumerate(DEMO_QUERIES, 1):
        cprint(f"\n{'═' * 70}", Fore.BLUE)
        cprint(f"  DEMO {i}/{len(DEMO_QUERIES)}: {demo['label']}", Fore.YELLOW, Style.BRIGHT)
        cprint(f"{'═' * 70}\n", Fore.BLUE)

        try:
            result = executor.invoke({"input": demo["query"]})
        except KeyboardInterrupt:
            cprint("\n  ⚠️  Demo interrupted.", Fore.YELLOW)
            break
        except Exception as exc:
            display_error(str(exc))
            continue

        steps = result.get("intermediate_steps", [])
        if steps:
            display_intermediate_steps(steps)

        display_final_answer(result.get("output", "No answer returned."))

    cprint(f"\n{'═' * 70}", Fore.BLUE)
    cprint("  ✅ Demo complete! Run  python main.py  for interactive mode.", Fore.GREEN, Style.BRIGHT)
    cprint(f"{'═' * 70}\n", Fore.BLUE)


if __name__ == "__main__":
    run_demo()

