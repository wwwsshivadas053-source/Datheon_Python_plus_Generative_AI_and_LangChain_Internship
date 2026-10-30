"""
utils/display.py
================
Rich console display utilities for the autonomous agent.

Provides colour-coded, structured output for the agent's
reasoning chain (Thought → Action → Observation → Final Answer).
"""

import re
import textwrap
from datetime import datetime

# Try to use colorama for Windows colour support
try:
    from colorama import init, Fore, Style, Back
    init(autoreset=True)
    COLORS = True
except ImportError:
    COLORS = False

    class _Dummy:
        def __getattr__(self, name):
            return ""
    Fore = Style = Back = _Dummy()


# --------------------------------------------------------------------------- #
#  Colour / style helpers                                                       #
# --------------------------------------------------------------------------- #
def _c(text: str, *codes) -> str:
    """Apply colorama codes (no-op if colorama unavailable)."""
    if not COLORS:
        return text
    prefix = "".join(codes)
    return f"{prefix}{text}{Style.RESET_ALL}"


def cprint(text: str, *codes) -> None:
    print(_c(text, *codes))


# --------------------------------------------------------------------------- #
#  Banner                                                                       #
# --------------------------------------------------------------------------- #
BANNER = r"""
  ███╗   ███╗██╗   ██╗██╗  ████████╗██╗    ████████╗ ██████╗  ██████╗ ██╗
  ████╗ ████║██║   ██║██║  ╚══██╔══╝██║    ╚══██╔══╝██╔═══██╗██╔═══██╗██║
  ██╔████╔██║██║   ██║██║     ██║   ██║       ██║   ██║   ██║██║   ██║██║
  ██║╚██╔╝██║██║   ██║██║     ██║   ██║       ██║   ██║   ██║██║   ██║██║
  ██║ ╚═╝ ██║╚██████╔╝███████╗██║   ██║       ██║   ╚██████╔╝╚██████╔╝███████╗
  ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝   ╚═╝       ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
                A U T O N O M O U S   A I   A G E N T
                Powered by LangChain  ·  Python 3  ·  ReAct
"""


def print_banner() -> None:
    """Print the application banner."""
    cprint(BANNER, Fore.CYAN, Style.BRIGHT)
    cprint("  Tools: 🔍 Web Search  🧮 Calculator  📄 File Reader", Fore.WHITE)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cprint(f"  Session started: {now}", Fore.DARK_GREY if hasattr(Fore, 'DARK_GREY') else Fore.WHITE)
    print()


def print_separator(char: str = "─", width: int = 70) -> None:
    cprint(char * width, Fore.BLUE)


def print_section(title: str, icon: str = "▶") -> None:
    print()
    cprint(f"{icon} {title}", Fore.YELLOW, Style.BRIGHT)
    print_separator()


# --------------------------------------------------------------------------- #
#  Intermediate step display                                                    #
# --------------------------------------------------------------------------- #
def display_intermediate_steps(steps: list) -> None:
    """
    Pretty-print the agent's reasoning chain from intermediate_steps.

    Each step is a tuple: (AgentAction, observation_string)
    """
    if not steps:
        return

    print()
    cprint("  ┌─── REASONING CHAIN ───────────────────────────────────────┐", Fore.CYAN)

    for i, (action, observation) in enumerate(steps, 1):
        tool_name = getattr(action, "tool", "unknown")
        tool_input = getattr(action, "tool_input", "")
        log = getattr(action, "log", "")

        # Extract Thought from log (text before "Action:")
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

        print()
        # Step number
        cprint(f"  │  Step {i}", Fore.BLUE, Style.BRIGHT)

        # Thought
        if thought:
            wrapped = textwrap.fill(thought, width=62, subsequent_indent="  │    ")
            cprint(f"  │  💭 Thought: {wrapped}", Fore.WHITE)

        # Action
        cprint(f"  │  ⚡ Action:  {tool_name}", Fore.MAGENTA, Style.BRIGHT)

        # Action Input (truncate if long)
        input_str = str(tool_input)
        if len(input_str) > 120:
            input_str = input_str[:117] + "..."
        cprint(f"  │  📥 Input:   {input_str}", Fore.CYAN)

        # Observation (truncate if very long)
        obs_str = str(observation)
        if len(obs_str) > 300:
            obs_str = obs_str[:297] + "..."
        obs_lines = obs_str.splitlines()
        first_line = obs_lines[0] if obs_lines else ""
        cprint(f"  │  👁  Observe: {first_line}", Fore.GREEN)
        for extra_line in obs_lines[1:3]:
            cprint(f"  │           {extra_line}", Fore.GREEN)
        if len(obs_lines) > 3:
            cprint(f"  │           ... ({len(obs_lines) - 3} more lines)", Fore.GREEN)

    print()
    cprint("  └───────────────────────────────────────────────────────────┘", Fore.CYAN)


# --------------------------------------------------------------------------- #
#  Final answer display                                                         #
# --------------------------------------------------------------------------- #
def display_final_answer(answer: str) -> None:
    """Display the agent's final answer in a highlighted box."""
    print()
    cprint("  ╔═══ FINAL ANSWER ═══════════════════════════════════════════╗", Fore.GREEN, Style.BRIGHT)
    print()

    # Word-wrap the answer
    paragraphs = answer.strip().split("\n")
    for para in paragraphs:
        if not para.strip():
            print()
        else:
            wrapped = textwrap.fill(para, width=66, initial_indent="  ║  ", subsequent_indent="  ║  ")
            cprint(wrapped, Fore.WHITE, Style.BRIGHT)

    print()
    cprint("  ╚═════════════════════════════════════════════════════════════╝", Fore.GREEN, Style.BRIGHT)


# --------------------------------------------------------------------------- #
#  Error display                                                                #
# --------------------------------------------------------------------------- #
def display_error(message: str) -> None:
    """Display an error message."""
    cprint(f"\n  ❌ ERROR: {message}", Fore.RED, Style.BRIGHT)


# --------------------------------------------------------------------------- #
#  Help display                                                                 #
# --------------------------------------------------------------------------- #
HELP_TEXT = """
┌─── AVAILABLE COMMANDS ────────────────────────────────────────┐
│                                                               │
│  Type any question to query the agent.                        │
│                                                               │
│  Special commands:                                            │
│    /help       → Show this help message                       │
│    /tools      → List all available tools                     │
│    /clear      → Clear conversation memory                    │
│    /history    → Show conversation history                    │
│    /examples   → Show example queries                         │
│    /exit, /q   → Quit the application                         │
│                                                               │
│  Example queries:                                             │
│    "What is the latest news about AI?"                        │
│    "Calculate the compound interest on $5000 at 8% for 3y"   │
│    "Read and summarize the file at C:/notes/summary.txt"      │
│    "Search for Python best practices and save key points"     │
│                                                               │
└───────────────────────────────────────────────────────────────┘
"""

EXAMPLES_TEXT = """
┌─── EXAMPLE QUERIES ───────────────────────────────────────────┐
│                                                               │
│  🔍 Web Search:                                               │
│    "What is the current price of Bitcoin?"                    │
│    "Latest developments in large language models"             │
│    "How does the Python GIL work?"                            │
│                                                               │
│  🧮 Calculator:                                               │
│    "What is 2^32?"                                            │
│    "Calculate sqrt(144) + log(1000, 10)"                      │
│    "Solve x^2 - 5x + 6 = 0"                                   │
│    "What is the derivative of x^3 + 2x?"                      │
│                                                               │
│  📄 File Reader:                                              │
│    "Read the file at C:/Users/me/notes.txt"                   │
│    "Summarize the PDF at D:/reports/q3.pdf"                   │
│    "What's in requirements.txt in the current folder?"        │
│                                                               │
│  🔀 Combined (multi-tool):                                    │
│    "Search for the GDP of India and convert it to EUR"        │
│    "Read my data.csv and calculate the average of column A"   │
│                                                               │
└───────────────────────────────────────────────────────────────┘
"""


def print_help() -> None:
    cprint(HELP_TEXT, Fore.CYAN)


def print_examples() -> None:
    cprint(EXAMPLES_TEXT, Fore.CYAN)


def print_tools_list(tools: list) -> None:
    """Display the list of registered tools."""
    print()
    cprint("  🛠  Registered Tools:", Fore.YELLOW, Style.BRIGHT)
    for i, tool in enumerate(tools, 1):
        cprint(f"    {i}. {tool.name}", Fore.MAGENTA, Style.BRIGHT)
        desc_lines = textwrap.wrap(tool.description, width=60)
        for line in desc_lines[:2]:
            cprint(f"       {line}", Fore.WHITE)
    print()

