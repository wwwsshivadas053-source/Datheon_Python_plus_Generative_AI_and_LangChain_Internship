"""
tools/calculator_tool.py
========================
A safe Python math calculator tool for the autonomous agent.

Features:
  - Arithmetic: +, -, *, /, //, %, **
  - Math functions: sqrt, log, sin, cos, tan, pi, e, abs, round, etc.
  - Symbolic algebra via sympy (solve, simplify, derivative, integral)
  - Numeric evaluation via numexpr (fast array math)
  - Safe: no access to builtins like open, exec, eval, import
"""

import math
import operator
import re
from typing import Union

try:
    import sympy
    from sympy import (
        symbols, solve, simplify, expand, factor,
        diff, integrate, limit, series, sqrt as sp_sqrt,
        log as sp_log, sin as sp_sin, cos as sp_cos, tan as sp_tan,
        pi as sp_pi, E as sp_e, sympify, latex, Rational
    )
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

from langchain_core.tools import Tool


# --------------------------------------------------------------------------- #
#  Safe math namespace (whitelist only)                                         #
# --------------------------------------------------------------------------- #
SAFE_MATH_NS: dict = {
    "__builtins__": {},          # block all Python builtins
    # math constants
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
    "tau": math.tau,
    # math functions
    "sqrt": math.sqrt,
    "cbrt": lambda x: x ** (1 / 3),
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "degrees": math.degrees,
    "radians": math.radians,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "lcm": getattr(math, "lcm", lambda a, b: abs(a * b) // math.gcd(a, b)),
    "ceil": math.ceil,
    "floor": math.floor,
    "round": round,
    "abs": abs,
    "pow": pow,
    "min": min,
    "max": max,
    "sum": sum,
    "divmod": divmod,
}

# Add sympy symbols if available
if SYMPY_AVAILABLE:
    SAFE_MATH_NS.update({
        "x": symbols("x"),
        "y": symbols("y"),
        "z": symbols("z"),
        "n": symbols("n"),
        "solve": solve,
        "simplify": simplify,
        "expand": expand,
        "factor": factor,
        "diff": diff,
        "integrate": integrate,
        "limit": limit,
        "series": series,
        "symbols": symbols,
        "Rational": Rational,
    })


def _safe_eval(expression: str) -> str:
    """
    Evaluate a mathematical expression inside a restricted namespace.
    Returns the result as a string or an informative error message.
    """
    expression = expression.strip()

    # Block obvious injection attempts
    forbidden = ["import", "exec", "eval", "__", "open", "os.", "sys.", "subprocess"]
    for kw in forbidden:
        if kw in expression:
            return f"❌ Forbidden keyword detected: '{kw}'. Only math expressions are allowed."

    try:
        result = eval(expression, SAFE_MATH_NS)  # noqa: S307 – controlled namespace
        # Pretty-print sympy objects
        if SYMPY_AVAILABLE and hasattr(result, "_sympy_"):
            return str(result)
        if isinstance(result, float):
            # Show up to 10 significant digits
            return f"{result:.10g}"
        return str(result)
    except ZeroDivisionError:
        return "❌ Math Error: Division by zero."
    except OverflowError:
        return "❌ Math Error: Result is too large (overflow)."
    except Exception as exc:
        return f"❌ Calculation Error: {exc}"


def _calculator_func(expression: str) -> str:
    """
    Main entry point for the calculator tool.
    Handles multi-line expressions and returns formatted results.
    """
    lines = [line.strip() for line in expression.strip().splitlines() if line.strip()]
    if not lines:
        return "❌ No expression provided. Please enter a math expression."

    if len(lines) == 1:
        result = _safe_eval(lines[0])
        return f"Result: {result}"

    # Multi-expression batch mode
    results = []
    for i, line in enumerate(lines, 1):
        res = _safe_eval(line)
        results.append(f"  [{i}] {line}  ->  {res}")
    return "Batch Results:\n" + "\n".join(results)


def get_calculator_tool() -> Tool:
    """Returns a LangChain Tool wrapping the safe Python calculator."""
    return Tool(
        name="python_calculator",
        func=_calculator_func,
        description=(
            "A safe Python math calculator. "
            "Input: a mathematical expression string (e.g., '2 ** 10', 'sqrt(144)', 'log(100, 10)'). "
            "Supports: arithmetic, powers, roots, logarithms, trigonometry, factorials, GCD/LCM. "
            "Also supports symbolic math: 'solve(x**2 - 4, x)', 'diff(x**3, x)', 'integrate(x**2, x)'. "
            "Multi-line input runs each line as a separate expression. "
            "Output: computed numeric or symbolic result. "
            "Use for ANY math calculation, unit conversion formulas, or algebraic problems."
        ),
    )
