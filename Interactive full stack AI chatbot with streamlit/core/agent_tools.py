"""
Agent Tools Module.
Provides modular utility tools for Agent execution pipelines:
- Python Sandboxed Calculator / Evaluator
- Text summarization helper
- DuckDuckGo web search integration (with safe fallback)
"""

from typing import Dict, Any, List
import math
import datetime

try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False


class AgentToolbox:
    """Collection of agent utilities and execution tools."""

    @staticmethod
    def calculate(expression: str) -> str:
        """
        Safely evaluates mathematical expressions using math namespace.
        """
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        allowed_names.update({
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow
        })

        try:
            # Strip unsafe constructs
            clean_expr = expression.strip().replace("^", "**")
            if any(term in clean_expr for term in ["import", "exec", "eval", "os", "sys", "open", "__"]):
                return "Error: Restricted expression."
            result = eval(clean_expr, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as e:
            return f"Calculation error: {str(e)}"

    @staticmethod
    def get_system_time() -> str:
        """Return formatted current date and time."""
        now = datetime.datetime.now()
        return now.strftime("%A, %B %d, %Y - %H:%M:%S")

    @staticmethod
    def web_search(query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """
        Performs web search via DuckDuckGo.
        """
        if not HAS_DDGS or not query.strip():
            return [{
                "title": f"Web search simulation for '{query}'",
                "href": "https://duckduckgo.com",
                "body": f"Simulated web results for query: {query}. Install duckduckgo-search for live live queries."
            }]

        try:
            results = []
            with DDGS() as ddgs:
                raw_results = ddgs.text(query, max_results=max_results)
                for r in raw_results:
                    results.append({
                        "title": r.get("title", ""),
                        "href": r.get("href", ""),
                        "body": r.get("body", "")
                    })
            return results
        except Exception as e:
            return [{
                "title": f"Search fallback for '{query}'",
                "href": "https://duckduckgo.com",
                "body": f"Unable to reach live search service: {str(e)}"
            }]

