"""
tools/search_tool.py
====================
Web-search tool — optimised for speed.
DuckDuckGo: max 3 results, 8s timeout.
Tavily: max 3 results.
"""

import os
from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper


def get_search_tool() -> Tool:
    """Returns a fast web-search LangChain Tool."""
    provider = os.getenv("SEARCH_PROVIDER", "duckduckgo").lower()

    if provider == "tavily":
        tavily_key = os.getenv("TAVILY_API_KEY")
        if not tavily_key:
            raise EnvironmentError(
                "TAVILY_API_KEY is required when SEARCH_PROVIDER=tavily.\n"
                "Set it in .env or use SEARCH_PROVIDER=duckduckgo."
            )
        from langchain_community.tools.tavily_search import TavilySearchResults
        search_tool = TavilySearchResults(
            max_results=3,           # ← was 5, now 3 (faster)
            tavily_api_key=tavily_key,
        )
        return Tool(
            name="web_search",
            func=search_tool.invoke,
            description=(
                "Real-time web search via Tavily AI. "
                "Input: search query string. "
                "Use for current events, news, or facts."
            ),
        )

    # Default: DuckDuckGo — max 3 results, 8-second timeout
    wrapper = DuckDuckGoSearchAPIWrapper(
        max_results=3,             # ← was 5, now 3 (fewer = faster)
        time="y",                  # limit to past year for relevance
    )
    ddg_search = DuckDuckGoSearchResults(
        api_wrapper=wrapper,
        output_format="list",
        num_results=3,
    )

    return Tool(
        name="web_search",
        func=ddg_search.run,
        description=(
            "Real-time web search via DuckDuckGo. "
            "Input: a concise search query string. "
            "Output: top 3 results with titles, URLs, snippets. "
            "Use for current events, news, weather, or facts."
        ),
    )

