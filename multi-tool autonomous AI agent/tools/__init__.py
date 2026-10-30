"""
tools/__init__.py
=================
Exports all agent tools as a ready-to-use list.
"""

from tools.search_tool import get_search_tool
from tools.calculator_tool import get_calculator_tool
from tools.file_reader_tool import get_file_reader_tool


def get_all_tools() -> list:
    """
    Instantiate and return all available agent tools.

    Returns:
        list[Tool]: [web_search, python_calculator, file_reader]
    """
    return [
        get_search_tool(),
        get_calculator_tool(),
        get_file_reader_tool(),
    ]


__all__ = ["get_all_tools", "get_search_tool", "get_calculator_tool", "get_file_reader_tool"]

