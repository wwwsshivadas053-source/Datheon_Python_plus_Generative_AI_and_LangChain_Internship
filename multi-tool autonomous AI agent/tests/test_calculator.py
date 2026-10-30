"""
tests/test_calculator.py
========================
Unit tests for the safe Python calculator tool.
Tests arithmetic, math functions, and symbolic algebra.

Run with:
    python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from tools.calculator_tool import _safe_eval, _calculator_func, get_calculator_tool


class TestSafeEval:
    """Unit tests for _safe_eval (the core eval engine)."""

    # ── Basic arithmetic ────────────────────────────────────────────────────
    def test_addition(self):
        assert _safe_eval("2 + 3") == "5"

    def test_subtraction(self):
        assert _safe_eval("100 - 42") == "58"

    def test_multiplication(self):
        assert _safe_eval("7 * 8") == "56"

    def test_division(self):
        result = _safe_eval("10 / 4")
        assert result == "2.5"

    def test_integer_division(self):
        assert _safe_eval("17 // 5") == "3"

    def test_modulo(self):
        assert _safe_eval("17 % 5") == "2"

    def test_power(self):
        assert _safe_eval("2 ** 10") == "1024"

    # ── Math functions ──────────────────────────────────────────────────────
    def test_sqrt(self):
        assert _safe_eval("sqrt(144)") == "12"

    def test_log10(self):
        assert _safe_eval("log10(1000)") == "3"

    def test_log_natural(self):
        import math
        result = float(_safe_eval("log(e)"))
        assert abs(result - 1.0) < 1e-9

    def test_sin_pi(self):
        result = float(_safe_eval("sin(pi)"))
        assert abs(result) < 1e-10

    def test_cos_zero(self):
        assert _safe_eval("cos(0)") == "1"

    def test_factorial(self):
        assert _safe_eval("factorial(5)") == "120"

    def test_abs(self):
        assert _safe_eval("abs(-42)") == "42"

    def test_ceil(self):
        assert _safe_eval("ceil(3.2)") == "4"

    def test_floor(self):
        assert _safe_eval("floor(3.9)") == "3"

    def test_pi_constant(self):
        result = float(_safe_eval("pi"))
        assert abs(result - 3.14159265) < 1e-6

    def test_compound_expression(self):
        result = float(_safe_eval("sqrt(9) + log10(100) * 3"))
        assert abs(result - 9.0) < 1e-9   # sqrt(9)=3, log10(100)=2, 2*3=6 → 3+6=9

    # ── Division by zero ────────────────────────────────────────────────────
    def test_division_by_zero(self):
        result = _safe_eval("1 / 0")
        assert "Division by zero" in result or "❌" in result

    # ── Security: forbidden keywords ────────────────────────────────────────
    def test_import_blocked(self):
        result = _safe_eval("import os")
        assert "❌" in result

    def test_exec_blocked(self):
        result = _safe_eval("exec('print(1)')")
        assert "❌" in result

    def test_dunder_blocked(self):
        result = _safe_eval("__import__('os')")
        assert "❌" in result

    def test_open_blocked(self):
        result = _safe_eval("open('test.txt')")
        assert "❌" in result


class TestCalculatorFunc:
    """Tests for the _calculator_func dispatcher."""

    def test_single_expression(self):
        result = _calculator_func("2 + 2")
        assert "4" in result
        assert "Result:" in result

    def test_empty_input(self):
        result = _calculator_func("")
        assert "No expression" in result or "❌" in result

    def test_batch_mode(self):
        result = _calculator_func("2 + 2\n3 * 3\n4 ** 2")
        assert "4" in result
        assert "9" in result
        assert "16" in result

    def test_whitespace_only(self):
        result = _calculator_func("   \n   ")
        assert "No expression" in result or "❌" in result


class TestCalculatorTool:
    """Tests for the LangChain Tool wrapper."""

    def test_tool_creation(self):
        tool = get_calculator_tool()
        assert tool is not None
        assert tool.name == "python_calculator"
        assert callable(tool.func)
        assert len(tool.description) > 50

    def test_tool_invocation(self):
        tool = get_calculator_tool()
        result = tool.func("sqrt(16)")
        assert "4" in result

    def test_tool_handles_error_gracefully(self):
        tool = get_calculator_tool()
        result = tool.func("import os; os.getcwd()")
        assert "❌" in result or "Forbidden" in result

