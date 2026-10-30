"""
tests/test_file_reader.py
=========================
Unit tests for the local file reader tool.
Tests text reading, safety checks, error handling, and the LangChain tool wrapper.

Run with:
    python -m pytest tests/ -v
"""

import sys
import os
import json
import tempfile
import pathlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from tools.file_reader_tool import _read_file, _is_safe_path, get_file_reader_tool


class TestSafePathCheck:
    """Tests for the path safety validator."""

    def test_valid_path(self):
        is_safe, _ = _is_safe_path("C:/Users/test/file.txt")
        assert is_safe is True

    def test_traversal_blocked(self):
        is_safe, msg = _is_safe_path("C:/Users/../secret/file.txt")
        assert is_safe is False
        assert "traversal" in msg.lower() or "❌" in msg


class TestReadFile:
    """Tests for the _read_file dispatcher."""

    def test_nonexistent_file(self):
        result = _read_file("C:/this/file/does/not/exist.txt")
        assert "not found" in result.lower() or "❌" in result

    def test_read_plain_text(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Hello from unit test!\nLine 2 here.")
            tmp_path = f.name
        try:
            result = _read_file(tmp_path)
            assert "Hello from unit test!" in result
        finally:
            os.unlink(tmp_path)

    def test_read_json_file(self):
        data = {"name": "agent", "version": 1, "tools": ["search", "calc", "reader"]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            tmp_path = f.name
        try:
            result = _read_file(tmp_path)
            assert "agent" in result
            assert "tools" in result
        finally:
            os.unlink(tmp_path)

    def test_read_markdown_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("# Test Header\n\nThis is a **markdown** file.")
            tmp_path = f.name
        try:
            result = _read_file(tmp_path)
            assert "Test Header" in result
        finally:
            os.unlink(tmp_path)

    def test_directory_listing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a couple of files
            (pathlib.Path(tmp_dir) / "file1.txt").write_text("hello")
            (pathlib.Path(tmp_dir) / "file2.py").write_text("print(1)")
            result = _read_file(tmp_dir)
            assert "file1.txt" in result or "Directory" in result

    def test_traversal_in_path_blocked(self):
        result = _read_file("C:/windows/../system32/evil.txt")
        # Should either block due to traversal check or return not found
        assert "traversal" in result.lower() or "❌" in result or "not found" in result.lower()

    def test_format_header_present(self):
        """Check that output includes the formatted header with file info."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Content here")
            tmp_path = f.name
        try:
            result = _read_file(tmp_path)
            assert "KB" in result   # size shown in KB
        finally:
            os.unlink(tmp_path)


class TestFileReaderTool:
    """Tests for the LangChain Tool wrapper."""

    def test_tool_creation(self):
        tool = get_file_reader_tool()
        assert tool is not None
        assert tool.name == "file_reader"
        assert callable(tool.func)
        assert len(tool.description) > 50

    def test_tool_nonexistent_file(self):
        tool = get_file_reader_tool()
        result = tool.func("/nonexistent/path/file.xyz")
        assert "not found" in result.lower() or "❌" in result

    def test_tool_reads_text(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Tool test content — reading via LangChain tool wrapper.")
            tmp_path = f.name
        try:
            tool = get_file_reader_tool()
            result = tool.func(tmp_path)
            assert "Tool test content" in result
        finally:
            os.unlink(tmp_path)

