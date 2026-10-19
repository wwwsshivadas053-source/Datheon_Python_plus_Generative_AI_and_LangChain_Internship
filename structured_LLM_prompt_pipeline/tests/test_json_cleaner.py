"""Unit tests for JSONCleaner extraction, sanitization, and healing."""

import pytest
from core.json_cleaner import JSONCleaner


def test_clean_json_direct():
    """Verify parsing clean JSON."""
    raw = '{"name": "Alice", "age": 30}'
    data, cleaned = JSONCleaner.extract_and_parse(raw)
    assert data["name"] == "Alice"
    assert data["age"] == 30


def test_clean_json_in_markdown_fences():
    """Verify extraction of JSON inside ```json ... ``` markdown code block."""
    raw = """Here is the extracted invoice:
```json
{
  "vendor": "Acme",
  "amount": 150.50
}
```
Let me know if you need anything else!"""
    data, cleaned = JSONCleaner.extract_and_parse(raw)
    assert data["vendor"] == "Acme"
    assert data["amount"] == 150.50


def test_clean_json_with_trailing_commas_and_smart_quotes():
    """Verify healing of smart quotes and trailing commas."""
    raw = """{
  “vendor”: “MegaCorp”,
  “items”: [
    “Item 1”,
    “Item 2”,
  ],
  “total”: 200.0,
}"""
    data, cleaned = JSONCleaner.extract_and_parse(raw)
    assert data["vendor"] == "MegaCorp"
    assert len(data["items"]) == 2
    assert data["total"] == 200.0


def test_clean_json_with_outermost_braces():
    """Verify extraction when surrounded by conversational text without markdown tags."""
    raw = 'Sure thing! The data is: {"status": "ok", "count": 5} Hope that helps.'
    data, cleaned = JSONCleaner.extract_and_parse(raw)
    assert data["status"] == "ok"
    assert data["count"] == 5


def test_invalid_json_raises_value_error():
    """Verify that completely unparseable text raises ValueError."""
    raw = "This is just random text with no JSON at all."
    with pytest.raises(ValueError) as exc:
        JSONCleaner.extract_and_parse(raw)
    assert "Failed to extract valid JSON" in str(exc.value)
