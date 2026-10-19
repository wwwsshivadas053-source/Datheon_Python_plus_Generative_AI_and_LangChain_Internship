"""Unit tests for PromptBuilder."""

from prompts import PromptBuilder


def test_prompt_builder_invoice():
    """Verify prompt builder for invoice domain."""
    builder = PromptBuilder(domain="invoice", enable_few_shot=True)
    schema_str = builder.get_json_schema_str()
    assert "InvoiceExtraction" in schema_str
    assert "vendor" in schema_str

    user_prompt = builder.build_user_prompt("Sample raw invoice text")
    assert "TARGET JSON SCHEMA:" in user_prompt
    assert "FEW-SHOT EXTRACTION DEMONSTRATION" in user_prompt
    assert "Sample raw invoice text" in user_prompt


def test_prompt_builder_few_shot_disabled():
    """Verify few-shot section is omitted when disabled."""
    builder = PromptBuilder(domain="resume", enable_few_shot=False)
    user_prompt = builder.build_user_prompt("Sample raw resume text")
    assert "FEW-SHOT EXTRACTION DEMONSTRATION" not in user_prompt


def test_prompt_builder_self_correction():
    """Verify self-correction prompt construction."""
    builder = PromptBuilder(domain="invoice")
    correction_prompt = builder.build_self_correction_prompt(
        unstructured_text="Invoice #123 Total $50",
        previous_output='{"total": "fifty"}',
        validation_errors="Input should be a valid number",
    )
    assert "FAILED strict Pydantic validation rules" in correction_prompt
    assert "Input should be a valid number" in correction_prompt
    assert '{"total": "fifty"}' in correction_prompt
