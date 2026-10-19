"""Unit and integration tests for StructuredPromptPipeline."""

from core.pipeline import StructuredPromptPipeline
from providers import MockProvider


def test_pipeline_invoice_extraction():
    """Verify end-to-end invoice extraction with mock provider."""
    provider = MockProvider(simulate_latency_sec=0.01)
    pipeline = StructuredPromptPipeline(provider=provider)

    sample_text = "INVOICE #INV-2024-9042 from OmniTech Solutions Inc. Billed to Global Horizon. Subtotal $6300 Total $6504."
    result = pipeline.extract(domain="invoice", source=sample_text)

    assert result.success is True
    assert result.domain == "invoice"
    assert result.validated_data is not None
    assert result.validated_data["metadata"]["invoice_number"] == "INV-2024-9042"
    assert result.attempts == 1
    assert len(result.audit_trail) == 1


def test_pipeline_resume_extraction():
    """Verify end-to-end resume extraction with mock provider."""
    provider = MockProvider(simulate_latency_sec=0.01)
    pipeline = StructuredPromptPipeline(provider=provider)

    sample_text = "Marcus Vance - Principal Generative AI Architect with 7+ years of experience in LangChain and PyTorch."
    result = pipeline.extract(domain="resume", source=sample_text)

    assert result.success is True
    assert result.domain == "resume"
    assert result.validated_data is not None
    assert result.validated_data["personal_info"]["full_name"] == "Marcus Vance"
    assert result.validated_data["seniority_level"] == "Lead"


def test_pipeline_self_correction_flow():
    """Verify that intentional failure triggers self-correction and succeeds on retry."""
    provider = MockProvider(simulate_latency_sec=0.01, fail_once_for_demo=True)
    pipeline = StructuredPromptPipeline(provider=provider, max_retries=2)

    sample_text = "Invoice #INV-2024-9042 Total $6504."
    result = pipeline.extract(domain="invoice", source=sample_text)

    assert result.success is True
    # Attempt 1 failed with validation error, Attempt 2 succeeded via self-correction!
    assert result.attempts == 2
    assert len(result.audit_trail) == 2
    assert result.audit_trail[0]["status"] == "FAILED"
    assert result.audit_trail[1]["status"] == "SUCCESS"
