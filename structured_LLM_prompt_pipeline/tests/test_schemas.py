"""Unit tests for Pydantic document schemas and validation constraints."""

import pytest
from pydantic import ValidationError
from schemas import (
    InvoiceExtraction,
    VendorInfo,
    CustomerInfo,
    InvoiceMetadata,
    LineItem,
    FinancialSummary,
    ResumeExtraction,
    PersonalInfo,
    WorkExperience,
    Education,
    SkillSet,
    SeniorityLevel,
)


def test_valid_invoice_schema():
    """Verify that a compliant invoice data payload passes validation cleanly."""
    data = {
        "vendor": {"name": "Acme Corp", "email": "billing@acme.com"},
        "customer": {"name": "Beta LLC", "email": "ap@beta.com"},
        "metadata": {
            "invoice_number": "INV-001",
            "issue_date": "2024-10-01",
            "due_date": "2024-10-31",
            "currency": "USD"
        },
        "line_items": [
            {
                "description": "Consulting Services",
                "quantity": 10.0,
                "unit_price": 100.0,
                "line_total": 1000.0
            }
        ],
        "financials": {
            "subtotal": 1000.0,
            "tax_amount": 80.0,
            "discount_amount": 0.0,
            "shipping_fee": 0.0,
            "grand_total": 1080.0,
            "currency": "USD"
        }
    }
    invoice = InvoiceExtraction.model_validate(data)
    assert invoice.vendor.name == "Acme Corp"
    assert invoice.financials.grand_total == 1080.0
    assert invoice.math_discrepancy_detected is False


def test_invoice_math_discrepancy_flag():
    """Verify that math discrepancies between lines and totals trigger the detection flag."""
    data = {
        "vendor": {"name": "Discrepancy Store"},
        "customer": {"name": "Shopper"},
        "metadata": {"invoice_number": "INV-DISC-01", "issue_date": "2024-10-01", "currency": "USD"},
        "line_items": [
            {"description": "Item A", "quantity": 1.0, "unit_price": 50.0, "line_total": 50.0}
        ],
        "financials": {
            "subtotal": 100.0,  # Line item total (50) does not match subtotal (100)
            "grand_total": 100.0,
            "currency": "USD"
        }
    }
    invoice = InvoiceExtraction.model_validate(data)
    assert invoice.math_discrepancy_detected is True


def test_invoice_missing_required_field_raises():
    """Verify that omitting a mandatory field raises ValidationError."""
    data = {
        "vendor": {},  # Missing required 'name'
        "customer": {"name": "Beta LLC"},
        "metadata": {"invoice_number": "INV-001", "issue_date": "2024-10-01", "currency": "USD"},
        "financials": {"subtotal": 0.0, "grand_total": 0.0, "currency": "USD"}
    }
    with pytest.raises(ValidationError):
        InvoiceExtraction.model_validate(data)


def test_resume_schema_and_email_normalization():
    """Verify resume schema validation and email normalization validator."""
    data = {
        "personal_info": {
            "full_name": "Jordan Rivera",
            "email": "  JORDAN.RIVERA@Example.COM  ",
            "phone": "555-0199"
        },
        "professional_summary": "Senior Software Engineer with 6 years experience.",
        "seniority_level": "Senior",
        "total_years_experience": 6.0,
        "skills": {
            "programming_languages": ["Python", "Rust"],
            "frameworks_and_libraries": ["FastAPI", "Pydantic"]
        }
    }
    resume = ResumeExtraction.model_validate(data)
    assert resume.personal_info.full_name == "Jordan Rivera"
    assert resume.personal_info.email == "jordan.rivera@example.com"
    assert resume.seniority_level == SeniorityLevel.SENIOR
