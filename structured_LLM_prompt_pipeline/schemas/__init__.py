"""Schema exports and registry for structured LLM prompt pipeline."""

from typing import Dict, Type
from pydantic import BaseModel
from .invoice_schema import (
    InvoiceExtraction,
    VendorInfo,
    CustomerInfo,
    InvoiceMetadata,
    LineItem,
    FinancialSummary,
    PaymentDetails,
)
from .resume_schema import (
    ResumeExtraction,
    PersonalInfo,
    WorkExperience,
    Education,
    SkillSet,
    Certification,
    Project,
    SeniorityLevel,
)

SCHEMA_REGISTRY: Dict[str, Type[BaseModel]] = {
    "invoice": InvoiceExtraction,
    "resume": ResumeExtraction,
}


def get_schema_class(domain: str) -> Type[BaseModel]:
    """Retrieve the Pydantic schema model for a given domain."""
    normalized = domain.lower().strip()
    if normalized not in SCHEMA_REGISTRY:
        raise ValueError(
            f"Unsupported domain '{domain}'. Available domains: {list(SCHEMA_REGISTRY.keys())}"
        )
    return SCHEMA_REGISTRY[normalized]


__all__ = [
    "SCHEMA_REGISTRY",
    "get_schema_class",
    "InvoiceExtraction",
    "VendorInfo",
    "CustomerInfo",
    "InvoiceMetadata",
    "LineItem",
    "FinancialSummary",
    "PaymentDetails",
    "ResumeExtraction",
    "PersonalInfo",
    "WorkExperience",
    "Education",
    "SkillSet",
    "Certification",
    "Project",
    "SeniorityLevel",
]
