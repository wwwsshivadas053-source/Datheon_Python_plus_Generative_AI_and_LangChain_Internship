"""Realistic Offline Mock LLM Provider.

Enables instant testing, hackathon demonstrations, CI/CD evaluation, and offline
reviews without requiring active API keys or consuming cloud credits.
"""

import time
import json
import re
from typing import Optional
from .base import BaseLLMProvider


class MockProvider(BaseLLMProvider):
    """High-fidelity simulated LLM provider with realistic latency and domain heuristics."""

    def __init__(self, simulate_latency_sec: float = 0.3, fail_once_for_demo: bool = False):
        self.simulate_latency_sec = simulate_latency_sec
        self.fail_once_for_demo = fail_once_for_demo
        self._has_failed = False

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        json_mode: bool = True
    ) -> str:
        """Simulate LLM response with domain-aware dynamic extraction."""
        if self.simulate_latency_sec > 0:
            time.sleep(self.simulate_latency_sec)

        # Check if this is a self-correction prompt
        if "FAILED strict Pydantic validation rules" in user_prompt:
            # Check target schema in correction prompt
            if "resume" in user_prompt.lower():
                return self._mock_resume_extraction(user_prompt)
            return self._mock_invoice_extraction(user_prompt)

        # Demo trigger: if fail_once_for_demo is True, deliberately return malformed field to trigger self-correction
        if self.fail_once_for_demo and not self._has_failed:
            self._has_failed = True
            return self._generate_deliberately_invalid_response(user_prompt)

        # Exact domain detection via prompt header or schema name
        lower_prompt = user_prompt.lower()
        if "extract structured resume data" in lower_prompt or "resumeextraction" in lower_prompt:
            return self._mock_resume_extraction(user_prompt)
        elif "extract structured invoice data" in lower_prompt or "invoiceextraction" in lower_prompt:
            return self._mock_invoice_extraction(user_prompt)
        elif "seniority_level" in lower_prompt or "work_experience" in lower_prompt:
            return self._mock_resume_extraction(user_prompt)
        else:
            return self._mock_invoice_extraction(user_prompt)

    def _extract_doc_segment(self, prompt: str) -> str:
        """Extract the specific input document section from the prompt envelope."""
        if "UNSTRUCTURED INPUT DOCUMENT:" in prompt:
            parts = prompt.split("UNSTRUCTURED INPUT DOCUMENT:")
            return parts[-1]
        return prompt

    def _mock_invoice_extraction(self, prompt: str) -> str:
        """Dynamic heuristic extraction for invoice documents."""
        doc = self._extract_doc_segment(prompt)

        # Detect currency
        currency = "USD"
        if "eur" in doc.lower() or "€" in doc:
            currency = "EUR"
        elif "inr" in doc.lower() or "₹" in doc:
            currency = "INR"
        elif "gbp" in doc.lower() or "£" in doc:
            currency = "GBP"

        # Look for invoice number
        inv_match = re.search(r"Invoice Number:\s*([A-Za-z0-9\-]+)", doc, re.IGNORECASE)
        if not inv_match:
            inv_match = re.search(r"(?:invoice\s*(?:#|number|ref)?|inv-?)\s*[:#]\s*([A-Za-z0-9\-]+)", doc, re.IGNORECASE)
        if not inv_match:
            inv_match = re.search(r"(?:INV-[0-9]{4}-[0-9]+|ETL-[0-9A-Z\-]+|TXN-[0-9A-Z\-]+)", doc)
        inv_no = inv_match.group(1) if (inv_match and inv_match.groups()) else (inv_match.group(0) if inv_match else "INV-2024-9042")

        # Look for date
        date_match = re.search(r"(\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})", doc)
        issue_date = "2024-10-18"
        if date_match:
            issue_date = date_match.group(1)

        # Look for vendor name
        vendor_match = re.search(r"from\s+([A-Za-z0-9\s]+?)(?:\.|\s+Billed|\n)", doc, re.IGNORECASE)
        if vendor_match:
            vendor_name = vendor_match.group(1).strip()
        else:
            clean_lines = [
                l.strip(" =*-\t\r\n") for l in doc.splitlines()
                if l.strip(" =*-\t\r\n")
                and not l.strip().startswith("```")
                and "task:" not in l.lower()
                and "target json" not in l.lower()
                and "tax invoice" not in l.lower()
                and "invoice #" not in l.lower()
                and len(l.strip(" =*-\t\r\n")) > 3
            ]
            vendor_name = clean_lines[0] if clean_lines else "Cyberdynamic Enterprises LLC"

        # Mock payload
        payload = {
            "vendor": {
                "name": vendor_name,
                "address": "452 Innovation Parkway, Austin, TX 78759",
                "tax_id": "US-84920194",
                "email": "billing@omnitech.io",
                "phone": "+1 (512) 800-4491",
                "website": "https://omnitech.io"
            },
            "customer": {
                "name": "Global Horizon Logistics Corp.",
                "billing_address": "880 Pacific Highway, Suite 1200, San Diego, CA 92101",
                "shipping_address": "880 Pacific Highway, Suite 1200, San Diego, CA 92101",
                "email": "invoices@globalhorizon.com",
                "phone": "+1 (619) 555-0144",
                "contact_person": "David Martinez"
            },
            "metadata": {
                "invoice_number": inv_no,
                "issue_date": "2024-10-18",
                "due_date": "2024-11-17",
                "po_number": "PO-88219",
                "currency": currency,
                "payment_terms": "Net 30 Days"
            },
            "line_items": [
                {
                    "item_id": "SVC-CLOUD-01",
                    "description": "Multi-Tenant Cloud Infrastructure Provisioning & Setup",
                    "quantity": 1.0,
                    "unit_price": 2400.0,
                    "tax_rate_pct": 8.0,
                    "line_total": 2400.0
                },
                {
                    "item_id": "CON-AI-OPS",
                    "description": "Generative AI Prompt Pipeline & Evaluation Architecture Consulting",
                    "quantity": 20.0,
                    "unit_price": 150.0,
                    "tax_rate_pct": 8.0,
                    "line_total": 3000.0
                },
                {
                    "item_id": "LIC-SEC-ENTERPRISE",
                    "description": "Enterprise Security Gateway Annual License (Tier 2)",
                    "quantity": 2.0,
                    "unit_price": 450.0,
                    "tax_rate_pct": 8.0,
                    "line_total": 900.0
                }
            ],
            "financials": {
                "subtotal": 6300.0,
                "tax_amount": 504.0,
                "discount_amount": 300.0,
                "shipping_fee": 0.0,
                "grand_total": 6504.0,
                "currency": currency,
                "payment_status": "Unpaid"
            },
            "payment_info": {
                "bank_name": "JPMorgan Chase Bank, N.A.",
                "account_number": "78201938210",
                "routing_or_swift": "CHASUS33XXX",
                "ifsc_code": None,
                "payment_method": "ACH / Wire Transfer"
            },
            "notes": "Thank you for your business. Please remit payment within 30 days.",
            "extraction_confidence": 0.98,
            "math_discrepancy_detected": False
        }
        return json.dumps(payload, indent=2)

    def _mock_resume_extraction(self, prompt: str) -> str:
        """Dynamic heuristic extraction for resume documents."""
        doc = self._extract_doc_segment(prompt)

        # Candidate name is typically on the very first header line
        lines = [
            l.strip() for l in doc.splitlines()
            if l.strip()
            and not l.strip().startswith("---")
            and not l.strip().startswith("```")
            and not l.strip().startswith("=")
            and "task:" not in l.lower()
            and "target json" not in l.lower()
        ]
        name = "Marcus Vance"
        if lines:
            first_line = lines[0].replace("DR. ", "").replace("Dr. ", "").strip()
            # If first line has separators (|, -, ,), extract candidate name segment
            if "|" in first_line:
                first_line = first_line.split("|")[0].strip()
            elif " - " in first_line:
                first_line = first_line.split(" - ")[0].strip()
            elif "," in first_line and len(first_line.split(",")[0].split()) in (2, 3):
                first_line = first_line.split(",")[0].strip()

            words = first_line.split()
            if len(words) in (2, 3, 4):
                name = first_line.title()

        payload = {
            "personal_info": {
                "full_name": name,
                "email": "marcus.vance.ai@gmail.com",
                "phone": "+1 (415) 789-2341",
                "location": "San Francisco, CA, USA",
                "linkedin_url": "https://linkedin.com/in/marcus-vance-ai",
                "github_url": "https://github.com/marcusvance",
                "portfolio_url": "https://marcusvance.dev"
            },
            "professional_summary": "Lead AI Solutions Architect and Generative AI Engineer with 7+ years of experience engineering production LLM systems, agentic workflow automation, and distributed ML pipelines.",
            "seniority_level": "Lead",
            "total_years_experience": 7.5,
            "work_experience": [
                {
                    "company": "NeuralScale Technologies",
                    "position": "Principal Generative AI Architect",
                    "start_date": "2022-01",
                    "end_date": "Present",
                    "is_current": True,
                    "location": "San Francisco, CA (Hybrid)",
                    "responsibilities": [
                        "Architected enterprise prompt engineering and structured extraction pipeline processing 10M+ documents/month with 99.8% Pydantic validation rate",
                        "Designed autonomous agent multi-step reasoning frameworks with LangGraph, reducing operational workflow turnaround by 65%",
                        "Led team of 8 ML/AI engineers deploying fine-tuned LLMs on Kubernetes with TensorRT-LLM and vLLM"
                    ],
                    "tech_stack_used": ["Python", "LangChain", "LangGraph", "PyTorch", "Kubernetes", "Docker", "vLLM", "AWS"]
                },
                {
                    "company": "Cognitive Cloud Labs",
                    "position": "Senior Machine Learning Engineer",
                    "start_date": "2018-06",
                    "end_date": "2021-12",
                    "is_current": False,
                    "location": "Seattle, WA",
                    "responsibilities": [
                        "Engineered end-to-end NLP document classification and NER pipelines using HuggingFace Transformers and spaCy",
                        "Built low-latency real-time inference microservices with FastAPI and Redis caching serving 40k RPM"
                    ],
                    "tech_stack_used": ["Python", "FastAPI", "Transformers", "PostgreSQL", "Redis", "Docker"]
                }
            ],
            "education": [
                {
                    "institution": "Stanford University",
                    "degree": "Master of Science",
                    "field_of_study": "Computer Science (Artificial Intelligence Specialization)",
                    "start_year": 2016,
                    "graduation_year": 2018,
                    "gpa_or_grade": "3.94 / 4.0",
                    "honors": ["Graduate Fellowship Award"]
                },
                {
                    "institution": "University of California, Berkeley",
                    "degree": "Bachelor of Science",
                    "field_of_study": "Electrical Engineering & Computer Sciences (EECS)",
                    "start_year": 2012,
                    "graduation_year": 2016,
                    "gpa_or_grade": "3.88 / 4.0",
                    "honors": ["Dean's Honors List"]
                }
            ],
            "skills": {
                "programming_languages": ["Python", "TypeScript", "C++", "SQL", "Rust"],
                "frameworks_and_libraries": ["LangChain", "LangGraph", "PyTorch", "FastAPI", "Transformers", "Pydantic"],
                "databases": ["PostgreSQL", "Pinecone", "Qdrant", "Redis", "MongoDB"],
                "cloud_and_devops": ["AWS (ECS/EKS/SageMaker)", "Docker", "Kubernetes", "Terraform", "GitHub Actions"],
                "ai_and_data_tools": ["vLLM", "TensorRT-LLM", "Weights & Biases", "MLflow", "RAG Triad"],
                "soft_skills": ["Technical Leadership", "System Architecture", "Mentorship", "Stakeholder Communication"]
            },
            "certifications": [
                {
                    "name": "AWS Certified Solutions Architect - Professional",
                    "issuing_organization": "Amazon Web Services",
                    "issue_year": "2023",
                    "credential_id_or_url": "AWS-PSA-99128"
                },
                {
                    "name": "Deep Learning Specialization",
                    "issuing_organization": "DeepLearning.AI",
                    "issue_year": "2021",
                    "credential_id_or_url": None
                }
            ],
            "projects": [
                {
                    "title": "AgenticDocExtractor",
                    "description": "Open-source autonomous document understanding framework with automated self-healing validation.",
                    "role": "Creator & Lead Maintainer",
                    "technologies": ["Python", "LangChain", "Pydantic", "FastAPI"],
                    "link": "https://github.com/marcusvance/agentic-doc-extractor"
                }
            ],
            "languages": [
                {
                    "language": "English",
                    "proficiency": "Native"
                },
                {
                    "language": "German",
                    "proficiency": "Conversational"
                }
            ],
            "extraction_confidence": 0.99
        }
        return json.dumps(payload, indent=2)

    def _generate_deliberately_invalid_response(self, prompt: str) -> str:
        """Returns JSON with an intentional type mismatch (e.g., string in place of float)."""
        return """{
  "vendor": { "name": "Broken Demo Corp" },
  "customer": { "name": "Acme Test" },
  "metadata": { "invoice_number": "INV-ERR-001", "issue_date": "2024-10-18", "currency": "USD" },
  "line_items": [
    {
      "description": "Testing Broken Item",
      "quantity": "INVALID_NON_NUMERIC_QTY",
      "unit_price": 100.0,
      "line_total": 100.0
    }
  ],
  "financials": {
    "subtotal": 100.0,
    "grand_total": 100.0,
    "currency": "USD"
  }
}"""

    def _generate_healed_response(self, prompt: str) -> str:
        """Healed response following simulated self-correction."""
        return self._mock_invoice_extraction(prompt)

    def get_provider_name(self) -> str:
        return "Offline Mock Provider"

    def get_model_name(self) -> str:
        return "mock-llm-v1"
