"""Curated few-shot exemplars for structured extraction.

Few-shot learning provides the LLM with concrete input-output pairs,
significantly boosting accuracy and schema adherence on noisy documents.
"""

from typing import Dict, List

INVOICE_FEW_SHOT_INPUT = """
INVOICE #INV-2024-8891
Date: 14 Oct 2024   Due: 28 Oct 2024
Billed By:
Apex Cloud Solutions Inc.
100 Tech Boulevard, Suite 400, Austin, TX 78701
Tax ID / EIN: 82-4910294
Email: billing@apexcloud.io | Ph: +1 (512) 555-0199

Bill To:
Meridian Global Logistics LLC
Attention: Sarah Jenkins (Accounts Payable)
450 Harbor Way, Long Beach, CA 90802
PO Reference: PO-99412

Items / Services:
1. Cloud Infrastructure Managed Hosting (Oct 2024) - Qty: 1.0 @ $1,500.00 = $1,500.00
2. DevOps Engineering Consulting (15.0 hrs @ $120.00/hr) = $1,800.00
3. SSL Wildcard Certificate Renewal (Item ID: SEC-SSL-01) - Qty: 2 @ $75.00 = $150.00

Subtotal: $3,450.00
Tax (8.25%): $284.63
Enterprise Discount: -$150.00
Grand Total: $3,584.63 USD

Payment Details:
Wire Transfer to Silicon Valley Bank
Account #: 9876543210
Routing (ABA): 121000358
Terms: Net 14 days
"""

INVOICE_FEW_SHOT_OUTPUT = """{
  "vendor": {
    "name": "Apex Cloud Solutions Inc.",
    "address": "100 Tech Boulevard, Suite 400, Austin, TX 78701",
    "tax_id": "82-4910294",
    "email": "billing@apexcloud.io",
    "phone": "+1 (512) 555-0199",
    "website": "https://apexcloud.io"
  },
  "customer": {
    "name": "Meridian Global Logistics LLC",
    "billing_address": "450 Harbor Way, Long Beach, CA 90802",
    "shipping_address": null,
    "email": null,
    "phone": null,
    "contact_person": "Sarah Jenkins"
  },
  "metadata": {
    "invoice_number": "INV-2024-8891",
    "issue_date": "2024-10-14",
    "due_date": "2024-10-28",
    "po_number": "PO-99412",
    "currency": "USD",
    "payment_terms": "Net 14 days"
  },
  "line_items": [
    {
      "item_id": null,
      "description": "Cloud Infrastructure Managed Hosting (Oct 2024)",
      "quantity": 1.0,
      "unit_price": 1500.0,
      "tax_rate_pct": 8.25,
      "line_total": 1500.0
    },
    {
      "item_id": null,
      "description": "DevOps Engineering Consulting",
      "quantity": 15.0,
      "unit_price": 120.0,
      "tax_rate_pct": 8.25,
      "line_total": 1800.0
    },
    {
      "item_id": "SEC-SSL-01",
      "description": "SSL Wildcard Certificate Renewal",
      "quantity": 2.0,
      "unit_price": 75.0,
      "tax_rate_pct": 8.25,
      "line_total": 150.0
    }
  ],
  "financials": {
    "subtotal": 3450.0,
    "tax_amount": 284.63,
    "discount_amount": 150.0,
    "shipping_fee": 0.0,
    "grand_total": 3584.63,
    "currency": "USD",
    "payment_status": "Unpaid"
  },
  "payment_info": {
    "bank_name": "Silicon Valley Bank",
    "account_number": "9876543210",
    "routing_or_swift": "121000358",
    "ifsc_code": null,
    "payment_method": "Wire Transfer"
  },
  "notes": "Payment due within 14 days of issue.",
  "extraction_confidence": 0.98,
  "math_discrepancy_detected": false
}"""

RESUME_FEW_SHOT_INPUT = """
ELENA ROSTOVA
San Francisco, CA | (415) 890-1234 | elena.rostova@gmail.com
LinkedIn: linkedin.com/in/erostova | GitHub: github.com/erostova-ai

SUMMARY
Senior AI/ML Engineer with 6+ years of experience designing and deploying scalable LLM applications, RAG pipelines, and agentic workflows in production cloud environments.

WORK EXPERIENCE
Lead Generative AI Engineer | Synthetix AI, San Francisco, CA
March 2022 - Present
- Architected agentic multi-model RAG system serving 2.5M daily queries, reducing query latency by 42%.
- Fine-tuned open-source LLMs (Llama-3, Mistral) using LoRA/QLoRA on distributed GPU clusters with vLLM.
- Spearheaded team of 5 engineers in deploying LangChain & LangGraph production microservices on AWS EKS.

Machine Learning Engineer | DataVision Corp, Seattle, WA
July 2018 - February 2022
- Built computer vision & NLP ingestion pipelines processing 500k+ documents monthly with PyTorch and FastAPI.
- Implemented automated schema validation and vector database indexing with Qdrant and Redis.

EDUCATION
University of California, Berkeley
Master of Science in Computer Science (Specialization: Artificial Intelligence) | 2016 - 2018 | GPA: 3.92
University of Washington
Bachelor of Science in Computer Engineering | 2012 - 2016 | Magna Cum Laude

SKILLS
Languages: Python, C++, TypeScript, SQL
Frameworks: PyTorch, LangChain, FastAPI, Transformers, vLLM
Databases & Vector DBs: PostgreSQL, Pinecone, Qdrant, Redis
Cloud & Tools: AWS, Docker, Kubernetes, CI/CD, Git

CERTIFICATIONS
- AWS Certified Machine Learning - Specialty (2023)
"""

RESUME_FEW_SHOT_OUTPUT = """{
  "personal_info": {
    "full_name": "Elena Rostova",
    "email": "elena.rostova@gmail.com",
    "phone": "+1 415-890-1234",
    "location": "San Francisco, CA, USA",
    "linkedin_url": "https://linkedin.com/in/erostova",
    "github_url": "https://github.com/erostova-ai",
    "portfolio_url": null
  },
  "professional_summary": "Senior AI/ML Engineer with 6+ years of experience designing and deploying scalable LLM applications, RAG pipelines, and agentic workflows in production cloud environments.",
  "seniority_level": "Lead",
  "total_years_experience": 6.5,
  "work_experience": [
    {
      "company": "Synthetix AI",
      "position": "Lead Generative AI Engineer",
      "start_date": "2022-03",
      "end_date": "Present",
      "is_current": true,
      "location": "San Francisco, CA",
      "responsibilities": [
        "Architected agentic multi-model RAG system serving 2.5M daily queries, reducing latency by 42%",
        "Fine-tuned open-source LLMs using LoRA/QLoRA on distributed GPU clusters with vLLM",
        "Spearheaded team of 5 engineers in deploying LangChain & LangGraph production microservices on AWS EKS"
      ],
      "tech_stack_used": [
        "Python",
        "LangChain",
        "LangGraph",
        "vLLM",
        "AWS EKS",
        "PyTorch",
        "LoRA"
      ]
    },
    {
      "company": "DataVision Corp",
      "position": "Machine Learning Engineer",
      "start_date": "2018-07",
      "end_date": "2022-02",
      "is_current": false,
      "location": "Seattle, WA",
      "responsibilities": [
        "Built computer vision & NLP ingestion pipelines processing 500k+ documents monthly with PyTorch and FastAPI",
        "Implemented automated schema validation and vector database indexing with Qdrant and Redis"
      ],
      "tech_stack_used": [
        "PyTorch",
        "FastAPI",
        "Qdrant",
        "Redis",
        "Python"
      ]
    }
  ],
  "education": [
    {
      "institution": "University of California, Berkeley",
      "degree": "Master of Science",
      "field_of_study": "Computer Science (Artificial Intelligence)",
      "start_year": 2016,
      "graduation_year": 2018,
      "gpa_or_grade": "3.92 / 4.0",
      "honors": []
    },
    {
      "institution": "University of Washington",
      "degree": "Bachelor of Science",
      "field_of_study": "Computer Engineering",
      "start_year": 2012,
      "graduation_year": 2016,
      "gpa_or_grade": null,
      "honors": ["Magna Cum Laude"]
    }
  ],
  "skills": {
    "programming_languages": ["Python", "C++", "TypeScript", "SQL"],
    "frameworks_and_libraries": ["PyTorch", "LangChain", "FastAPI", "Transformers", "vLLM", "LangGraph"],
    "databases": ["PostgreSQL", "Pinecone", "Qdrant", "Redis"],
    "cloud_and_devops": ["AWS", "Docker", "Kubernetes", "CI/CD", "Git"],
    "ai_and_data_tools": ["Hugging Face", "LoRA", "RAG"],
    "soft_skills": ["Team Leadership", "Architecture Design", "Cross-functional Collaboration"]
  },
  "certifications": [
    {
      "name": "AWS Certified Machine Learning - Specialty",
      "issuing_organization": "Amazon Web Services",
      "issue_year": "2023",
      "credential_id_or_url": null
    }
  ],
  "projects": [],
  "languages": [
    {
      "language": "English",
      "proficiency": "Fluent"
    }
  ],
  "extraction_confidence": 0.99
}"""

FEW_SHOT_REGISTRY: Dict[str, Dict[str, str]] = {
    "invoice": {
        "input": INVOICE_FEW_SHOT_INPUT.strip(),
        "output": INVOICE_FEW_SHOT_OUTPUT.strip(),
    },
    "resume": {
        "input": RESUME_FEW_SHOT_INPUT.strip(),
        "output": RESUME_FEW_SHOT_OUTPUT.strip(),
    },
}


def get_few_shot_exemplar(domain: str) -> Dict[str, str]:
    """Retrieve few-shot input/output for a given domain."""
    normalized = domain.lower().strip()
    return FEW_SHOT_REGISTRY.get(normalized, {})
