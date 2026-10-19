"""Prompt templates for structured data extraction and self-correction."""

SYSTEM_PROMPT_TEMPLATE = """You are an elite, highly accurate document extraction and data structuring AI.
Your sole mission is to extract structured information from unstructured, messy text documents (such as invoices, receipts, resumes, or CVs) and output strictly valid, schema-compliant JSON.

CRITICAL INSTRUCTIONS & CONSTRAINTS:
1. TRUTHFULNESS & GROUNDING: Extract ONLY information that is explicitly stated or directly inferable from the source text. NEVER invent, hallucinate, or assume facts not present.
2. MISSING VALUES: If an optional field is not present in the input text, set its value to null (or an empty list [] for array fields). Never fill placeholders like 'N/A' or 'Unknown' unless instructed.
3. NUMERIC ACCURACY: Extract exact numbers, quantities, prices, and totals. Convert written numbers or formatted currency (e.g. '$1,299.50') to raw float values (e.g. 1299.50).
4. DATE NORMALIZATION: Convert dates to standard ISO-8601 strings (YYYY-MM-DD) whenever possible. If only year or month is available, retain 'YYYY' or 'YYYY-MM'.
5. PURE JSON OUTPUT: Your entire response must be a single, valid JSON object that strictly adheres to the provided JSON Schema.
6. NO EXTRA TEXT: Do NOT preface your response with conversational filler (e.g., 'Here is the JSON:'). Output ONLY the JSON block.
"""

EXTRACTION_USER_PROMPT_TEMPLATE = """TASK: Extract structured {domain_name} data from the unstructured text provided below into the exact JSON schema specified.

======================================================================
TARGET JSON SCHEMA:
======================================================================
{target_schema_json}

{few_shot_section}

======================================================================
UNSTRUCTURED INPUT DOCUMENT:
======================================================================
```text
{unstructured_text}
```

{custom_instructions_section}

Please provide your response as a strictly valid JSON object matching the schema above:
"""

SELF_CORRECTION_PROMPT_TEMPLATE = """Your previous extraction generated a JSON object that FAILED strict Pydantic validation rules.

======================================================================
TARGET JSON SCHEMA:
======================================================================
{target_schema_json}

======================================================================
VALIDATION ERRORS ENCOUNTERED:
======================================================================
{validation_errors}

======================================================================
YOUR PREVIOUS INVALID JSON OUTPUT:
======================================================================
```json
{previous_output}
```

======================================================================
SOURCE TEXT FOR REFERENCE:
======================================================================
```text
{unstructured_text}
```

TASK:
Analyze each validation error above, correct the fields accordingly, and output the corrected, valid JSON object.
Ensure all field types, required fields, and mathematical or format constraints are fully satisfied. Output ONLY the corrected JSON object.
"""
