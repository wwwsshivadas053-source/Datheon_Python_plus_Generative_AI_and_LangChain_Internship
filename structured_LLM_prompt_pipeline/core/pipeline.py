"""Structured Prompt Pipeline & Extraction Orchestrator.

Orchestrates document ingestion, parameterized prompt generation, few-shot conditioning,
LLM inference, multi-stage JSON healing, Pydantic schema validation, and agentic self-correction.
"""

import time
import json
from typing import Optional, Union, Dict, Any, List
from pathlib import Path
from pydantic import BaseModel, ValidationError

from schemas import get_schema_class
from prompts import PromptBuilder
from providers import BaseLLMProvider, get_llm_provider
from core.doc_loader import DocumentLoader
from core.json_cleaner import JSONCleaner
from core.config import settings


class PipelineResult(BaseModel):
    """Encapsulates the final outcome, validated data, and execution audit log."""
    success: bool
    domain: str
    validated_data: Optional[Dict[str, Any]] = None
    raw_json_dict: Optional[Dict[str, Any]] = None
    raw_llm_output: str = ""
    attempts: int = 1
    latency_seconds: float = 0.0
    audit_trail: List[Dict[str, Any]] = []
    error_message: Optional[str] = None
    provider_used: str = ""
    model_used: str = ""

    def to_pretty_json(self) -> str:
        """Return pretty-printed validated JSON string."""
        if self.validated_data:
            return json.dumps(self.validated_data, indent=2)
        elif self.raw_json_dict:
            return json.dumps(self.raw_json_dict, indent=2)
        return "{}"


class StructuredPromptPipeline:
    """Production-grade LLM Prompt Pipeline and JSON Extractor."""

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        provider_name: Optional[str] = None,
        max_retries: Optional[int] = None,
    ):
        self.provider = provider or get_llm_provider(provider_name=provider_name)
        self.max_retries = (
            max_retries
            if max_retries is not None
            else settings.max_self_correction_retries
        )

    def extract(
        self,
        domain: str,
        source: Union[str, Path],
        custom_instructions: Optional[str] = None,
        temperature: Optional[float] = None,
        enable_few_shot: Optional[bool] = None,
    ) -> PipelineResult:
        """Run end-to-end extraction pipeline with schema validation and self-correction."""
        start_time = time.time()
        temp = temperature if temperature is not None else settings.default_temperature
        few_shot = enable_few_shot if enable_few_shot is not None else settings.enable_few_shot

        # Step 1: Ingest document
        try:
            document_text = DocumentLoader.load(source)
            if not document_text:
                raise ValueError("Loaded document is empty.")
        except Exception as err:
            return PipelineResult(
                success=False,
                domain=domain,
                error_message=f"Document loading failed: {str(err)}",
                latency_seconds=round(time.time() - start_time, 3),
                provider_used=self.provider.get_provider_name(),
                model_used=self.provider.get_model_name(),
            )

        # Step 2: Initialize PromptBuilder and Target Schema
        prompt_builder = PromptBuilder(domain=domain, enable_few_shot=few_shot)
        schema_cls = get_schema_class(domain)

        system_prompt = prompt_builder.build_system_prompt()
        user_prompt = prompt_builder.build_user_prompt(
            unstructured_text=document_text,
            custom_instructions=custom_instructions,
        )

        audit_trail: List[Dict[str, Any]] = []
        current_user_prompt = user_prompt
        last_raw_output = ""
        last_error = ""

        # Step 3: Inference and Validation Loop with Self-Correction Retries
        for attempt in range(1, self.max_retries + 2):
            attempt_start = time.time()
            try:
                # LLM Call
                raw_output = self.provider.generate(
                    system_prompt=system_prompt,
                    user_prompt=current_user_prompt,
                    temperature=temp,
                    json_mode=True,
                )
                last_raw_output = raw_output

                # JSON Extraction & Healing
                parsed_dict, cleaned_str = JSONCleaner.extract_and_parse(raw_output)

                # Strict Pydantic Schema Validation
                validated_model = schema_cls.model_validate(parsed_dict)

                # Validation succeeded!
                elapsed = time.time() - start_time
                audit_trail.append({
                    "attempt": attempt,
                    "status": "SUCCESS",
                    "latency": round(time.time() - attempt_start, 3),
                    "note": "Output passed JSON parsing and strict Pydantic schema validation."
                })

                return PipelineResult(
                    success=True,
                    domain=domain,
                    validated_data=validated_model.model_dump(),
                    raw_json_dict=parsed_dict,
                    raw_llm_output=raw_output,
                    attempts=attempt,
                    latency_seconds=round(elapsed, 3),
                    audit_trail=audit_trail,
                    provider_used=self.provider.get_provider_name(),
                    model_used=self.provider.get_model_name(),
                )

            except (ValidationError, ValueError) as err:
                error_desc = str(err)
                last_error = error_desc
                audit_trail.append({
                    "attempt": attempt,
                    "status": "FAILED",
                    "latency": round(time.time() - attempt_start, 3),
                    "error": error_desc,
                    "note": f"Triggering self-correction retry ({attempt}/{self.max_retries + 1})."
                })

                if attempt <= self.max_retries:
                    # Construct Reflexion / Self-Correction Prompt
                    current_user_prompt = prompt_builder.build_self_correction_prompt(
                        unstructured_text=document_text,
                        previous_output=last_raw_output,
                        validation_errors=error_desc,
                    )
                else:
                    break

            except Exception as err:
                # Provider or network fatal error
                elapsed = time.time() - start_time
                audit_trail.append({
                    "attempt": attempt,
                    "status": "FATAL_ERROR",
                    "latency": round(time.time() - attempt_start, 3),
                    "error": str(err),
                })
                return PipelineResult(
                    success=False,
                    domain=domain,
                    raw_llm_output=last_raw_output,
                    error_message=f"Pipeline error during inference: {str(err)}",
                    attempts=attempt,
                    latency_seconds=round(elapsed, 3),
                    audit_trail=audit_trail,
                    provider_used=self.provider.get_provider_name(),
                    model_used=self.provider.get_model_name(),
                )

        # Retries exhausted
        elapsed = time.time() - start_time
        return PipelineResult(
            success=False,
            domain=domain,
            raw_llm_output=last_raw_output,
            error_message=f"Validation failed after {self.max_retries + 1} attempts. Last error: {last_error}",
            attempts=self.max_retries + 1,
            latency_seconds=round(elapsed, 3),
            audit_trail=audit_trail,
            provider_used=self.provider.get_provider_name(),
            model_used=self.provider.get_model_name(),
        )
