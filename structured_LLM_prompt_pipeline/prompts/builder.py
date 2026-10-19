"""PromptBuilder: Parameterized prompt constructor and schema synthesizer.

Binds Pydantic schemas, domain guidelines, few-shot exemplars, and custom
user guidance into production-grade prompts for direct LLM calls and LangChain.
"""

import json
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel
from schemas import get_schema_class
from .templates import (
    SYSTEM_PROMPT_TEMPLATE,
    EXTRACTION_USER_PROMPT_TEMPLATE,
    SELF_CORRECTION_PROMPT_TEMPLATE,
)
from .few_shots import get_few_shot_exemplar


class PromptBuilder:
    """Constructs parameterized extraction prompts tailored to schemas and domains."""

    def __init__(self, domain: str, enable_few_shot: bool = True):
        self.domain = domain.lower().strip()
        self.schema_cls: Type[BaseModel] = get_schema_class(self.domain)
        self.enable_few_shot = enable_few_shot

    def get_json_schema_str(self) -> str:
        """Generate formatted JSON schema string from Pydantic model."""
        schema_dict = self.schema_cls.model_json_schema()
        return json.dumps(schema_dict, indent=2)

    def build_system_prompt(self) -> str:
        """Return standardized system prompt with extraction principles."""
        return SYSTEM_PROMPT_TEMPLATE.strip()

    def build_few_shot_section(self) -> str:
        """Construct the few-shot demonstration section if enabled."""
        if not self.enable_few_shot:
            return ""

        exemplar = get_few_shot_exemplar(self.domain)
        if not exemplar or "input" not in exemplar:
            return ""

        return (
            "======================================================================\n"
            "FEW-SHOT EXTRACTION DEMONSTRATION (LEARNING EXAMPLE):\n"
            "======================================================================\n"
            "INPUT UNSTRUCTURED TEXT:\n"
            f"```text\n{exemplar['input']}\n```\n\n"
            "CORRESPONDING VALID JSON OUTPUT:\n"
            f"```json\n{exemplar['output']}\n```\n"
        )

    def build_user_prompt(
        self,
        unstructured_text: str,
        custom_instructions: Optional[str] = None
    ) -> str:
        """Build the full parameterized user prompt for initial extraction."""
        schema_str = self.get_json_schema_str()
        few_shot_str = self.build_few_shot_section()

        custom_sec = ""
        if custom_instructions and custom_instructions.strip():
            custom_sec = (
                "======================================================================\n"
                "USER-SPECIFIED CUSTOM INSTRUCTIONS:\n"
                "======================================================================\n"
                f"{custom_instructions.strip()}\n"
            )

        return EXTRACTION_USER_PROMPT_TEMPLATE.format(
            domain_name=self.domain.capitalize(),
            target_schema_json=schema_str,
            few_shot_section=few_shot_str,
            unstructured_text=unstructured_text.strip(),
            custom_instructions_section=custom_sec,
        )

    def build_self_correction_prompt(
        self,
        unstructured_text: str,
        previous_output: str,
        validation_errors: str
    ) -> str:
        """Build a targeted self-correction / reflexion prompt to heal invalid outputs."""
        schema_str = self.get_json_schema_str()
        return SELF_CORRECTION_PROMPT_TEMPLATE.format(
            target_schema_json=schema_str,
            validation_errors=validation_errors.strip(),
            previous_output=previous_output.strip(),
            unstructured_text=unstructured_text.strip(),
        )

    def to_langchain_messages(
        self,
        unstructured_text: str,
        custom_instructions: Optional[str] = None
    ) -> list:
        """Format as message tuples compatible with LangChain ChatPromptTemplate."""
        system_text = self.build_system_prompt()
        user_text = self.build_user_prompt(unstructured_text, custom_instructions)
        return [
            ("system", system_text),
            ("user", user_text),
        ]
