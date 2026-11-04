"""
Configuration module for the Interactive Full-Stack AI Chatbot.
Handles model catalogs, system personas, default parameters, and environment settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os


@dataclass
class ModelOption:
    name: str
    id: str
    context_window: int = 8192
    supports_vision: bool = False
    description: str = ""


PROVIDER_MODELS: Dict[str, List[ModelOption]] = {
    "Demo Mode (No API Key Required)": [
        ModelOption(
            name="Demo Intelligent Assistant (Instant Stream)",
            id="demo-fast",
            context_window=8192,
            description="Built-in simulated streaming engine with RAG support, zero API key required."
        ),
        ModelOption(
            name="Demo Research Specialist (Simulated Deep Thinker)",
            id="demo-reasoner",
            context_window=16384,
            description="Simulates chain-of-thought analysis and comprehensive document breakdown."
        )
    ],
    "Google Gemini": [
        ModelOption(
            name="Gemini 2.5 Flash (Ultra-fast, High Intelligence)",
            id="gemini-2.5-flash",
            context_window=1048576,
            supports_vision=True,
            description="Recommended default for fast streaming and massive context windows."
        ),
        ModelOption(
            name="Gemini 2.5 Pro (Complex Reasoning)",
            id="gemini-2.5-pro",
            context_window=2097152,
            supports_vision=True,
            description="Deep multimodal reasoning and coding."
        ),
        ModelOption(
            name="Gemini 1.5 Flash (Legacy)",
            id="gemini-1.5-flash",
            context_window=1000000,
            supports_vision=True
        )
    ],
    "OpenAI": [
        ModelOption(
            name="GPT-4o (Omni Flagship)",
            id="gpt-4o",
            context_window=128000,
            supports_vision=True,
            description="High-intelligence multimodal flagship model."
        ),
        ModelOption(
            name="GPT-4o-mini (Fast & Efficient)",
            id="gpt-4o-mini",
            context_window=128000,
            supports_vision=True,
            description="Affordable, fast, ideal for everyday chat & lightweight RAG."
        ),
        ModelOption(
            name="o3-mini (Reasoning Model)",
            id="o3-mini",
            context_window=200000,
            description="Specialized in STEM, math, and structured code synthesis."
        )
    ],
    "Groq (Ultra-Fast Inference)": [
        ModelOption(
            name="Llama 3.3 70B Versatile",
            id="llama-3.3-70b-versatile",
            context_window=128000,
            description="High quality open-source model running at blazing speeds."
        ),
        ModelOption(
            name="Mixtral 8x7B 32k",
            id="mixtral-8x7b-32768",
            context_window=32768,
            description="MoE architecture for high throughput."
        )
    ],
    "Anthropic": [
        ModelOption(
            name="Claude 3.5 Sonnet",
            id="claude-3-5-sonnet-latest",
            context_window=200000,
            supports_vision=True,
            description="State-of-the-art coding and nuanced analysis."
        ),
        ModelOption(
            name="Claude 3.5 Haiku",
            id="claude-3-5-haiku-latest",
            context_window=200000,
            description="Ultra-fast conversational AI."
        )
    ]
}

PERSONAS: Dict[str, str] = {
    "Helpful AI Assistant": (
        "You are an intelligent, empathetic, and concise full-stack AI conversational assistant. "
        "Answer questions clearly, structure explanations with bullet points or markdown when appropriate, "
        "and maintain a collaborative, friendly tone."
    ),
    "Senior Software Architect": (
        "You are a seasoned Principal Software Architect and Lead Developer. "
        "Provide production-grade code, architectural best practices, clean design patterns, "
        "security recommendations, and robust error handling. Always format code blocks cleanly."
    ),
    "RAG & Data Research Analyst": (
        "You are a meticulous research analyst and information synthesis specialist. "
        "When context or documents are provided, ground your answers directly in the retrieved excerpts, "
        "cite relevant passages or sections, and clearly disclose when an answer is not found in the source text."
    ),
    "Executive Summarizer": (
        "You are an executive briefing specialist. Deliver concise, high-impact bulleted summaries, "
        "key takeaways, action items, and bottom-line implications without fluff."
    ),
    "Creative & Technical Writer": (
        "You are a master technical writer and creative communicator. Transform complex or dry concepts "
        "into engaging, elegant, and illuminating prose with intuitive analogies."
    )
}

DEFAULT_CONFIG = {
    "app_title": "Full-Stack AI Nexus",
    "app_subtitle": "Enterprise Conversational AI & RAG Engine",
    "app_version": "v1.2.0",
    "default_provider": "Demo Mode (No API Key Required)",
    "default_temperature": 0.7,
    "default_max_tokens": 2048,
    "default_top_k": 4,
    "chunk_size": 750,
    "chunk_overlap": 150,
    "max_history_turns": 10
}

