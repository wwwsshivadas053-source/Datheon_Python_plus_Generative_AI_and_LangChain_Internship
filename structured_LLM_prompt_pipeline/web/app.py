"""Flask Web Application & REST API for Structured LLM Prompt Pipeline.

Provides interactive dashboard for document extraction, prompt inspection,
model evaluation, schema exploration, and self-correction visual tracking.
"""

import os
import sys
import tempfile
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# Ensure base directory in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.pipeline import StructuredPromptPipeline
from core.config import settings
from core.doc_loader import DocumentLoader
from schemas import get_schema_class, SCHEMA_REGISTRY
from prompts import PromptBuilder
from providers import get_llm_provider, MockProvider

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "web" / "templates"),
    static_folder=str(BASE_DIR / "web" / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload


@app.route("/")
def index():
    """Render master interactive dashboard."""
    return render_template(
        "index.html",
        default_provider=settings.default_provider,
        has_gemini_key=settings.has_gemini_key(),
        has_openai_key=settings.has_openai_key(),
    )


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Retrieve preloaded sample documents for quick testing."""
    sample_dir = BASE_DIR / "data"
    samples = {
        "invoice": [],
        "resume": []
    }

    inv_dir = sample_dir / "sample_invoices"
    if inv_dir.exists():
        for file in sorted(inv_dir.glob("*.txt")):
            samples["invoice"].append({
                "id": file.stem,
                "title": file.stem.replace("_", " ").title(),
                "content": file.read_text(encoding="utf-8")
            })

    res_dir = sample_dir / "sample_resumes"
    if res_dir.exists():
        for file in sorted(res_dir.glob("*.txt")):
            samples["resume"].append({
                "id": file.stem,
                "title": file.stem.replace("_", " ").title(),
                "content": file.read_text(encoding="utf-8")
            })

    return jsonify(samples)


@app.route("/api/schema/<domain>", methods=["GET"])
def get_schema(domain: str):
    """Retrieve the JSON schema for a target domain."""
    try:
        schema_cls = get_schema_class(domain)
        return jsonify({
            "domain": domain,
            "schema": schema_cls.model_json_schema()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/prompt_preview", methods=["POST"])
def prompt_preview():
    """Generate and return the formatted prompt without executing LLM call."""
    data = request.json or {}
    domain = data.get("domain", "invoice")
    text = data.get("text", "")
    enable_few_shot = data.get("enable_few_shot", True)
    custom_instructions = data.get("custom_instructions", "")

    try:
        builder = PromptBuilder(domain=domain, enable_few_shot=enable_few_shot)
        system_prompt = builder.build_system_prompt()
        user_prompt = builder.build_user_prompt(
            unstructured_text=text or "[Unstructured document text will appear here...]",
            custom_instructions=custom_instructions
        )
        return jsonify({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "few_shot_included": enable_few_shot
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Parse text from uploaded .txt, .pdf, or .docx file."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    uploaded = request.files["file"]
    if not uploaded.filename:
        return jsonify({"error": "No filename specified."}), 400

    suffix = Path(uploaded.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        uploaded.save(tmp.name)
        tmp_path = tmp.name

    try:
        extracted_text = DocumentLoader.load_from_file(tmp_path)
        os.remove(tmp_path)
        return jsonify({
            "filename": uploaded.filename,
            "text": extracted_text,
            "characters": len(extracted_text)
        })
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return jsonify({"error": f"Failed to parse document: {str(e)}"}), 500


@app.route("/api/extract", methods=["POST"])
def extract():
    """Execute pipeline extraction."""
    data = request.json or {}
    domain = data.get("domain", "invoice")
    text = data.get("text", "")
    provider_name = data.get("provider", "mock")
    api_key = data.get("api_key") or None
    temperature = float(data.get("temperature", 0.1))
    enable_few_shot = bool(data.get("enable_few_shot", True))
    max_retries = int(data.get("max_retries", 2))
    custom_instructions = data.get("custom_instructions") or None
    simulate_fail_once = bool(data.get("simulate_fail_once", False))

    if not text.strip():
        return jsonify({"error": "Document text cannot be empty."}), 400

    try:
        if provider_name == "mock" and simulate_fail_once:
            provider = MockProvider(simulate_latency_sec=0.25, fail_once_for_demo=True)
        else:
            provider = get_llm_provider(
                provider_name=provider_name,
                api_key=api_key,
            )

        pipeline = StructuredPromptPipeline(provider=provider, max_retries=max_retries)
        result = pipeline.extract(
            domain=domain,
            source=text,
            custom_instructions=custom_instructions,
            temperature=temperature,
            enable_few_shot=enable_few_shot,
        )

        return jsonify({
            "success": result.success,
            "domain": result.domain,
            "validated_data": result.validated_data,
            "raw_json_dict": result.raw_json_dict,
            "raw_llm_output": result.raw_llm_output,
            "attempts": result.attempts,
            "latency_seconds": result.latency_seconds,
            "audit_trail": result.audit_trail,
            "error_message": result.error_message,
            "provider_used": result.provider_used,
            "model_used": result.model_used,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def create_app():
    return app


if __name__ == "__main__":
    app.run(host=settings.host, port=settings.port, debug=settings.debug)
