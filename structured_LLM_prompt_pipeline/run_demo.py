"""Executive CLI Demonstration Script for Structured LLM Prompt Pipeline.

Showcases document ingestion, parameterized prompt generation, few-shot conditioning,
schema validation, self-healing reflection, and rich visualization.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Optional

# Ensure UTF-8 stdout encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.tree import Tree
from rich import box

# Ensure base directory in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.pipeline import StructuredPromptPipeline, PipelineResult
from core.config import settings
from providers import get_llm_provider, MockProvider

console = Console(force_terminal=True, legacy_windows=False)


def display_banner():
    """Display rich welcome banner."""
    banner_text = """[bold cyan]+------------------------------------------------------------------------------+[/bold cyan]
[bold cyan]|[/bold cyan]  [bold yellow]STRUCTURED LLM PROMPT PIPELINE & JSON EXTRACTOR[/bold yellow]                             [bold cyan]|[/bold cyan]
[bold cyan]|[/bold cyan]  [dim]Generative AI & LangChain Production Ingestion System[/dim]                        [bold cyan]|[/bold cyan]
[bold cyan]+------------------------------------------------------------------------------+[/bold cyan]"""
    console.print(banner_text)


def display_invoice_summary(data: dict):
    """Render structured financial summary tables."""
    vendor = data.get("vendor", {})
    customer = data.get("customer", {})
    meta = data.get("metadata", {})
    fin = data.get("financials", {})
    items = data.get("line_items", [])

    # Overview Table
    overview = Table(title="[bold yellow]DOCUMENT & STAKEHOLDER OVERVIEW[/bold yellow]", box=box.ROUNDED)
    overview.add_column("Category", style="cyan bold", width=18)
    overview.add_column("Details", style="white")

    overview.add_row("Vendor / Issuer", f"[bold green]{vendor.get('name', 'N/A')}[/bold green]\n{vendor.get('address', 'N/A')}\nEmail: {vendor.get('email', 'N/A')} | Tax ID: {vendor.get('tax_id', 'N/A')}")
    overview.add_row("Billed Customer", f"[bold blue]{customer.get('name', 'N/A')}[/bold blue]\n{customer.get('billing_address', 'N/A')}\nContact: {customer.get('contact_person', 'N/A')}")
    overview.add_row("Metadata", f"Invoice #: [bold]{meta.get('invoice_number', 'N/A')}[/bold] | Date: {meta.get('issue_date', 'N/A')} | Due: {meta.get('due_date', 'N/A')} | PO: {meta.get('po_number', 'N/A')}")
    console.print(overview)

    # Line Items Table
    if items:
        item_table = Table(title="[bold yellow]ITEMIZED BILLING BREAKDOWN[/bold yellow]", box=box.ROUNDED)
        item_table.add_column("SKU / ID", style="dim", width=12)
        item_table.add_column("Description", style="white")
        item_table.add_column("Qty", justify="right", style="cyan")
        item_table.add_column("Unit Price", justify="right", style="cyan")
        item_table.add_column("Tax %", justify="right", style="dim")
        item_table.add_column("Total", justify="right", style="bold green")

        for itm in items:
            item_table.add_row(
                str(itm.get("item_id") or "-"),
                str(itm.get("description", "")),
                f"{itm.get('quantity', 0.0):.2f}",
                f"{itm.get('unit_price', 0.0):.2f}",
                f"{itm.get('tax_rate_pct', 0.0):.1f}%",
                f"{itm.get('line_total', 0.0):.2f}"
            )
        console.print(item_table)

    # Financial Totals Table
    fin_table = Table(title="[bold yellow]FINANCIAL SUMMARY & RECONCILIATION[/bold yellow]", box=box.ROUNDED)
    fin_table.add_column("Financial Metric", style="cyan")
    fin_table.add_column("Amount", justify="right", style="bold")

    curr = fin.get("currency", "USD")
    fin_table.add_row("Subtotal", f"{fin.get('subtotal', 0.0):.2f} {curr}")
    fin_table.add_row("Tax Assessed", f"{fin.get('tax_amount', 0.0):.2f} {curr}")
    fin_table.add_row("Discounts Deducted", f"-{fin.get('discount_amount', 0.0):.2f} {curr}", style="magenta")
    fin_table.add_row("Shipping / Handling", f"{fin.get('shipping_fee', 0.0):.2f} {curr}")
    fin_table.add_row("[bold white]GRAND TOTAL PAYABLE[/bold white]", f"[bold green]{fin.get('grand_total', 0.0):.2f} {curr}[/bold green]")
    fin_table.add_row("Payment Status", f"[bold yellow]{fin.get('payment_status', 'Unpaid')}[/bold yellow]")
    console.print(fin_table)


def display_resume_summary(data: dict):
    """Render structured candidate profile summary."""
    person = data.get("personal_info", {})
    exp_list = data.get("work_experience", [])
    skills = data.get("skills", {})
    edu = data.get("education", [])

    # Personal Profile Panel
    prof_text = f"[bold green]{person.get('full_name', 'N/A')}[/bold green] | [yellow]{data.get('seniority_level', 'Mid-Level')}[/yellow] ({data.get('total_years_experience', 'N/A')} yrs exp)\n"
    prof_text += f"[white]Email:[/white] {person.get('email', 'N/A')} | [white]Phone:[/white] {person.get('phone', 'N/A')} | [white]Location:[/white] {person.get('location', 'N/A')}\n"
    if person.get("linkedin_url"):
        prof_text += f"[dim]LinkedIn: {person.get('linkedin_url')} | GitHub: {person.get('github_url', 'N/A')}[/dim]\n"
    prof_text += f"\n[italic white]\"{data.get('professional_summary', '')}\"[/italic white]"
    console.print(Panel(prof_text, title="[bold yellow]CANDIDATE IDENTITY & SUMMARY[/bold yellow]", box=box.ROUNDED))

    # Experience Table
    if exp_list:
        exp_table = Table(title="[bold yellow]CHRONOLOGICAL WORK EXPERIENCE[/bold yellow]", box=box.ROUNDED)
        exp_table.add_column("Company & Role", style="cyan bold", width=30)
        exp_table.add_column("Dates & Location", style="dim", width=22)
        exp_table.add_column("Key Achievements & Tech", style="white")

        for job in exp_list:
            comp_role = f"[bold]{job.get('position')}[/bold]\n[green]{job.get('company')}[/green]"
            period = f"{job.get('start_date')} → {job.get('end_date')}\n{job.get('location', 'Remote')}"
            bullets = "\n".join([f"• {r}" for r in job.get("responsibilities", [])[:2]])
            tech = "Tech: " + ", ".join(job.get("tech_stack_used", [])[:5])
            exp_table.add_row(comp_role, period, f"{bullets}\n[dim magenta]{tech}[/dim magenta]")
        console.print(exp_table)

    # Skills Tree
    tree = Tree("[bold yellow]SKILLS MATRIX TAXONOMY[/bold yellow]")
    if skills.get("programming_languages"):
        tree.add(f"[cyan bold]Languages:[/cyan bold] {', '.join(skills['programming_languages'])}")
    if skills.get("frameworks_and_libraries"):
        tree.add(f"[green bold]Frameworks & Libraries:[/green bold] {', '.join(skills['frameworks_and_libraries'])}")
    if skills.get("databases"):
        tree.add(f"[yellow bold]Databases & Stores:[/yellow bold] {', '.join(skills['databases'])}")
    if skills.get("cloud_and_devops"):
        tree.add(f"[magenta bold]Cloud & DevOps:[/magenta bold] {', '.join(skills['cloud_and_devops'])}")
    if skills.get("ai_and_data_tools"):
        tree.add(f"[blue bold]AI & Data:[/blue bold] {', '.join(skills['ai_and_data_tools'])}")
    console.print(tree)


def run_cli():
    """Parse arguments and execute extraction pipeline."""
    parser = argparse.ArgumentParser(
        description="Structured LLM Prompt Pipeline & JSON Extractor"
    )
    parser.add_argument(
        "--domain",
        choices=["invoice", "resume"],
        default="invoice",
        help="Target document domain (default: invoice)"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to input document (.txt, .pdf, .docx, .md)"
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "gemini", "openai", "langchain_gemini", "langchain_openai"],
        default=None,
        help="LLM provider backend (default: mock if no keys set)"
    )
    parser.add_argument(
        "--no-few-shot",
        action="store_true",
        help="Disable few-shot exemplar conditioning"
    )
    parser.add_argument(
        "--test-self-heal",
        action="store_true",
        help="Deliberately simulate a schema validation fault on attempt 1 to demonstrate agentic self-healing reflection"
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Save output validated JSON to target file path"
    )

    args = parser.parse_args()

    display_banner()

    # Determine default sample file if none provided
    domain = args.domain.lower()
    file_path = args.file

    if not file_path:
        if domain == "invoice":
            file_path = str(BASE_DIR / "data" / "sample_invoices" / "tech_consulting_invoice.txt")
        else:
            file_path = str(BASE_DIR / "data" / "sample_resumes" / "senior_ai_engineer_resume.txt")

    # Provider configuration
    provider_name = args.provider or settings.default_provider
    if args.test_self_heal:
        provider = MockProvider(simulate_latency_sec=0.2, fail_once_for_demo=True)
    else:
        provider = get_llm_provider(provider_name=provider_name)

    few_shot = not args.no_few_shot

    few_shot_badge = "[bold green]ON[/bold green]" if few_shot else "[bold red]OFF[/bold red]"
    heal_badge = "[bold yellow]ACTIVE[/bold yellow]" if args.test_self_heal else "[dim]INACTIVE[/dim]"

    # Status Panel
    console.print(Panel(
        f"[white]Domain:[/white] [bold cyan]{domain.upper()}[/bold cyan]  |  "
        f"[white]Provider:[/white] [bold green]{provider.get_provider_name()}[/bold green] ({provider.get_model_name()})  |  "
        f"[white]Few-Shot:[/white] {few_shot_badge}  |  "
        f"[white]Self-Heal Test:[/white] {heal_badge}\n"
        f"[white]Input File:[/white] [dim]{file_path}[/dim]",
        title="[bold yellow]PIPELINE EXECUTION PARAMETERS[/bold yellow]",
        box=box.ROUNDED
    ))

    # Initialize Pipeline
    pipeline = StructuredPromptPipeline(provider=provider, max_retries=2)

    with console.status("[bold green]Executing prompt pipeline & JSON extraction...", spinner="dots"):
        result: PipelineResult = pipeline.extract(
            domain=domain,
            source=file_path,
            enable_few_shot=few_shot,
        )

    # Render Results
    if result.success:
        console.print("\n[bold green]✓ EXTRACTION & PYDANTIC VALIDATION SUCCESSFUL![/bold green]")
        console.print(f"[dim]Execution Time: {result.latency_seconds:.2f}s | Attempts: {result.attempts}[/dim]\n")

        if domain == "invoice":
            display_invoice_summary(result.validated_data)
        else:
            display_resume_summary(result.validated_data)

        # Audit Trail Panel
        audit_table = Table(title="[bold yellow]PIPELINE AUDIT TRAIL[/bold yellow]", box=box.ROUNDED)
        audit_table.add_column("Attempt", justify="center", width=8)
        audit_table.add_column("Status", width=12)
        audit_table.add_column("Latency", justify="right", width=10)
        audit_table.add_column("Diagnostics", style="dim")

        for audit in result.audit_trail:
            st = f"[bold green]{audit['status']}[/bold green]" if audit['status'] == "SUCCESS" else f"[bold red]{audit['status']}[/bold red]"
            audit_table.add_row(
                str(audit.get("attempt", 1)),
                st,
                f"{audit.get('latency', 0.0):.2f}s",
                str(audit.get("note") or audit.get("error") or "")
            )
        console.print(audit_table)

        # JSON preview
        console.print("\n[bold yellow]VALIDATED JSON OUTPUT (FIRST 40 LINES):[/bold yellow]")
        json_str = result.to_pretty_json()
        preview_lines = "\n".join(json_str.splitlines()[:40])
        if len(json_str.splitlines()) > 40:
            preview_lines += "\n  ... [truncated, use --export to save full JSON]"
        console.print(Syntax(preview_lines, "json", theme="monokai", line_numbers=True))

        # Export if requested
        if args.export:
            export_path = Path(args.export)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            export_path.write_text(json_str, encoding="utf-8")
            console.print(f"\n[bold green]✓ Validated JSON exported to {export_path}[/bold green]")

    else:
        console.print("\n[bold red]✗ EXTRACTION FAILED[/bold red]")
        console.print(f"[red]Error: {result.error_message}[/red]")
        console.print(f"[dim]Total Latency: {result.latency_seconds:.2f}s | Attempts: {result.attempts}[/dim]")


if __name__ == "__main__":
    run_cli()
