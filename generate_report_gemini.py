#!/usr/bin/env python3
"""
Generate MRI Radiology Report using Template-Driven Approach

This script uses a TEMPLATE-DRIVEN method where:
1. A rigid, human-written template defines the report structure
2. Rule-based sentence generators fill each placeholder
3. All slot values are deterministically derived from structured facts

The template is 100% controlled - no external model can modify structure.

Usage:
    python generate_report_gemini.py <case_folder>
    python generate_report_gemini.py results/BraTS-GLI-00009-000
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Import template-based report generation
from report_templates import (
    ReportTemplateFiller,
    generate_report_from_summary,
    generate_report_simple,
    MRI_BRAIN_TEMPLATE,
    SLOT_SPECIFICATIONS,
    SlotValidator,
    FactExtractor,
    FactsToSlotMapper,
)

# Clinically significant thresholds (now handled in report_templates.py)
MIDLINE_SHIFT_THRESHOLD_MM = 2.0


def load_summary(case_folder: Path) -> dict:
    """Load the LLM-ready summary JSON file."""
    summary_path = case_folder / "feature_extraction" / "llm_ready_summary.json"
    
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    
    with open(summary_path, 'r') as f:
        return json.load(f)


def generate_template_report(summary: dict) -> tuple:
    """
    Generate the radiology report using the 4-STEP TEMPLATE-DRIVEN approach.
    
    Pipeline:
        Step 1: Rigid template (MRI_BRAIN_TEMPLATE) - human-written
        Step 2: Slot specifications with constraints
        Step 3: FactExtractor - model outputs → structured facts
        Step 4: FactsToSlotMapper - facts → slot values (deterministic)
    
    Returns:
        Tuple of (report_string, validation_log, extracted_facts)
    """
    return generate_report_from_summary(summary, validate=True)


def save_report(report: str, case_folder: Path, case_id: str, method: str = "template"):
    """Save the generated report to files."""
    output_folder = case_folder / "feature_extraction"
    
    # Save as text file
    report_path = output_folder / "radiology_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved to: {report_path}")
    
    # Save as JSON with metadata
    report_json = {
        "case_id": case_id,
        "generated_at": datetime.now().isoformat(),
        "generation_method": method,
        "template_version": "1.0",
        "report": report
    }
    json_path = output_folder / "radiology_report.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_json, f, indent=2)
    print(f"JSON saved to: {json_path}")
    
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate MRI radiology report using template-driven approach"
    )
    parser.add_argument(
        "case_folder",
        type=str,
        help="Path to the case results folder (e.g., results/BraTS-GLI-00009-000)"
    )
    
    args = parser.parse_args()
    
    # Resolve case folder path
    case_folder = Path(args.case_folder)
    if not case_folder.is_absolute():
        case_folder = Path(__file__).parent / case_folder
    
    if not case_folder.exists():
        print(f"Error: Case folder not found: {case_folder}")
        sys.exit(1)
    
    case_id = case_folder.name
    
    print("=" * 70)
    print("TEMPLATE-DRIVEN RADIOLOGY REPORT GENERATOR")
    print("=" * 70)
    print(f"\nCase ID: {case_id}")
    print(f"Case folder: {case_folder}")
    print(f"Method: Template-driven (deterministic)")
    
    # Load summary
    print("\nLoading analysis summary...")
    try:
        summary = load_summary(case_folder)
        print(f"Loaded summary for: {summary.get('case_id', 'Unknown')}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you have run the full pipeline first:")
        print("  python run_full_pipeline.py <case_folder>")
        sys.exit(1)
    
    # Generate report using 4-step template pipeline
    print("\n" + "-" * 50)
    print("4-STEP TEMPLATE PIPELINE")
    print("-" * 50)
    print("Step 1: Rigid Template (human-written)")
    print("Step 2: Slot Specifications (constraints)")
    print("Step 3: Fact Extraction (deterministic)")
    print("Step 4: Facts -> Slot Values (deterministic)")
    print("-" * 50)
    
    print(f"\nSlot specifications: {len(SLOT_SPECIFICATIONS)} slots defined")
    report, validation_log, facts = generate_template_report(summary)
    method = "template (4-step pipeline)"
    
    # Show extracted facts summary
    print(f"\nStep 3 - Extracted facts:")
    print(f"  Lesion count: {facts.get('lesion_count', 'N/A')}")
    print(f"  Hemisphere: {facts.get('hemisphere', 'N/A')}")
    print(f"  Size: {facts.get('size_cm', 'N/A')} cm")
    print(f"  Edema degree: {facts.get('edema_degree', 'N/A')}")
    print(f"  Ring-enhancing: {facts.get('is_ring_enhancing', 'N/A')}")
    
    # Report validation results
    if validation_log:
        print(f"\n[!] Validation found {len(validation_log)} issues (auto-corrected):")
        for entry in validation_log:
            print(f"  - {entry['slot']}: {entry['violations']}")
    else:
        print("\n[OK] All slots passed validation")
    
    # Save report
    print("\nSaving report...")
    report_path = save_report(report, case_folder, case_id, method)
    
    print("\n" + "=" * 70)
    print("REPORT GENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nMethod: {method}")
    print(f"Slots validated: {len(SLOT_SPECIFICATIONS)}")
    print(f"Output: {report_path}")
    
    # Print preview
    print("\n" + "-" * 70)
    print("REPORT PREVIEW:")
    print("-" * 70)
    preview_lines = report.split('\n')[:35]
    print('\n'.join(preview_lines))
    if len(report.split('\n')) > 35:
        print("\n... [truncated for preview] ...")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
