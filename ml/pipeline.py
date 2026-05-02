"""
NeuroMatch — Main Pipeline
Orchestrates all layers: diagnosis → trial fetching → validation → gap analysis.
Single entry point for the full NeuroMatch workflow.
"""

import json
from diagnosis import generate_diagnosis, load_symptom_dataset, format_diagnosis_output
from parser import fetch_trials_for_diagnoses, parse_trials_batch
from validator import validate_trials_batch, validate_patient_profile
from gap_explainer import analyze_all_trials


def run_neuromatch(patient_profile: dict, audience: str = "patient") -> dict:
    """
    Full NeuroMatch pipeline.

    Args:
        patient_profile: Patient data following patient_schema.json
        audience: "patient" | "neurologist" | "gp" | "researcher"

    Returns:
        Final result dict with diagnosis, matched trials, and gap analysis
    """

    print("\n" + "="*60)
    print("🧠 NeuroMatch Pipeline Starting")
    print("="*60)

    # ── STEP 0: Validate patient input ──
    print("\n[Step 0] Validating patient profile...")
    is_valid, issues = validate_patient_profile(patient_profile)
    if not is_valid:
        return {
            "error": "Invalid patient profile",
            "issues": issues
        }
    print(f"  ✅ Patient profile valid")

    # ── STEP 1: Generate diagnosis ──
    print("\n[Step 1] Generating neurological diagnosis...")
    disease_profiles = load_symptom_dataset()
    diagnosis_result = generate_diagnosis(patient_profile, disease_profiles)

    if "error" in diagnosis_result:
        return {"error": f"Diagnosis failed: {diagnosis_result['error']}"}

    diagnoses = patient_profile.get("predicted_diagnosis", [])
    print(f"  ✅ Top diagnosis: {diagnoses[0]['disease'] if diagnoses else 'Unknown'} "
          f"({int(diagnoses[0]['confidence'] * 100)}% confidence)" if diagnoses else "  ✅ Diagnosis complete")

    # ── STEP 2: Fetch clinical trials ──
    print("\n[Step 2] Fetching clinical trials from ClinicalTrials.gov...")
    trials = fetch_trials_for_diagnoses(diagnoses, max_per_condition=5)

    if not trials:
        return {
            "diagnosis": diagnosis_result,
            "diagnosis_formatted": format_diagnosis_output(diagnosis_result, audience),
            "trials": [],
            "gap_analysis": [],
            "message": "No recruiting trials found for your condition at this time."
        }

    print(f"  ✅ Fetched {len(trials)} trials")

    # ── STEP 3: Parse eligibility criteria ──
    print("\n[Step 3] Parsing eligibility criteria with LLM...")
    enriched_trials = parse_trials_batch(trials)
    print(f"  ✅ Parsed eligibility for {len(enriched_trials)} trials")

    # ── STEP 4: Validate parsed criteria ──
    print("\n[Step 4] Validating parsed criteria...")
    validated_trials = validate_trials_batch(enriched_trials)
    print(f"  ✅ Validation complete")

    # ── STEP 5: Gap analysis ──
    print("\n[Step 5] Analyzing eligibility gaps...")
    gap_results = analyze_all_trials(patient_profile, validated_trials)

    eligible = [g for g in gap_results if g["overall_status"] == "likely_eligible"]
    needs_info = [g for g in gap_results if g["overall_status"] == "needs_more_info"]
    ineligible = [g for g in gap_results if g["overall_status"] == "likely_ineligible"]

    print(f"  ✅ {len(eligible)} likely eligible, {len(needs_info)} need more info, {len(ineligible)} likely ineligible")

    # ── STEP 6: Format diagnosis output ──
    diagnosis_formatted = format_diagnosis_output(diagnosis_result, audience)

    print("\n" + "="*60)
    print("✅ NeuroMatch Pipeline Complete")
    print("="*60)

    return {
        "patient_id": patient_profile.get("patient_id", "unknown"),
        "audience": audience,
        "diagnosis": diagnosis_result,
        "diagnosis_formatted": diagnosis_formatted,
        "trials_analyzed": len(gap_results),
        "gap_analysis": gap_results,
        "summary": {
            "likely_eligible": len(eligible),
            "needs_more_info": len(needs_info),
            "likely_ineligible": len(ineligible),
            "top_match": gap_results[0] if gap_results else None
        }
    }


def format_pipeline_output(result: dict) -> str:
    """
    Formats the full pipeline result for display.
    """
    if "error" in result:
        return f"❌ Error: {result['error']}\n" + "\n".join(result.get("issues", []))

    output = []
    output.append("=" * 60)
    output.append("🧠 NeuroMatch Results")
    output.append("=" * 60)

    # Diagnosis section
    output.append("\n📋 DIAGNOSIS\n")
    output.append(result.get("diagnosis_formatted", ""))

    # Trial matches
    output.append("\n\n🔬 CLINICAL TRIAL MATCHES\n")
    output.append(f"Analyzed {result['trials_analyzed']} recruiting trials\n")

    summary = result.get("summary", {})
    output.append(f"  ✅ Likely eligible: {summary.get('likely_eligible', 0)}")
    output.append(f"  ❓ Needs more info: {summary.get('needs_more_info', 0)}")
    output.append(f"  ❌ Likely ineligible: {summary.get('likely_ineligible', 0)}")

    # Top matches
    gap_analysis = result.get("gap_analysis", [])
    top_trials = [g for g in gap_analysis if g["overall_status"] != "likely_ineligible"][:3]

    if top_trials:
        output.append("\n\n🏆 TOP TRIAL MATCHES\n")
        for i, gap in enumerate(top_trials, 1):
            llm = gap.get("llm_analysis", {})
            status_icon = "✅" if gap["overall_status"] == "likely_eligible" else "❓"
            output.append(f"{i}. {status_icon} {gap['trial_title'][:70]}")
            output.append(f"   NCT ID: {gap['trial_id']} | Confidence: {int(gap['confidence'] * 100)}%")
            output.append(f"   {llm.get('summary_patient', '')}")

            gaps = llm.get("gaps", [])
            if gaps:
                output.append(f"   ⚠️  Gaps: {', '.join([g['criterion'] for g in gaps[:2]])}")
            output.append("")

    return "\n".join(output)


# ── TEST ──

if __name__ == "__main__":
    test_patient = {
        "patient_id": "pipeline_test_001",
        "symptoms": ["memory loss", "confusion", "behavioral changes", "difficulty with daily tasks"],
        "duration": "6 months",
        "age": 70,
        "gender": "F",
        "existing_conditions": [],
        "medications": [],
        "report_text": None,
        "predicted_diagnosis": [],
        "matched_trials": []
    }

    # Run full pipeline for patient audience
    result = run_neuromatch(test_patient, audience="patient")

    # Print formatted output
    print("\n\n")
    print(format_pipeline_output(result))

    # Save result to JSON for inspection
    with open("pipeline_test_output.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("\n📄 Full result saved to pipeline_test_output.json")