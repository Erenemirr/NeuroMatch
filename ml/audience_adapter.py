"""
NeuroMatch — Layer 6: Audience Adapter
Formats the full pipeline output for 4 different audiences:
patient, neurologist, gp, researcher
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ── SECTION 1: Rule-Based Formatters (no LLM, instant) ──

def format_for_patient(result: dict) -> str:
    """Plain English, empathetic, no medical jargon."""
    lines = []
    lines.append("=" * 55)
    lines.append("🧠 Your NeuroMatch Results")
    lines.append("=" * 55)

    # Diagnosis
    diagnoses = result.get("diagnosis", {}).get("preliminary_diagnoses", [])
    if diagnoses:
        lines.append("\n📋 Based on your symptoms, possible conditions are:\n")
        for d in diagnoses[:3]:
            pct = int(d["confidence"] * 100)
            bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
            lines.append(f"  {d['disease']}")
            lines.append(f"  [{bar}] {pct}% match\n")

    # Recommended tests
    tests = result.get("diagnosis", {}).get("recommended_tests", [])
    if tests:
        lines.append(f"🔬 Recommended tests: {', '.join(tests)}\n")

    # Urgency
    urgency = result.get("diagnosis", {}).get("urgency_level", "")
    urgency_map = {
        "high": "⚠️  Please seek medical attention soon.",
        "medium": "📅 Schedule an appointment with your doctor.",
        "low": "💬 Mention these symptoms at your next doctor visit."
    }
    if urgency in urgency_map:
        lines.append(urgency_map[urgency] + "\n")

    # Trial matches
    gap_analysis = result.get("gap_analysis", [])
    eligible = [g for g in gap_analysis if g["overall_status"] == "likely_eligible"]
    maybe = [g for g in gap_analysis if g["overall_status"] == "needs_more_info"]

    lines.append("─" * 55)
    lines.append(f"🔍 We searched {len(gap_analysis)} active clinical trials for you.\n")

    if eligible:
        lines.append(f"✅ Good news! You may qualify for {len(eligible)} trial(s):\n")
        for g in eligible[:2]:
            llm = g.get("llm_analysis", {})
            lines.append(f"  • {g['trial_title'][:65]}")
            lines.append(f"    {llm.get('summary_patient', '')}\n")

    if maybe:
        lines.append(f"❓ {len(maybe)} trial(s) need more information from you.\n")
        lines.append("  Ask your doctor about getting tested for:")
        all_gaps = []
        for g in maybe[:3]:
            for gap in g.get("llm_analysis", {}).get("gaps", [])[:2]:
                action = gap.get("action", "")
                if action and action not in all_gaps:
                    all_gaps.append(action)
        for action in all_gaps[:4]:
            lines.append(f"  → {action}")

    lines.append("\n" + "─" * 55)
    lines.append("⚠️  This is not a medical diagnosis.")
    lines.append("    Please consult a neurologist before making any decisions.")

    return "\n".join(lines)


def format_for_neurologist(result: dict) -> str:
    """Clinical detail, confidence scores, criteria breakdown."""
    lines = []
    lines.append("=" * 60)
    lines.append("NEUROMATCH — Clinical Assessment Report")
    lines.append("=" * 60)

    # Diagnosis
    diagnoses = result.get("diagnosis", {}).get("preliminary_diagnoses", [])
    lines.append("\n[PRELIMINARY DIAGNOSIS]\n")
    for d in diagnoses:
        lines.append(f"  [{d['confidence']:.2f}] {d['disease']}")
        lines.append(f"         Matching: {', '.join(d.get('matching_symptoms', []))}")
        lines.append(f"         Missing:  {', '.join(d.get('missing_indicators', []))}\n")

    summary = result.get("diagnosis", {}).get("summary", "")
    if summary:
        lines.append(f"Clinical Summary: {summary}\n")

    urgency = result.get("diagnosis", {}).get("urgency_level", "").upper()
    tests = result.get("diagnosis", {}).get("recommended_tests", [])
    lines.append(f"Urgency: {urgency}")
    lines.append(f"Recommended workup: {', '.join(tests)}\n")

    # Trial matches
    gap_analysis = result.get("gap_analysis", [])
    lines.append("─" * 60)
    lines.append(f"[CLINICAL TRIAL MATCHING] — {len(gap_analysis)} trials evaluated\n")

    for g in gap_analysis[:5]:
        llm = g.get("llm_analysis", {})
        status_map = {"likely_eligible": "ELIGIBLE", "needs_more_info": "PENDING", "likely_ineligible": "INELIGIBLE"}
        status = status_map.get(g["overall_status"], "UNKNOWN")
        lines.append(f"  [{status}] {g['trial_id']} — {g['trial_title'][:60]}")
        lines.append(f"  Confidence: {int(g['confidence'] * 100)}%")

        # Rule checks
        for check in g.get("rule_checks", []):
            icon = "✓" if check["status"] == "pass" else "✗" if check["status"] == "fail" else "?"
            lines.append(f"    {icon} {check['field']}: {check['note']}")

        # Gaps
        gaps = llm.get("gaps", [])
        if gaps:
            lines.append(f"  Unmet criteria:")
            for gap in gaps[:3]:
                lines.append(f"    - [{gap.get('type')}] {gap.get('criterion')}")

        lines.append(f"  {llm.get('summary_clinician', '')}\n")

    return "\n".join(lines)


def format_for_gp(result: dict) -> str:
    """Referral summary — brief, actionable, no deep technical detail."""
    lines = []
    lines.append("=" * 55)
    lines.append("NeuroMatch — GP Referral Summary")
    lines.append("=" * 55)

    diagnoses = result.get("diagnosis", {}).get("preliminary_diagnoses", [])
    top = diagnoses[0] if diagnoses else {}
    urgency = result.get("diagnosis", {}).get("urgency_level", "unknown")
    tests = result.get("diagnosis", {}).get("recommended_tests", [])

    lines.append(f"\nPrimary concern: {top.get('disease', 'Unknown')} "
                 f"({int(top.get('confidence', 0) * 100)}% AI confidence)")
    lines.append(f"Urgency level:  {urgency.upper()}")
    lines.append(f"Suggested workup: {', '.join(tests[:3])}\n")

    gap_analysis = result.get("gap_analysis", [])
    eligible = [g for g in gap_analysis if g["overall_status"] == "likely_eligible"]
    pending = [g for g in gap_analysis if g["overall_status"] == "needs_more_info"]

    lines.append(f"Clinical trial screening: {len(gap_analysis)} trials reviewed")
    lines.append(f"  → Potentially eligible: {len(eligible)}")
    lines.append(f"  → Awaiting further data: {len(pending)}\n")

    if eligible:
        lines.append("Recommended referral trials:")
        for g in eligible[:2]:
            lines.append(f"  • {g['trial_id']}: {g['trial_title'][:60]}")

    if pending:
        lines.append("\nPre-referral assessments to consider:")
        collected = []
        for g in pending[:3]:
            for gap in g.get("llm_analysis", {}).get("gaps", [])[:2]:
                action = gap.get("action", "")
                if action and action not in collected:
                    collected.append(action)
        for item in collected[:4]:
            lines.append(f"  → {item}")

    lines.append("\nGenerated by NeuroMatch AI — for clinical decision support only.")

    return "\n".join(lines)


def format_for_researcher(result: dict) -> str:
    """Full structured JSON dump — maximum detail."""
    return json.dumps(result, indent=2, default=str)


# ── SECTION 2: LLM-Enhanced Adapter (optional, richer output) ──

ADAPTER_PROMPT = """
You are NeuroMatch, an AI clinical trial matching system.
You have analyzed a patient's symptoms and matched them to clinical trials.

Pipeline Results Summary:
{summary}

Your task: Write a {audience_desc} for this patient's results.

Guidelines:
{guidelines}

Keep it concise, accurate, and appropriate for the audience.
Return plain text only, no JSON, no markdown headers.
"""

AUDIENCE_CONFIGS = {
    "patient": {
        "desc": "compassionate, plain-English summary",
        "guidelines": (
            "- Use simple language, no medical jargon\n"
            "- Be empathetic and encouraging\n"
            "- Explain what clinical trials are briefly\n"
            "- Focus on what the patient can DO next\n"
            "- Keep under 150 words"
        )
    },
    "neurologist": {
        "desc": "detailed clinical assessment note",
        "guidelines": (
            "- Use clinical terminology\n"
            "- Include confidence scores and differential diagnosis\n"
            "- Mention specific eligibility criteria gaps\n"
            "- Note recommended workup\n"
            "- Keep under 200 words"
        )
    },
    "gp": {
        "desc": "brief GP referral note",
        "guidelines": (
            "- Focus on top diagnosis and urgency\n"
            "- List 2-3 pre-referral actions\n"
            "- Mention best trial matches by NCT ID\n"
            "- Keep under 100 words"
        )
    },
    "researcher": {
        "desc": "structured research summary",
        "guidelines": (
            "- Include all diagnoses with confidence scores\n"
            "- List all trial IDs evaluated\n"
            "- Note eligibility gap types and frequencies\n"
            "- Mention data completeness issues\n"
            "- Keep under 250 words"
        )
    }
}


def adapt_with_llm(result: dict, audience: str) -> str:
    """
    Uses LLM to generate a polished audience-specific summary.
    Falls back to rule-based if LLM fails.
    """
    config = AUDIENCE_CONFIGS.get(audience, AUDIENCE_CONFIGS["patient"])

    # Build a compact summary for the prompt
    diagnoses = result.get("diagnosis", {}).get("preliminary_diagnoses", [])
    gap_analysis = result.get("gap_analysis", [])

    summary = {
        "top_diagnoses": [
            {"disease": d["disease"], "confidence": d["confidence"]}
            for d in diagnoses[:3]
        ],
        "urgency": result.get("diagnosis", {}).get("urgency_level", "unknown"),
        "trials_analyzed": len(gap_analysis),
        "likely_eligible": len([g for g in gap_analysis if g["overall_status"] == "likely_eligible"]),
        "needs_more_info": len([g for g in gap_analysis if g["overall_status"] == "needs_more_info"]),
        "top_trial": gap_analysis[0].get("trial_title", "") if gap_analysis else "",
        "top_gaps": [
            gap.get("criterion", "")
            for g in gap_analysis[:2]
            for gap in g.get("llm_analysis", {}).get("gaps", [])[:2]
        ]
    }

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are NeuroMatch, a clinical trial matching AI. Be concise and accurate."
                },
                {
                    "role": "user",
                    "content": ADAPTER_PROMPT.format(
                        summary=json.dumps(summary, indent=2),
                        audience_desc=config["desc"],
                        guidelines=config["guidelines"]
                    )
                }
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[ADAPTER] LLM failed for {audience}, using rule-based: {e}")
        return adapt_rule_based(result, audience)


def adapt_rule_based(result: dict, audience: str) -> str:
    """Rule-based fallback — no API call needed."""
    formatters = {
        "patient": format_for_patient,
        "neurologist": format_for_neurologist,
        "gp": format_for_gp,
        "researcher": format_for_researcher,
    }
    formatter = formatters.get(audience, format_for_patient)
    return formatter(result)


def adapt_output(result: dict, audience: str, use_llm: bool = True) -> str:
    """
    Main entry point for audience adaptation.

    Args:
        result: Full pipeline output from run_neuromatch()
        audience: "patient" | "neurologist" | "gp" | "researcher"
        use_llm: If True, uses LLM for richer output; False uses rule-based only

    Returns:
        Formatted string for the given audience
    """
    if use_llm:
        return adapt_with_llm(result, audience)
    else:
        return adapt_rule_based(result, audience)


# ── TEST ──

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from pipeline import run_neuromatch

    test_patient = {
        "patient_id": "adapter_test_001",
        "symptoms": ["memory loss", "confusion", "behavioral changes"],
        "duration": "6 months",
        "age": 70,
        "gender": "F",
        "existing_conditions": [],
        "medications": [],
        "report_text": None,
        "predicted_diagnosis": [],
        "matched_trials": []
    }

    print("Running pipeline...")
    result = run_neuromatch(test_patient, audience="patient")

    for audience in ["patient", "neurologist", "gp"]:
        print(f"\n{'='*60}")
        print(f"AUDIENCE: {audience.upper()} (rule-based)")
        print("="*60)
        print(adapt_output(result, audience, use_llm=False))

    print(f"\n{'='*60}")
    print("AUDIENCE: PATIENT (LLM-enhanced)")
    print("="*60)
    print(adapt_output(result, "patient", use_llm=True))