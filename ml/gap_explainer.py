"""
NeuroMatch — Layer 5: Gap Explainer
Explains why a patient does or does not qualify for a clinical trial.
Produces human-readable gap analysis for each audience type.
"""

import os
import json
import asyncio
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()
client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

# ── Persistent Gap Cache ──
_GAP_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".gap_cache.json")

def _load_gap_cache() -> dict:
    try:
        if os.path.exists(_GAP_CACHE_PATH):
            with open(_GAP_CACHE_PATH, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_gap_cache(cache: dict):
    try:
        with open(_GAP_CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass

GAP_CACHE = _load_gap_cache()


# ── SECTION 1: Rule-Based Eligibility Checker ──

def check_age_eligibility(patient: dict, trial: dict) -> dict:
    """Checks if patient age falls within trial age range."""
    patient_age = patient.get("age")
    min_age_str = trial.get("min_age", "N/A")
    max_age_str = trial.get("max_age", "N/A")

    result = {"field": "age", "patient_value": patient_age, "status": "unknown", "note": ""}

    try:
        if patient_age is None:
            result["status"] = "missing"
            result["note"] = "Patient age not provided"
            return result

        min_age = int(min_age_str.replace(" Years", "").replace(" Year", "").strip()) if min_age_str != "N/A" else 0
        max_age = int(max_age_str.replace(" Years", "").replace(" Year", "").strip()) if max_age_str != "N/A" else 120

        if min_age <= int(patient_age) <= max_age:
            result["status"] = "pass"
            result["note"] = f"Age {patient_age} is within {min_age}-{max_age}"
        else:
            result["status"] = "fail"
            result["note"] = f"Age {patient_age} is outside required range {min_age}-{max_age}"

    except (ValueError, AttributeError):
        result["status"] = "unknown"
        result["note"] = f"Could not parse age range: {min_age_str} - {max_age_str}"

    return result


def check_gender_eligibility(patient: dict, trial: dict) -> dict:
    """Checks if patient gender matches trial requirements."""
    patient_gender = patient.get("gender", "").upper()
    trial_gender = trial.get("gender", "ALL").upper()

    result = {"field": "gender", "patient_value": patient_gender, "status": "unknown", "note": ""}

    if trial_gender in ["ALL", "BOTH", ""]:
        result["status"] = "pass"
        result["note"] = "Trial accepts all genders"
    elif patient_gender in ["M", "MALE"] and trial_gender in ["M", "MALE"]:
        result["status"] = "pass"
        result["note"] = "Gender matches"
    elif patient_gender in ["F", "FEMALE"] and trial_gender in ["F", "FEMALE"]:
        result["status"] = "pass"
        result["note"] = "Gender matches"
    else:
        result["status"] = "fail"
        result["note"] = f"Patient gender '{patient_gender}' does not match trial requirement '{trial_gender}'"

    return result


def check_diagnosis_match(patient: dict, trial: dict) -> dict:
    """Checks if patient's predicted diagnosis matches trial's target condition."""
    patient_diagnoses = [d.get("disease", "").lower() for d in patient.get("predicted_diagnosis", [])]
    matched_condition = trial.get("matched_condition", "").lower()

    result = {"field": "diagnosis", "patient_value": patient_diagnoses, "status": "unknown", "note": ""}

    if not matched_condition:
        result["status"] = "unknown"
        result["note"] = "Trial condition not specified"
        return result

    for diagnosis in patient_diagnoses:
        # Partial match — e.g. "alzheimer" in "alzheimer's disease"
        if matched_condition in diagnosis or diagnosis in matched_condition:
            result["status"] = "pass"
            result["note"] = f"Diagnosis '{diagnosis}' matches trial condition '{matched_condition}'"
            return result

    result["status"] = "fail"
    result["note"] = f"Patient diagnoses {patient_diagnoses} do not match trial condition '{matched_condition}'"
    return result


def run_rule_based_checks(patient: dict, trial: dict) -> list:
    """Runs all rule-based eligibility checks."""
    checks = []
    checks.append(check_age_eligibility(patient, trial))
    checks.append(check_gender_eligibility(patient, trial))
    checks.append(check_diagnosis_match(patient, trial))
    return checks


# ── SECTION 2: LLM Gap Analysis ──

GAP_EXPLAINER_PROMPT = """
You are a clinical trial eligibility expert helping explain why a patient may or may not qualify for a trial.

Patient Profile:
- Age: {age}
- Gender: {gender}
- Symptoms: {symptoms}
- Diagnoses: {diagnoses}
- Medications: {medications}
- Existing Conditions: {conditions}

Trial: {trial_title}
Matched Condition: {matched_condition}

Trial Inclusion Criteria:
{inclusion_criteria}

Trial Exclusion Criteria:
{exclusion_criteria}

Rule-Based Check Results:
{rule_checks}

Based on the above, return a JSON with this structure:
{{
    "overall_status": "likely_eligible" | "likely_ineligible" | "needs_more_info",
    "confidence": 0.0 to 1.0,
    "gaps": [
        {{
            "criterion": "the criterion that is not met or unclear",
            "type": "missing_info" | "does_not_meet" | "borderline",
            "explanation": "plain language explanation for the patient",
            "action": "what the patient can do about this"
        }}
    ],
    "strengths": ["criteria the patient clearly meets"],
    "summary_patient": "2-3 sentence plain English summary for the patient",
    "summary_clinician": "2-3 sentence clinical summary for the neurologist"
}}

Only return valid JSON. Be honest but compassionate in explanations.
"""


async def explain_gaps_with_llm(patient: dict, trial: dict, rule_checks: list) -> dict:
    """
    Uses LLM to generate a detailed gap analysis between patient profile and trial criteria.
    """
    validated = trial.get("validated_eligibility", trial.get("parsed_eligibility", {}))
    inclusion = validated.get("inclusion_criteria", [])
    exclusion = validated.get("exclusion_criteria", [])

    # Format criteria for prompt
    inclusion_text = "\n".join([
        f"- [{c.get('type', 'other')}] {c.get('criterion', '')}"
        for c in inclusion[:10]
    ]) or "Not specified"

    exclusion_text = "\n".join([
        f"- [{c.get('type', 'other')}] {c.get('criterion', '')}"
        for c in exclusion[:10]
    ]) or "Not specified"

    rule_text = "\n".join([
        f"- {c['field']}: {c['status'].upper()} — {c['note']}"
        for c in rule_checks
    ])

    diagnoses_text = ", ".join([
        f"{d.get('disease')} ({int(d.get('confidence', 0) * 100)}%)"
        for d in patient.get("predicted_diagnosis", [])
    ]) or "None"

    prompt = GAP_EXPLAINER_PROMPT.format(
        age=patient.get("age", "Unknown"),
        gender=patient.get("gender", "Unknown"),
        symptoms=", ".join(patient.get("symptoms", [])),
        diagnoses=diagnoses_text,
        medications=", ".join(patient.get("medications", [])) or "None",
        conditions=", ".join(patient.get("existing_conditions", [])) or "None",
        trial_title=trial.get("title", "Unknown Trial")[:100],
        matched_condition=trial.get("matched_condition", "Unknown"),
        inclusion_criteria=inclusion_text,
        exclusion_criteria=exclusion_text,
        rule_checks=rule_text
    )

    # Check gap cache first
    cache_key = f"{trial.get('nct_id', '')}_{patient.get('age', '')}_{patient.get('existing_conditions', [])[:1]}"
    if cache_key in GAP_CACHE:
        print(f"[CACHE] Using cached gap analysis for {trial.get('nct_id', '')}")
        return GAP_CACHE[cache_key]

    try:
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a clinical trial eligibility expert. Always respond with valid JSON only."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    print(f"[WARNING] Rate limit hit. Retrying in 5s... (Attempt {attempt+1})")
                    await asyncio.sleep(5)
                else:
                    raise e

        result = json.loads(response.choices[0].message.content)
        print(f"[GAP] {trial.get('nct_id', '')}: {result.get('overall_status')} "
              f"({len(result.get('gaps', []))} gaps, {len(result.get('strengths', []))} strengths)")
        # Save to persistent cache
        GAP_CACHE[cache_key] = result
        _save_gap_cache(GAP_CACHE)
        return result

    except json.JSONDecodeError as e:
        print(f"[ERROR] Gap JSON parse failed: {e}")
        return {"overall_status": "needs_more_info", "gaps": [], "strengths": [], "confidence": 0}
    except Exception as e:
        print(f"[ERROR] Gap analysis failed: {e}")
        return {"overall_status": "needs_more_info", "gaps": [], "strengths": [], "confidence": 0}


# ── SECTION 3: Full Gap Analysis Pipeline ──

async def analyze_patient_trial_gaps(patient: dict, trial: dict) -> dict:
    """
    Full gap analysis: rule-based checks + LLM explanation.

    Returns:
        Complete gap analysis dict
    """
    rule_checks = run_rule_based_checks(patient, trial)
    llm_analysis = await explain_gaps_with_llm(patient, trial, rule_checks)

    return {
        "trial_id": trial.get("nct_id", ""),
        "trial_title": trial.get("title", ""),
        "matched_condition": trial.get("matched_condition", ""),
        "start_date": trial.get("start_date", "N/A"),
        "completion_date": trial.get("completion_date", "N/A"),
        "rule_checks": rule_checks,
        "llm_analysis": llm_analysis,
        "overall_status": llm_analysis.get("overall_status", "needs_more_info"),
        "confidence": llm_analysis.get("confidence", 0)
    }


async def analyze_all_trials(patient: dict, trials: list) -> list:
    """
    Runs gap analysis for all matched trials.
    Returns list sorted by confidence (best matches first).
    """
    sem = asyncio.Semaphore(1)  # Sequential to prevent concurrent 429s

    async def bounded_analyze(trial):
        async with sem:
            await asyncio.sleep(0.5)
            return await analyze_patient_trial_gaps(patient, trial)

    tasks = [bounded_analyze(trial) for trial in trials]
    results = await asyncio.gather(*tasks)

    # Sort: likely_eligible first, then by confidence
    status_order = {"likely_eligible": 0, "needs_more_info": 1, "likely_ineligible": 2}
    results.sort(key=lambda x: (
        status_order.get(x["overall_status"], 3),
        -x["confidence"]
    ))

    return results


# ── TEST ──

if __name__ == "__main__":
    print("\n" + "="*50)
    print("TEST: Gap Explainer")
    print("="*50)

    # Simulated patient
    test_patient = {
        "patient_id": "test_001",
        "symptoms": ["memory loss", "confusion", "behavioral changes"],
        "age": 70,
        "gender": "F",
        "existing_conditions": [],
        "medications": [],
        "predicted_diagnosis": [
            {"disease": "Alzheimer's Disease", "confidence": 0.80},
            {"disease": "Dementia", "confidence": 0.55}
        ],
        "matched_trials": []
    }

    # Simulated trial (as if it came from parser + validator)
    test_trial = {
        "nct_id": "NCT_TEST001",
        "title": "A Study of Memory and Cognitive Decline in Older Adults with Alzheimer's",
        "min_age": "55 Years",
        "max_age": "85 Years",
        "gender": "ALL",
        "matched_condition": "Alzheimer's Disease",
        "validated_eligibility": {
            "inclusion_criteria": [
                {"criterion": "Diagnosed with Alzheimer's Disease", "type": "diagnosis", "value": None, "extractable": True},
                {"criterion": "MMSE score between 18 and 26", "type": "cognitive_score", "value": "18-26", "extractable": True},
                {"criterion": "Age 55-85 years", "type": "age", "value": "55-85", "extractable": True},
            ],
            "exclusion_criteria": [
                {"criterion": "Currently taking antipsychotic medication", "type": "medication", "value": None, "extractable": True},
                {"criterion": "Severe cardiovascular disease", "type": "diagnosis", "value": None, "extractable": True},
            ]
        }
    }

    gap_result = analyze_patient_trial_gaps(test_patient, test_trial)

    print(f"\nOverall Status: {gap_result['overall_status'].upper()}")
    print(f"Confidence: {int(gap_result['confidence'] * 100)}%")

    print("\n--- Rule Checks ---")
    for check in gap_result["rule_checks"]:
        icon = "✅" if check["status"] == "pass" else "❌" if check["status"] == "fail" else "❓"
        print(f"  {icon} {check['field']}: {check['note']}")

    llm = gap_result["llm_analysis"]

    print(f"\n--- Strengths ({len(llm.get('strengths', []))}) ---")
    for s in llm.get("strengths", []):
        print(f"  ✅ {s}")

    print(f"\n--- Gaps ({len(llm.get('gaps', []))}) ---")
    for g in llm.get("gaps", []):
        print(f"  ⚠️  [{g.get('type')}] {g.get('criterion')}")
        print(f"      → {g.get('explanation')}")
        print(f"      Action: {g.get('action')}")

    print("\n--- Patient Summary ---")
    print(f"  {llm.get('summary_patient')}")

    print("\n--- Clinician Summary ---")
    print(f"  {llm.get('summary_clinician')}")