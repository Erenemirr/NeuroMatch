"""
NeuroMatch — Layer 4: Validation Layer
Validates and cleans LLM-parsed eligibility criteria.
Catches hallucinations, fixes missing fields, ensures schema consistency.
"""

import json
from typing import Any


# ── VALID SCHEMA ──

VALID_TYPES = {"age", "diagnosis", "biomarker", "medication", "cognitive_score", "functional", "other"}

REQUIRED_KEYS = {"criterion", "type", "value", "extractable"}


# ── SECTION 1: Criterion Validator ──

def validate_criterion(item: Any, section: str, index: int) -> tuple[dict | None, list[str]]:
    """
    Validates a single criterion item.

    Returns:
        (cleaned_item, list_of_warnings)
        cleaned_item is None if the item should be skipped entirely
    """
    warnings = []

    # Must be a dict
    if not isinstance(item, dict):
        warnings.append(f"[{section}][{index}] Not a dict, skipped: {str(item)[:50]}")
        return None, warnings

    cleaned = dict(item)

    # Fix missing keys
    missing = REQUIRED_KEYS - set(cleaned.keys())
    if missing:
        for key in missing:
            if key == "value":
                cleaned[key] = None
            elif key == "extractable":
                cleaned[key] = True
            elif key == "type":
                cleaned[key] = "other"
            elif key == "criterion":
                cleaned[key] = ""
        if missing:
            warnings.append(f"[{section}][{index}] Missing keys filled: {missing}")

    # Skip empty criterion
    if not cleaned.get("criterion") or len(str(cleaned["criterion"]).strip()) < 3:
        warnings.append(f"[{section}][{index}] Empty criterion skipped")
        return None, warnings

    # Fix invalid type
    if cleaned.get("type") not in VALID_TYPES:
        original_type = cleaned.get("type")
        cleaned["type"] = infer_type(cleaned["criterion"])
        warnings.append(
            f"[{section}][{index}] Invalid type '{original_type}' → inferred '{cleaned['type']}'"
        )

    # Ensure extractable is bool
    if not isinstance(cleaned.get("extractable"), bool):
        cleaned["extractable"] = True

    # Truncate overly long criterion text
    if len(str(cleaned["criterion"])) > 500:
        cleaned["criterion"] = str(cleaned["criterion"])[:500] + "..."
        warnings.append(f"[{section}][{index}] Criterion text truncated to 500 chars")

    return cleaned, warnings


def infer_type(criterion_text: str) -> str:
    """
    Infers criterion type from text when LLM returns invalid type.
    """
    text = criterion_text.lower()

    if any(kw in text for kw in ["age", "year", "old", "years"]):
        return "age"
    elif any(kw in text for kw in ["mmse", "moca", "cdr", "score", "cognitive", "adas"]):
        return "cognitive_score"
    elif any(kw in text for kw in ["drug", "medication", "treatment", "therapy", "taking", "prescribed"]):
        return "medication"
    elif any(kw in text for kw in ["diagnosed", "diagnosis", "disease", "disorder", "condition", "history of"]):
        return "diagnosis"
    elif any(kw in text for kw in ["biomarker", "amyloid", "tau", "apoe", "genetic", "blood", "csf", "pet"]):
        return "biomarker"
    elif any(kw in text for kw in ["walk", "mobility", "function", "daily", "independent", "activity"]):
        return "functional"
    else:
        return "other"


# ── SECTION 2: Full Parsed Output Validator ──

def validate_parsed_criteria(parsed: dict, trial_id: str = "") -> dict:
    """
    Validates the full parsed eligibility output from the LLM.
    Fixes issues and returns a clean, validated structure.

    Args:
        parsed: Raw LLM output from parse_eligibility_with_llm()
        trial_id: NCT ID for logging

    Returns:
        Validated dict with inclusion_criteria, exclusion_criteria, validation_warnings
    """
    result = {
        "inclusion_criteria": [],
        "exclusion_criteria": [],
        "validation_warnings": [],
        "validation_passed": True
    }

    if not isinstance(parsed, dict):
        result["validation_warnings"].append(f"[{trial_id}] Input is not a dict, returning empty.")
        result["validation_passed"] = False
        return result

    for section in ["inclusion_criteria", "exclusion_criteria"]:
        raw_list = parsed.get(section, [])

        if not isinstance(raw_list, list):
            result["validation_warnings"].append(
                f"[{trial_id}][{section}] Expected list, got {type(raw_list).__name__}"
            )
            continue

        for i, item in enumerate(raw_list):
            cleaned, warnings = validate_criterion(item, section, i)
            result["validation_warnings"].extend(warnings)
            if cleaned is not None:
                result[section].append(cleaned)

    # Summary
    total_warnings = len(result["validation_warnings"])
    if total_warnings > 0:
        result["validation_passed"] = False
        print(f"[VALIDATOR] {trial_id}: {len(result['inclusion_criteria'])} inclusion + "
              f"{len(result['exclusion_criteria'])} exclusion criteria validated "
              f"({total_warnings} warnings)")
    else:
        print(f"[VALIDATOR] {trial_id}: All criteria passed validation ✓")

    return result


def validate_trials_batch(enriched_trials: list) -> list:
    """
    Runs validation on a batch of enriched trials.

    Args:
        enriched_trials: Output from parse_trials_batch()

    Returns:
        Trials with validated_eligibility added
    """
    for trial in enriched_trials:
        parsed = trial.get("parsed_eligibility", {})
        trial["validated_eligibility"] = validate_parsed_criteria(
            parsed, trial.get("nct_id", "")
        )
    return enriched_trials


# ── SECTION 3: Patient Profile Validator ──

def validate_patient_profile(profile: dict) -> tuple[bool, list[str]]:
    """
    Validates incoming patient profile against our schema.

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    required_fields = ["symptoms", "age", "gender"]

    for field in required_fields:
        if field not in profile or profile[field] is None:
            issues.append(f"Missing required field: '{field}'")

    if "symptoms" in profile:
        if not isinstance(profile["symptoms"], list):
            issues.append("'symptoms' must be a list")
        elif len(profile["symptoms"]) == 0:
            issues.append("'symptoms' list is empty")

    if "age" in profile and profile["age"] is not None:
        try:
            age = int(profile["age"])
            if age < 0 or age > 120:
                issues.append(f"Age {age} is out of valid range (0-120)")
        except (ValueError, TypeError):
            issues.append(f"Age '{profile['age']}' is not a valid number")

    if "gender" in profile and profile["gender"] not in ["M", "F", "male", "female", "Male", "Female", "ALL", None]:
        issues.append(f"Gender '{profile['gender']}' is not recognized")

    is_valid = len(issues) == 0
    return is_valid, issues


# ── TEST ──

if __name__ == "__main__":
    print("\n" + "="*50)
    print("TEST 1: Criterion Validation")
    print("="*50)

    # Simulate messy LLM output
    messy_parsed = {
        "inclusion_criteria": [
            {"criterion": "Age 50-80 years", "type": "age", "value": "50-80", "extractable": True},
            {"criterion": "MMSE score 18-26", "type": "invalid_type", "value": "18-26", "extractable": True},
            {"criterion": "", "type": "other", "value": None, "extractable": True},  # empty - should skip
            {"criterion": "Diagnosed with Alzheimer's Disease", "value": None},  # missing keys
            "this is not a dict",  # completely wrong format
        ],
        "exclusion_criteria": [
            {"criterion": "Currently taking antipsychotic medication", "type": "medication", "value": None, "extractable": True},
            {"criterion": "Severe cardiovascular disease", "type": "diagnosis", "value": None, "extractable": False},
        ]
    }

    validated = validate_parsed_criteria(messy_parsed, "TEST_NCT")

    print(f"\nInclusion ({len(validated['inclusion_criteria'])}):")
    for c in validated["inclusion_criteria"]:
        print(f"  ✅ [{c['type']}] {c['criterion'][:70]}")

    print(f"\nExclusion ({len(validated['exclusion_criteria'])}):")
    for c in validated["exclusion_criteria"]:
        print(f"  ❌ [{c['type']}] {c['criterion'][:70]}")

    print(f"\nWarnings ({len(validated['validation_warnings'])}):")
    for w in validated["validation_warnings"]:
        print(f"  ⚠️  {w}")

    print("\n" + "="*50)
    print("TEST 2: Patient Profile Validation")
    print("="*50)

    valid_patient = {
        "patient_id": "001",
        "symptoms": ["memory loss", "confusion"],
        "age": 70,
        "gender": "F",
        "existing_conditions": [],
        "medications": [],
        "report_text": None,
        "predicted_diagnosis": [],
        "matched_trials": []
    }

    invalid_patient = {
        "patient_id": "002",
        "symptoms": [],       # empty
        "age": 200,           # invalid
        "gender": "unknown",  # unrecognized
    }

    is_valid, issues = validate_patient_profile(valid_patient)
    print(f"\nValid patient: {'✅ PASSED' if is_valid else '❌ FAILED'}")

    is_valid, issues = validate_patient_profile(invalid_patient)
    print(f"Invalid patient: {'✅ PASSED' if is_valid else '❌ FAILED'}")
    for issue in issues:
        print(f"  ⚠️  {issue}")