"""
NeuroMatch — OASIS-2 Integration
Uses real longitudinal MRI + cognitive data from OASIS-2 to estimate
likely MMSE/CDR scores for a patient based on age, gender, and diagnosis.
Enriches gap analysis with evidence-based score estimates.
"""

import os
import pandas as pd
import numpy as np

OASIS_PATH = "data/datasets/oasis2/oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx"


def load_oasis_data(path: str = OASIS_PATH) -> pd.DataFrame:
    """Loads and cleans OASIS-2 demographic dataset."""
    try:
        df = pd.read_excel(path)
        df = df.dropna(subset=["Age", "MMSE", "CDR"])
        df["M/F"] = df["M/F"].str.upper().str.strip()
        print(f"[OASIS] Loaded {len(df)} records from OASIS-2 dataset")
        return df
    except FileNotFoundError:
        print(f"[OASIS] Dataset not found at {path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"[OASIS] Failed to load dataset: {e}")
        return pd.DataFrame()


def find_similar_patients(df: pd.DataFrame, age: int, gender: str,
                           group: str = None, age_range: int = 10) -> pd.DataFrame:
    """
    Finds OASIS-2 patients similar to the input patient.

    Args:
        df: OASIS-2 dataframe
        age: Patient age
        gender: "M" or "F"
        group: "Demented" | "Nondemented" | None (both)
        age_range: ± years to search

    Returns:
        Filtered dataframe of similar patients
    """
    mask = (
        (df["Age"] >= age - age_range) &
        (df["Age"] <= age + age_range)
    )

    gender_upper = gender.upper()[0] if gender else None
    if gender_upper in ["M", "F"]:
        mask &= (df["M/F"] == gender_upper)

    if group:
        mask &= (df["Group"] == group)

    similar = df[mask]
    print(f"[OASIS] Found {len(similar)} similar patients "
          f"(age {age}±{age_range}, gender={gender_upper}, group={group or 'all'})")
    return similar


def estimate_cognitive_scores(age: int, gender: str, diagnosis: str = None) -> dict:
    """
    Estimates likely MMSE and CDR score ranges for a patient
    based on OASIS-2 real data.

    Args:
        age: Patient age
        gender: "M" or "F"
        diagnosis: Top predicted diagnosis (used to determine group filter)

    Returns:
        Dict with estimated score ranges and statistics
    """
    df = load_oasis_data()
    if df.empty:
        return {"available": False, "reason": "OASIS-2 dataset not loaded"}

    # Determine group based on diagnosis
    group = None
    if diagnosis:
        diag_lower = diagnosis.lower()
        if any(kw in diag_lower for kw in ["alzheimer", "dementia", "vascular"]):
            group = "Demented"
        elif any(kw in diag_lower for kw in ["migraine", "neuropathy", "epilepsy"]):
            group = "Nondemented"

    similar = find_similar_patients(df, age, gender, group=group)

    # Fallback: widen age range if too few patients
    if len(similar) < 5:
        similar = find_similar_patients(df, age, gender, group=None, age_range=15)

    if similar.empty:
        return {"available": False, "reason": "No similar patients found in OASIS-2"}

    mmse_stats = {
        "mean": round(similar["MMSE"].mean(), 1),
        "std": round(similar["MMSE"].std(), 1),
        "min": round(similar["MMSE"].min(), 1),
        "max": round(similar["MMSE"].max(), 1),
        "median": round(similar["MMSE"].median(), 1),
        "estimated_range": (
            round(max(0, similar["MMSE"].mean() - similar["MMSE"].std()), 1),
            round(min(30, similar["MMSE"].mean() + similar["MMSE"].std()), 1)
        )
    }

    cdr_stats = {
        "mean": round(similar["CDR"].mean(), 2),
        "distribution": similar["CDR"].value_counts().to_dict(),
        "most_common": float(similar["CDR"].mode()[0])
    }

    # Brain volume stats (MRI data)
    nwbv_stats = {}
    if "nWBV" in similar.columns:
        nwbv_stats = {
            "mean": round(similar["nWBV"].mean(), 4),
            "range": (round(similar["nWBV"].min(), 4), round(similar["nWBV"].max(), 4))
        }

    group_dist = similar["Group"].value_counts().to_dict()

    return {
        "available": True,
        "sample_size": len(similar),
        "age_range_used": f"{age - 15} - {age + 15}",
        "gender": gender,
        "group_distribution": group_dist,
        "mmse": mmse_stats,
        "cdr": cdr_stats,
        "brain_volume": nwbv_stats,
        "interpretation": _interpret_scores(mmse_stats, cdr_stats, group_dist)
    }


def _interpret_scores(mmse: dict, cdr: dict, group_dist: dict) -> str:
    """Generates a plain-English interpretation of the estimated scores."""
    total = sum(group_dist.values())
    demented_pct = int(group_dist.get("Demented", 0) / total * 100) if total > 0 else 0

    mmse_range = mmse["estimated_range"]
    mmse_severity = ""
    if mmse["mean"] >= 24:
        mmse_severity = "mild cognitive impairment or none"
    elif mmse["mean"] >= 18:
        mmse_severity = "mild to moderate cognitive impairment"
    else:
        mmse_severity = "moderate to severe cognitive impairment"

    return (
        f"Among similar patients in OASIS-2 ({mmse['mean']} avg MMSE), "
        f"typical MMSE scores range from {mmse_range[0]} to {mmse_range[1]}, "
        f"suggesting {mmse_severity}. "
        f"CDR score is most commonly {cdr['most_common']}. "
        f"{demented_pct}% of similar-aged patients were classified as demented."
    )


def enrich_patient_with_oasis(patient_profile: dict) -> dict:
    """
    Adds OASIS-2 estimated cognitive scores to a patient profile.
    Called before gap analysis to provide evidence-based score estimates.

    Args:
        patient_profile: Standard patient dict

    Returns:
        Patient profile enriched with oasis_estimates field
    """
    age = patient_profile.get("age")
    gender = patient_profile.get("gender", "")
    diagnoses = patient_profile.get("predicted_diagnosis", [])
    top_diagnosis = diagnoses[0].get("disease", "") if diagnoses else ""

    if not age:
        return patient_profile

    estimates = estimate_cognitive_scores(age, gender, top_diagnosis)
    patient_profile["oasis_estimates"] = estimates

    if estimates.get("available"):
        mmse_range = estimates["mmse"]["estimated_range"]
        print(f"[OASIS] Estimated MMSE range for patient: {mmse_range[0]}-{mmse_range[1]}")

    return patient_profile


def format_oasis_summary(estimates: dict, audience: str = "patient") -> str:
    """Formats OASIS-2 estimates for display."""
    if not estimates.get("available"):
        return ""

    mmse = estimates["mmse"]
    cdr = estimates["cdr"]
    n = estimates["sample_size"]

    if audience == "patient":
        return (
            f"\n📊 Based on {n} similar patients in clinical research data:\n"
            f"  • Typical cognitive score (MMSE): {mmse['estimated_range'][0]}–{mmse['estimated_range'][1]} out of 30\n"
            f"  • {estimates['interpretation']}"
        )
    elif audience in ["neurologist", "gp"]:
        return (
            f"\n[OASIS-2 Reference — n={n}]\n"
            f"  MMSE: mean={mmse['mean']} ± {mmse['std']} "
            f"(estimated range {mmse['estimated_range'][0]}–{mmse['estimated_range'][1]})\n"
            f"  CDR: mode={cdr['most_common']}, distribution={cdr['distribution']}\n"
            f"  {estimates['interpretation']}"
        )
    return estimates["interpretation"]


# ── TEST ──

if __name__ == "__main__":
    print("\n" + "="*55)
    print("TEST: OASIS-2 Estimator")
    print("="*55)

    print("\n--- Test 1: 70yo Female, Alzheimer's ---")
    result = estimate_cognitive_scores(70, "F", "Alzheimer's Disease")
    if result["available"]:
        print(f"  Sample size: {result['sample_size']}")
        print(f"  MMSE mean: {result['mmse']['mean']} ± {result['mmse']['std']}")
        print(f"  MMSE estimated range: {result['mmse']['estimated_range']}")
        print(f"  CDR most common: {result['cdr']['most_common']}")
        print(f"  Group distribution: {result['group_distribution']}")
        print(f"  Brain volume (nWBV): {result['brain_volume']}")
        print(f"\n  Interpretation: {result['interpretation']}")
        print(f"\n  Patient view:\n{format_oasis_summary(result, 'patient')}")
        print(f"\n  Clinician view:\n{format_oasis_summary(result, 'neurologist')}")

    print("\n--- Test 2: 65yo Male, Parkinson's ---")
    result2 = estimate_cognitive_scores(65, "M", "Parkinson's Disease")
    if result2["available"]:
        print(f"  Sample size: {result2['sample_size']}")
        print(f"  MMSE range: {result2['mmse']['estimated_range']}")
        print(f"  Interpretation: {result2['interpretation']}")