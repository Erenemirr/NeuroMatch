"""
NeuroMatch — Layer 2: LLM-based Neurological Diagnosis
Generates confidence-scored preliminary diagnosis from patient symptoms.
"""

import os
import json
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── NEUROLOGICAL DISEASE PROFILES ──
# Loaded from our Kaggle datasets to build reference profiles
NEURO_DISEASES = [
    "Alzheimer's Disease",
    "Parkinson's Disease",
    "Multiple Sclerosis",
    "Epilepsy",
    "ALS (Amyotrophic Lateral Sclerosis)",
    "Huntington's Disease",
    "Migraine",
    "Peripheral Neuropathy",
    "Dementia",
    "Brain Tumor",
    "Cerebrovascular Disease",
    "Myasthenia Gravis"
]


def load_symptom_dataset(dataset_path: str = "data/datasets/symptom_disease/dataset.csv") -> dict:
    """
    Loads Kaggle symptom-disease dataset and builds neurological disease profiles.
    Returns a dict: {disease_name: [list of symptoms]}
    """
    try:
        df = pd.read_csv(dataset_path)
        disease_profiles = {}

        # Neuro keywords for broader matching
        neuro_keywords = [
            "alzheimer", "parkinson", "sclerosis", "epilep", "seizure",
            "dementia", "neuro", "brain", "migraine", "neuropath",
            "huntington", "als", "myasthenia", "cerebro", "stroke"
        ]

        for _, row in df.iterrows():
            disease = row.get("Disease", "").strip()
            # Broader filter — match any neuro keyword
            if any(kw in disease.lower() for kw in neuro_keywords):
                symptoms = [
                    str(row[col]).strip()
                    for col in df.columns
                    if "Symptom" in col and pd.notna(row[col]) and str(row[col]).strip()
                ]
                if disease not in disease_profiles:
                    disease_profiles[disease] = []
                disease_profiles[disease].extend(symptoms)

        print(f"[INFO] Loaded {len(disease_profiles)} neurological disease profiles from dataset.")
        return disease_profiles

    except FileNotFoundError:
        print(f"[WARNING] Dataset not found at {dataset_path}. Using default profiles.")
        return {}
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return {}


# ── DIAGNOSIS PROMPT ──
DIAGNOSIS_PROMPT = """
You are a neurological AI assistant. Your role is to analyze patient symptoms
and generate a preliminary neurological diagnosis with confidence scores.

IMPORTANT: You are NOT making a medical diagnosis. This is for clinical trial
matching purposes only.

Patient Information:
- Age: {age}
- Gender: {gender}
- Symptoms: {symptoms}
- Duration: {duration}
- Existing Conditions: {existing_conditions}
- Current Medications: {medications}
- Report Text (if any): {report_text}

Known Neurological Disease Profiles from our dataset:
{disease_profiles}

Based on the symptoms provided, return a JSON with the following structure:
{{
    "preliminary_diagnoses": [
        {{
            "disease": "disease name",
            "confidence": 0.0 to 1.0,
            "matching_symptoms": ["symptom1", "symptom2"],
            "missing_indicators": ["what would confirm this diagnosis"]
        }}
    ],
    "recommended_tests": ["test1", "test2"],
    "urgency_level": "low/medium/high",
    "summary": "brief clinical summary in English",
    "disclaimer": "This is not a medical diagnosis. Please consult a neurologist."
}}

Return top 3 most likely diagnoses, ordered by confidence score.
Only return valid JSON, nothing else.
"""


def generate_diagnosis(patient_profile: dict, disease_profiles: dict = None) -> dict:
    """
    Generates a confidence-scored neurological diagnosis from patient symptoms.

    Args:
        patient_profile: Patient data following patient_schema.json
        disease_profiles: Optional dict of disease->symptoms from our dataset

    Returns:
        Diagnosis dict with confidence scores
    """
    # Format disease profiles for prompt
    if disease_profiles:
        profiles_text = "\n".join([
            f"- {disease}: {', '.join(symptoms[:5])}"
            for disease, symptoms in list(disease_profiles.items())[:10]
        ])
    else:
        profiles_text = "Using general neurological knowledge."

    prompt = DIAGNOSIS_PROMPT.format(
        age=patient_profile.get("age", "Unknown"),
        gender=patient_profile.get("gender", "Unknown"),
        symptoms=", ".join(patient_profile.get("symptoms", [])),
        duration=patient_profile.get("duration", "Unknown"),
        existing_conditions=", ".join(patient_profile.get("existing_conditions", [])) or "None",
        medications=", ".join(patient_profile.get("medications", [])) or "None",
        report_text=patient_profile.get("report_text") or "Not provided",
        disease_profiles=profiles_text
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a neurological AI assistant specialized in clinical trial matching. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        # Add to patient profile
        patient_profile["predicted_diagnosis"] = [
            {
                "disease": d["disease"],
                "confidence": d["confidence"]
            }
            for d in result.get("preliminary_diagnoses", [])
        ]

        print(f"[INFO] Diagnosis generated: {[d['disease'] for d in patient_profile['predicted_diagnosis']]}")
        return result

    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON decode failed: {e}")
        return {"error": "Failed to parse diagnosis response"}
    except Exception as e:
        print(f"[ERROR] Diagnosis generation failed: {e}")
        return {"error": str(e)}


def format_diagnosis_output(diagnosis_result: dict, audience: str = "patient") -> str:
    """
    Formats diagnosis output for different audiences.

    Args:
        diagnosis_result: Raw diagnosis from generate_diagnosis()
        audience: "patient" | "neurologist" | "gp" | "researcher"

    Returns:
        Formatted string output
    """
    if "error" in diagnosis_result:
        return f"Error: {diagnosis_result['error']}"

    diagnoses = diagnosis_result.get("preliminary_diagnoses", [])
    tests = diagnosis_result.get("recommended_tests", [])
    urgency = diagnosis_result.get("urgency_level", "unknown")
    summary = diagnosis_result.get("summary", "")

    if audience == "patient":
        output = "🧠 NeuroMatch Analysis\n\n"
        output += "Based on your symptoms, here are possible conditions:\n\n"
        for d in diagnoses[:3]:
            confidence_pct = int(d["confidence"] * 100)
            output += f"• {d['disease']}: {confidence_pct}% match\n"
        output += f"\nRecommended next steps: {', '.join(tests)}\n"
        output += f"\n⚠️ {diagnosis_result.get('disclaimer', '')}"

    elif audience == "neurologist":
        output = "Clinical Preliminary Assessment\n\n"
        output += f"Summary: {summary}\n\n"
        for d in diagnoses:
            output += f"[{d['confidence']:.2f}] {d['disease']}\n"
            output += f"  Matching: {', '.join(d.get('matching_symptoms', []))}\n"
            output += f"  Missing indicators: {', '.join(d.get('missing_indicators', []))}\n\n"
        output += f"Urgency: {urgency.upper()}\n"
        output += f"Recommended tests: {', '.join(tests)}"

    elif audience == "gp":
        output = "Referral Summary\n\n"
        top = diagnoses[0] if diagnoses else {}
        output += f"Primary concern: {top.get('disease', 'Unknown')} ({int(top.get('confidence', 0) * 100)}% confidence)\n"
        output += f"Urgency level: {urgency}\n"
        output += f"Recommended: Neurology referral + {', '.join(tests[:2])}"

    elif audience == "researcher":
        output = json.dumps(diagnosis_result, indent=2)

    else:
        output = summary

    return output


# ── TEST ──
if __name__ == "__main__":
    # Load dataset profiles
    disease_profiles = load_symptom_dataset()

    # Test patient
    test_patient = {
        "patient_id": "test_001",
        "symptoms": ["memory loss", "behavioral changes", "confusion", "difficulty with daily tasks"],
        "duration": "6 months",
        "age": 70,
        "gender": "F",
        "existing_conditions": [],
        "medications": [],
        "report_text": None,
        "predicted_diagnosis": [],
        "matched_trials": []
    }

    print("\n" + "="*50)
    print("RUNNING DIAGNOSIS TEST")
    print("="*50)

    result = generate_diagnosis(test_patient, disease_profiles)

    print("\n--- PATIENT VIEW ---")
    print(format_diagnosis_output(result, "patient"))

    print("\n--- NEUROLOGIST VIEW ---")
    print(format_diagnosis_output(result, "neurologist"))

    print("\n--- GP VIEW ---")
    print(format_diagnosis_output(result, "gp"))