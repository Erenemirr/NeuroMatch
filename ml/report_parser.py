"""
NeuroMatch — Report Parser
Extracts clinical information from uploaded MRI/brain reports (PDF or text).
Parses: symptoms, MMSE scores, CDR scores, diagnoses, medications, duration.
Feeds structured data into the patient profile for pipeline use.
"""

import os
import re
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ── SECTION 1: Text Extraction ──

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts raw text from a PDF file."""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        print(f"[PARSER] Extracted {len(text)} chars from PDF: {pdf_path}")
        return text
    except ImportError:
        print("[PARSER] pdfplumber not installed, trying pypdf2...")
        try:
            import PyPDF2
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text
        except Exception as e:
            print(f"[PARSER] PDF extraction failed: {e}")
            return ""
    except Exception as e:
        print(f"[PARSER] PDF extraction failed: {e}")
        return ""


def extract_text_from_file(file_path: str) -> str:
    """Handles PDF, txt, and raw text input."""
    if not file_path:
        return ""

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in [".txt", ".text", ".md"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        print(f"[PARSER] Unsupported file type: {ext}")
        return ""


# ── SECTION 2: Rule-Based Extraction (fast, no API) ──

def extract_mmse_score(text: str) -> int | None:
    """Extracts MMSE score from text using regex."""
    patterns = [
        r"MMSE[:\s]+(\d{1,2})[/\s]*30",
        r"Mini.Mental[:\s]+(\d{1,2})",
        r"MMSE score[:\s]+(\d{1,2})",
        r"scored\s+(\d{1,2})\s+on.*MMSE",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 0 <= score <= 30:
                print(f"[PARSER] Found MMSE score: {score}")
                return score
    return None


def extract_cdr_score(text: str) -> float | None:
    """Extracts CDR score from text."""
    patterns = [
        r"CDR[:\s]+([\d.]+)",
        r"Clinical Dementia Rating[:\s]+([\d.]+)",
        r"CDR score[:\s]+([\d.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            if score in [0, 0.5, 1, 2, 3]:
                print(f"[PARSER] Found CDR score: {score}")
                return score
    return None


def extract_duration(text: str) -> str | None:
    """Extracts symptom duration from text."""
    patterns = [
        r"(\d+)\s*(year|month|week)s?\s*(ago|duration|history)",
        r"(over|for|past)\s+(\d+)\s*(year|month|week)s?",
        r"symptoms?\s+(for|over|past)\s+(\d+)\s*(year|month|week)s?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            duration = match.group(0)
            print(f"[PARSER] Found duration: {duration}")
            return duration
    return None


def extract_age_from_text(text: str) -> int | None:
    """Extracts patient age from report text."""
    patterns = [
        r"(\d{2,3})[\s-]*(year|yr)[\s-]*old",
        r"age[:\s]+(\d{2,3})",
        r"aged?\s+(\d{2,3})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            for group in match.groups():
                if group and group.isdigit():
                    age = int(group)
                    if 0 < age < 120:
                        print(f"[PARSER] Found age: {age}")
                        return age
    return None


def extract_gender_from_text(text: str) -> str | None:
    """Extracts patient gender from report text."""
    if re.search(r"\b(female|woman|girl|she|her)\b", text, re.IGNORECASE):
        return "F"
    elif re.search(r"\b(male|man|boy|he|his)\b", text, re.IGNORECASE):
        return "M"
    return None


# ── SECTION 3: LLM-Based Full Extraction ──

REPORT_PARSE_PROMPT = """
You are a clinical report parser. Extract structured information from the following medical report.

Medical Report:
{report_text}

Extract and return ONLY a valid JSON object with this structure:
{{
    "symptoms": ["symptom1", "symptom2"],
    "diagnoses": ["diagnosis1", "diagnosis2"],
    "medications": ["medication1", "medication2"],
    "existing_conditions": ["condition1", "condition2"],
    "mmse_score": null or integer (0-30),
    "cdr_score": null or float (0, 0.5, 1, 2, 3),
    "duration": "e.g. 6 months or null",
    "age": null or integer,
    "gender": "M" or "F" or null,
    "key_findings": ["finding1", "finding2"],
    "brain_regions_affected": ["region1", "region2"]
}}

Rules:
- symptoms: neurological symptoms mentioned (memory loss, tremor, etc.)
- diagnoses: any mentioned diagnoses or suspected conditions
- key_findings: important MRI or clinical findings
- brain_regions_affected: any brain regions mentioned (temporal lobe, hippocampus, etc.)
- If information is not present, use null or empty list
- Only return valid JSON, nothing else
"""


def parse_report_with_llm(report_text: str) -> dict:
    """Uses LLM to extract structured data from a medical report."""
    if not report_text or len(report_text.strip()) < 20:
        return {}

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a clinical report parser. Extract structured medical information. Return only valid JSON."
                },
                {
                    "role": "user",
                    "content": REPORT_PARSE_PROMPT.format(
                        report_text=report_text[:4000]
                    )
                }
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        parsed = json.loads(response.choices[0].message.content)
        print(f"[PARSER] LLM extracted: {len(parsed.get('symptoms', []))} symptoms, "
              f"{len(parsed.get('diagnoses', []))} diagnoses, "
              f"MMSE={parsed.get('mmse_score')}, CDR={parsed.get('cdr_score')}")
        return parsed

    except Exception as e:
        print(f"[PARSER] LLM extraction failed: {e}")
        return {}


# ── SECTION 4: Patient Profile Enrichment ──

def enrich_profile_from_report(patient_profile: dict, report_text: str = None,
                                 report_path: str = None) -> dict:
    """
    Main function: takes a patient profile and a report (text or file path),
    extracts clinical info and enriches the profile.

    Args:
        patient_profile: Existing patient profile dict
        report_text: Raw report text (if already extracted)
        report_path: Path to PDF or text file

    Returns:
        Enriched patient profile
    """
    # Get text
    if report_path and not report_text:
        report_text = extract_text_from_file(report_path)

    if not report_text:
        print("[PARSER] No report text available")
        return patient_profile

    # Store raw report text
    patient_profile["report_text"] = report_text[:3000]

    # Rule-based fast extraction
    mmse = extract_mmse_score(report_text)
    cdr = extract_cdr_score(report_text)
    duration = extract_duration(report_text)
    age_from_report = extract_age_from_text(report_text)
    gender_from_report = extract_gender_from_text(report_text)

    # LLM full extraction
    llm_data = parse_report_with_llm(report_text)

    # Merge into patient profile (existing values take priority)
    if llm_data.get("symptoms"):
        existing = set(patient_profile.get("symptoms", []))
        new_symptoms = [s for s in llm_data["symptoms"] if s not in existing]
        patient_profile["symptoms"] = list(existing) + new_symptoms
        print(f"[PARSER] Added {len(new_symptoms)} symptoms from report")

    if llm_data.get("medications") and not patient_profile.get("medications"):
        patient_profile["medications"] = llm_data["medications"]

    if llm_data.get("existing_conditions") and not patient_profile.get("existing_conditions"):
        patient_profile["existing_conditions"] = llm_data["existing_conditions"]

    if not patient_profile.get("duration") and duration:
        patient_profile["duration"] = duration
    elif not patient_profile.get("duration") and llm_data.get("duration"):
        patient_profile["duration"] = llm_data["duration"]

    if not patient_profile.get("age") and age_from_report:
        patient_profile["age"] = age_from_report
    elif not patient_profile.get("age") and llm_data.get("age"):
        patient_profile["age"] = llm_data["age"]

    if not patient_profile.get("gender") and gender_from_report:
        patient_profile["gender"] = gender_from_report
    elif not patient_profile.get("gender") and llm_data.get("gender"):
        patient_profile["gender"] = llm_data["gender"]

    # Store extracted scores for gap explainer
    report_findings = {
        "mmse_score": mmse or llm_data.get("mmse_score"),
        "cdr_score": cdr or llm_data.get("cdr_score"),
        "key_findings": llm_data.get("key_findings", []),
        "brain_regions_affected": llm_data.get("brain_regions_affected", []),
        "diagnoses_from_report": llm_data.get("diagnoses", [])
    }
    patient_profile["report_findings"] = report_findings

    print(f"[PARSER] Profile enriched: age={patient_profile.get('age')}, "
          f"gender={patient_profile.get('gender')}, "
          f"symptoms={len(patient_profile.get('symptoms', []))}, "
          f"MMSE={report_findings['mmse_score']}, CDR={report_findings['cdr_score']}")

    return patient_profile


# ── TEST ──

if __name__ == "__main__":
    print("\n" + "="*55)
    print("TEST: Report Parser")
    print("="*55)

    # Simulated MRI report text
    sample_report = """
    NEUROLOGY CLINIC — PATIENT REPORT
    
    Patient: Female, 72 years old
    Date: May 2026
    
    Chief Complaint:
    Patient presents with progressive memory loss over approximately 18 months.
    Family reports increasing confusion, difficulty with daily tasks, and behavioral changes.
    
    Cognitive Assessment:
    MMSE Score: 22/30 (mild cognitive impairment)
    CDR Score: 0.5 (questionable dementia)
    
    MRI Findings:
    T1-weighted MRI shows cortical atrophy predominantly in the temporal and parietal lobes.
    Hippocampal volume reduction noted bilaterally.
    No evidence of acute infarct or hemorrhage.
    Mild periventricular white matter changes consistent with age.
    
    Impression:
    Findings are consistent with early Alzheimer's Disease.
    Differential diagnosis includes frontotemporal dementia.
    
    Current Medications:
    Donepezil 5mg daily
    
    Recommendations:
    Amyloid PET scan recommended.
    Neuropsychological testing advised.
    Follow-up in 6 months.
    """

    # Start with minimal patient profile
    patient_profile = {
        "patient_id": "parser_test_001",
        "symptoms": ["memory loss"],  # only one known symptom
        "age": None,
        "gender": None,
        "duration": None,
        "existing_conditions": [],
        "medications": [],
        "report_text": None,
        "predicted_diagnosis": [],
        "matched_trials": []
    }

    print("\nBefore enrichment:")
    print(f"  Symptoms: {patient_profile['symptoms']}")
    print(f"  Age: {patient_profile['age']}")
    print(f"  Gender: {patient_profile['gender']}")

    enriched = enrich_profile_from_report(patient_profile, report_text=sample_report)

    print("\nAfter enrichment:")
    print(f"  Symptoms: {enriched['symptoms']}")
    print(f"  Age: {enriched['age']}")
    print(f"  Gender: {enriched['gender']}")
    print(f"  Duration: {enriched['duration']}")
    print(f"  Medications: {enriched['medications']}")
    print(f"  MMSE: {enriched['report_findings']['mmse_score']}")
    print(f"  CDR: {enriched['report_findings']['cdr_score']}")
    print(f"  Key findings: {enriched['report_findings']['key_findings']}")
    print(f"  Brain regions: {enriched['report_findings']['brain_regions_affected']}")