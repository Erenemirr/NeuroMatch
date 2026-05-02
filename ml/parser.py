"""
NeuroMatch — Layer 3: ClinicalTrials.gov API + LLM Eligibility Parser
Fetches real neurological trials and parses eligibility criteria into structured JSON.
"""

import os
import json
import requests
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CLINICALTRIALS_BASE = "https://clinicaltrials.gov/api/v2/studies"

NEURO_CONDITIONS = [
    "Alzheimer's Disease",
    "Parkinson's Disease",
    "Multiple Sclerosis",
    "Epilepsy",
    "ALS",
    "Dementia",
    "Huntington's Disease",
    "Migraine",
    "Neuropathy",
]


# ── SECTION 1: ClinicalTrials.gov API ──

def fetch_trials(condition: str, location: str = None, max_results: int = 10) -> list:
    params = {
        "query.cond": condition,
        "filter.overallStatus": "RECRUITING",
        "pageSize": max_results,
        "format": "json",
    }
    if location:
        params["query.locn"] = location

    try:
        print(f"[INFO] Fetching trials for: {condition}")
        response = requests.get(CLINICALTRIALS_BASE, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        studies = data.get("studies", [])
        print(f"[INFO] Found {len(studies)} trials for {condition}")
        return [parse_trial_summary(s) for s in studies]
    except requests.exceptions.Timeout:
        print(f"[ERROR] Request timed out for {condition}")
        return []
    except Exception as e:
        print(f"[ERROR] API request failed: {e}")
        return []


def parse_trial_summary(study: dict) -> dict:
    protocol = study.get("protocolSection", {})
    id_module = protocol.get("identificationModule", {})
    status_module = protocol.get("statusModule", {})
    eligibility_module = protocol.get("eligibilityModule", {})
    description_module = protocol.get("descriptionModule", {})
    contacts_module = protocol.get("contactsLocationsModule", {})

    locations = []
    for loc in contacts_module.get("locations", []):
        city = loc.get("city", "")
        country = loc.get("country", "")
        if city or country:
            locations.append(f"{city}, {country}".strip(", "))

    return {
        "nct_id": id_module.get("nctId", "N/A"),
        "title": id_module.get("briefTitle", "N/A"),
        "status": status_module.get("overallStatus", "N/A"),
        "phase": status_module.get("phase", "N/A"),
        "eligibility_criteria_raw": eligibility_module.get("eligibilityCriteria", ""),
        "min_age": eligibility_module.get("minimumAge", "N/A"),
        "max_age": eligibility_module.get("maximumAge", "N/A"),
        "gender": eligibility_module.get("sex", "ALL"),
        "brief_summary": description_module.get("briefSummary", "N/A"),
        "locations": locations[:5],
    }


def fetch_trials_for_diagnoses(diagnoses: list, max_per_condition: int = 5) -> list:
    all_trials = []
    seen_ids = set()

    for diagnosis in diagnoses[:2]:
        condition = diagnosis.get("disease", "")
        if not condition:
            continue
        trials = fetch_trials(condition, max_results=max_per_condition)
        for trial in trials:
            if trial["nct_id"] not in seen_ids:
                seen_ids.add(trial["nct_id"])
                trial["matched_condition"] = condition
                all_trials.append(trial)

    return all_trials


# ── SECTION 2: LLM Eligibility Parser ──

ELIGIBILITY_PARSE_PROMPT = """
You are a clinical trial eligibility expert. Parse the following eligibility criteria text
and extract structured inclusion and exclusion criteria.

Eligibility Criteria Text:
{criteria_text}

Return ONLY a valid JSON object with this exact structure:
{{
    "inclusion_criteria": [
        {{
            "criterion": "description of the criterion",
            "type": "age|diagnosis|biomarker|medication|cognitive_score|functional|other",
            "value": "specific value or null",
            "extractable": true
        }}
    ],
    "exclusion_criteria": [
        {{
            "criterion": "description of the criterion",
            "type": "age|diagnosis|biomarker|medication|cognitive_score|functional|other",
            "value": "specific value or null",
            "extractable": true
        }}
    ]
}}

Rules:
- Extract ALL criteria
- For age criteria, extract exact numbers
- For cognitive scores, extract ranges (e.g. MMSE 18-26)
- Only return valid JSON, nothing else
"""


def parse_eligibility_with_llm(criteria_text: str, trial_id: str = "") -> dict:
    if not criteria_text or len(criteria_text.strip()) < 20:
        return {"inclusion_criteria": [], "exclusion_criteria": []}

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a clinical trial expert. Always respond with valid JSON only. No markdown, no explanation."
                },
                {
                    "role": "user",
                    "content": ELIGIBILITY_PARSE_PROMPT.format(
                        criteria_text=criteria_text[:3000]
                    )
                }
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        raw = response.choices[0].message.content
        parsed = json.loads(raw)
        print(f"[INFO] Parsed {len(parsed.get('inclusion_criteria', []))} inclusion + "
              f"{len(parsed.get('exclusion_criteria', []))} exclusion criteria for {trial_id}")
        return parsed

    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parse failed for {trial_id}: {e}")
        return {"inclusion_criteria": [], "exclusion_criteria": []}
    except Exception as e:
        print(f"[ERROR] LLM parsing failed for {trial_id}: {e}")
        return {"inclusion_criteria": [], "exclusion_criteria": []}


def parse_trials_batch(trials: list) -> list:
    enriched = []
    for trial in trials:
        raw_criteria = trial.get("eligibility_criteria_raw", "")
        parsed = parse_eligibility_with_llm(raw_criteria, trial.get("nct_id", ""))
        trial["parsed_eligibility"] = parsed
        enriched.append(trial)
    return enriched


# ── TEST ──

if __name__ == "__main__":
    print("\n" + "="*50)
    print("TEST 1: Fetching Alzheimer Trials")
    print("="*50)

    trials = fetch_trials("Alzheimer's Disease", max_results=3)

    for i, trial in enumerate(trials):
        print(f"\n[Trial {i+1}] {trial['nct_id']}")
        print(f"  Title: {trial['title'][:80]}")
        print(f"  Phase: {trial['phase']}")
        print(f"  Age: {trial['min_age']} - {trial['max_age']}")
        print(f"  Locations: {trial['locations'][:2]}")

    print("\n" + "="*50)
    print("TEST 2: Parsing Eligibility Criteria")
    print("="*50)

    if trials:
        first_trial = trials[0]
        print(f"\nParsing criteria for: {first_trial['nct_id']}")
        parsed = parse_eligibility_with_llm(
            first_trial["eligibility_criteria_raw"],
            first_trial["nct_id"]
        )

        print(f"\nInclusion Criteria ({len(parsed.get('inclusion_criteria', []))}):")
        for c in parsed.get("inclusion_criteria", [])[:3]:
            print(f"  + [{c['type']}] {c['criterion'][:80]}")

        print(f"\nExclusion Criteria ({len(parsed.get('exclusion_criteria', []))}):")
        for c in parsed.get("exclusion_criteria", [])[:3]:
            print(f"  - [{c['type']}] {c['criterion'][:80]}")