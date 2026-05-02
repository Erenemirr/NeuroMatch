"""
NeuroMatch — Layer 3: ClinicalTrials.gov API + LLM Eligibility Parser
Fetches real neurological trials and parses eligibility criteria into structured JSON.
"""

import os
import json
import requests
import asyncio
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()
client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

# ── CACHES ──
TRIAL_CACHE = {}

# Persistent Cache for Eligibility
_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".eligibility_cache.json")

def _load_cache() -> dict:
    try:
        if os.path.exists(_CACHE_PATH):
            with open(_CACHE_PATH, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_cache(cache: dict):
    try:
        with open(_CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass

ELIGIBILITY_CACHE = _load_cache()

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

async def fetch_trials(condition: str, location: str = None, max_results: int = 10) -> list:
    cache_key = f"{condition}_{location}_{max_results}"
    if cache_key in TRIAL_CACHE:
        print(f"[INFO] Using cached trials for: {condition}")
        return TRIAL_CACHE[cache_key]

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
        def fetch_sync():
            return requests.get(CLINICALTRIALS_BASE, params=params, timeout=15.0)
        
        response = await asyncio.to_thread(fetch_sync)
        response.raise_for_status()
        data = response.json()
        studies = data.get("studies", [])
        print(f"[INFO] Found {len(studies)} trials for {condition}")
        results = [parse_trial_summary(s) for s in studies]
        TRIAL_CACHE[cache_key] = results
        return results
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
        "start_date": status_module.get("startDateStruct", {}).get("date", "N/A"),
        "completion_date": status_module.get("completionDateStruct", {}).get("date", "N/A"),
        "eligibility_criteria_raw": eligibility_module.get("eligibilityCriteria", ""),
        "min_age": eligibility_module.get("minimumAge", "N/A"),
        "max_age": eligibility_module.get("maximumAge", "N/A"),
        "gender": eligibility_module.get("sex", "ALL"),
        "brief_summary": description_module.get("briefSummary", "N/A"),
        "locations": locations[:5],
    }


async def fetch_trials_for_diagnoses(diagnoses: list, max_per_condition: int = 3) -> list:
    all_trials = []
    seen_ids = set()

    async def fetch_and_tag(diagnosis):
        condition = diagnosis.get("disease", "")
        if not condition:
            return []
        trials = await fetch_trials(condition, max_results=max_per_condition)
        for trial in trials:
            trial["matched_condition"] = condition
        return trials

    tasks = [fetch_and_tag(d) for d in diagnoses[:2]]
    results = await asyncio.gather(*tasks)

    for trials in results:
        for trial in trials:
            if trial["nct_id"] not in seen_ids:
                seen_ids.add(trial["nct_id"])
                all_trials.append(trial)

    return all_trials


# ── SECTION 2: LLM Eligibility Parser ──

def _smart_truncate(criteria_text: str, max_chars: int = 2000) -> str:
    """
    Balanced truncation: keeps parts of both inclusion and exclusion sections.
    """
    if len(criteria_text) <= max_chars:
        return criteria_text

    text_lower = criteria_text.lower()
    excl_markers = ["exclusion criteria", "exclusion:", "exclude:", "ineligible if"]
    excl_idx = -1
    for marker in excl_markers:
        idx = text_lower.find(marker)
        if idx > 0:
            excl_idx = idx
            break

    if excl_idx > 0:
        half = max_chars // 2
        inclusion_part = criteria_text[:excl_idx][:half]
        exclusion_part = criteria_text[excl_idx:][:half]
        return inclusion_part + "\n" + exclusion_part
    else:
        return criteria_text[:max_chars]


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


async def parse_eligibility_with_llm(criteria_text: str, trial_id: str = "") -> dict:
    if not criteria_text or len(criteria_text.strip()) < 20:
        return {"inclusion_criteria": [], "exclusion_criteria": []}

    # Persistent Cache Check
    if trial_id in ELIGIBILITY_CACHE:
        print(f"[CACHE] Using cached parsed criteria for {trial_id}")
        return ELIGIBILITY_CACHE[trial_id]

    # Use smaller, high-quota model for bulk structured extraction
    truncated_text = _smart_truncate(criteria_text, max_chars=2000)

    try:
        # Retry loop for 429 errors
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a clinical trial expert. Always respond with valid JSON only. No markdown, no explanation."
                        },
                        {
                            "role": "user",
                            "content": ELIGIBILITY_PARSE_PROMPT.format(
                                criteria_text=truncated_text
                            )
                        }
                    ],
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    print(f"[WARNING] Rate limit hit. Retrying in 5s... (Attempt {attempt+1})")
                    await asyncio.sleep(5)
                else:
                    raise e

        raw = response.choices[0].message.content
        parsed = json.loads(raw)
        print(f"[INFO] Parsed {len(parsed.get('inclusion_criteria', []))} inclusion + "
              f"{len(parsed.get('exclusion_criteria', []))} exclusion criteria for {trial_id}")
        
        # Save to persistent cache
        ELIGIBILITY_CACHE[trial_id] = parsed
        _save_cache(ELIGIBILITY_CACHE)
        
        return parsed

    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse LLM JSON for {trial_id}: {e}")
        return {"inclusion_criteria": [], "exclusion_criteria": [], "error": "JSON parse error"}
    except Exception as e:
        print(f"[ERROR] LLM parsing failed for {trial_id}: {e}")
        return {"inclusion_criteria": [], "exclusion_criteria": [], "error": str(e)}


async def parse_trials_batch(trials: list) -> list:
    semaphore = asyncio.Semaphore(2)
    
    async def limited_parse(trial):
        async with semaphore:
            trial_id = trial.get("nct_id", "")
            raw_text = trial.get("eligibility_criteria_raw", "")
            parsed = await parse_eligibility_with_llm(raw_text, trial_id)
            trial["eligibility_parsed"] = parsed
            return trial

    tasks = [limited_parse(t) for t in trials]
    return await asyncio.gather(*tasks)