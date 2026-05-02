# 🧠 NeuroMatch

> AI-powered neurological clinical trial matching — Sabancı NeuroBridge AI Hackathon 2026

NeuroMatch helps patients with neurological conditions find relevant clinical trials they may be eligible for. It takes a patient's symptoms, generates a confidence-scored diagnosis, fetches real recruiting trials from ClinicalTrials.gov, and explains eligibility gaps in plain language — adapted for 4 different audiences.

---

## How It Works

1. **Patient input** — The user describes their symptoms via free text or uploads a medical PDF report.
2. **Hybrid diagnosis** — Symptoms are vectorized with `sentence-transformers` and matched against 12 neurological disease profiles via cosine similarity. Scores are passed to an LLM (Groq / LLaMA 3.3-70B) which produces confidence-scored diagnoses.
3. **OASIS-2 enrichment** — The patient profile is matched against 371 real patient records from the OASIS-2 longitudinal MRI dataset to estimate MMSE and CDR score ranges.
4. **Trial fetching** — ClinicalTrials.gov API v2 is queried in real time for actively recruiting trials matching the top diagnoses.
5. **Eligibility parsing** — Each trial's eligibility criteria text is parsed into structured JSON by the LLM. Results are cached by NCT ID to minimize token usage.
6. **Gap analysis** — A two-stage analysis (rule-based + LLM) determines whether the patient is `likely_eligible`, `needs_more_info`, or `likely_ineligible` for each trial — with confidence scores and actionable next steps.
7. **Audience adaptation** — Results are reformatted for 4 audiences: Patient, Neurologist, General Practitioner, Researcher.
8. **PDF report** — A downloadable professional PDF report is generated via ReportLab, including the diagnosis table, OASIS-2 reference data, and trial match cards.

---

## Team

| Name | Role |
|------|------|
| **Eren** | ML Pipeline — LLM-based diagnosis with confidence scoring, ClinicalTrials.gov API integration, eligibility parser & cache, validation layer, gap explainer, OASIS-2 integration, MRI report parser, PDF report generator, API integration |
| **Sait** | Backend — FastAPI setup, symptom embedding (sentence-transformers, cosine similarity), matching engine, PDF generation, pipeline–backend integration |
| **Beyza** | Frontend + Audience Adapter + Presentation — UI, symptom form, report upload, result cards, audience dropdown (4 types), audience prompt writing, demo scenarios, presentation slides |

---

## Project Structure

```
neuromatch/
├── api.py                  # FastAPI endpoints (/analyze, /report, /conditions)
├── neuromatch-lite.html    # Frontend — single-file web UI
├── ml/
│   ├── pipeline.py         # Main orchestrator — runs all steps in sequence
│   ├── diagnosis.py        # Hybrid diagnosis (embedding + LLM)
│   ├── parser.py           # ClinicalTrials.gov API + eligibility LLM parser + cache
│   ├── validator.py        # Validates parsed eligibility criteria
│   ├── gap_explainer.py    # Rule-based + LLM eligibility gap analysis
│   ├── audience_adapter.py # Formats output for 4 audience types
│   ├── oasis_estimator.py  # OASIS-2 dataset integration
│   ├── report_parser.py    # Extracts clinical data from uploaded PDF reports
│   └── report_generator.py # Generates downloadable PDF reports (ReportLab)
└── data/
    ├── datasets/           # Kaggle neurological disease datasets + OASIS-2
    └── patient_schema.json # Patient profile schema
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Erenemirr/NeuroMatch.git
cd NeuroMatch
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your Groq API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

### 5. Run the backend

```bash
python api.py
```

The API will be available at `http://localhost:8000`.

### 6. Open the frontend

Open `neuromatch-lite.html` in your browser. Click **"Load demo case"** to try a pre-filled patient profile, then click **"Analyze & Match Trials"**.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Service status |
| `POST` | `/analyze` | Run full pipeline, return JSON results |
| `POST` | `/report` | Run full pipeline, return downloadable PDF |
| `GET` | `/conditions` | List of supported neurological conditions |

### Example request

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "age": 70,
    "gender": "F",
    "symptoms": ["memory loss", "confusion", "behavioral changes"],
    "existing_conditions": ["Alzheimer'\''s Disease"],
    "audience": "patient"
  }'
```

---

## Supported Conditions

Alzheimer's Disease · Parkinson's Disease · Multiple Sclerosis · Epilepsy · ALS · Dementia · Huntington's Disease · Migraine · Neuropathy

---

## Key Technical Features

- **Hybrid diagnosis** — sentence-transformers cosine similarity + LLM, not just keyword matching
- **Real trial data** — live ClinicalTrials.gov API v2, recruiting trials only
- **OASIS-2 integration** — real clinical reference data for MMSE/CDR estimation
- **Eligibility cache** — NCT ID-based JSON cache eliminates redundant LLM calls
- **Smart truncation** — preserves both inclusion and exclusion sections within token limits
- **MRI report parsing** — extracts age, symptoms, MMSE, CDR from uploaded PDF reports
- **4-audience output** — same pipeline, different language and depth per audience

---

## Data Sources

- [ClinicalTrials.gov](https://clinicaltrials.gov) — real-time recruiting trial data
- [OASIS-2](https://www.oasis-brains.org/) — Open Access Series of Imaging Studies (Marcus et al., 2010)
- Kaggle neurological disease symptom datasets

---

## Disclaimer

NeuroMatch is not a medical diagnostic tool. Results are AI-generated and for informational purposes only. Always consult a qualified neurologist before making any medical decisions or enrolling in a clinical trial.
