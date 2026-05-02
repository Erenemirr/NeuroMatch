"""
NeuroMatch — API Layer
FastAPI endpoints that connect the ML pipeline to the frontend.
Replaces the placeholder logic in the backend's main.py.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

# Add ml directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ml"))

try:
    from pipeline import run_neuromatch, format_pipeline_output
except ImportError as e:
    # Fallback if ml/pipeline.py is not yet created
    error_msg = str(e)
    print(f"IMPORT ERROR: {error_msg}")
    import traceback
    traceback.print_exc()
    async def run_neuromatch(p, audience="patient"): return {"error": f"ML Pipeline error: {error_msg}"}
    def format_pipeline_output(r): return r

from src.pdf_generator import generate_pdf_report
from fastapi.responses import FileResponse
import tempfile

app = FastAPI(
    title="NeuroMatch API",
    description="AI-powered neurological clinical trial matching",
    version="1.0.0"
)

# Allow frontend (Streamlit / React) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REQUEST / RESPONSE SCHEMAS ──

class PatientRequest(BaseModel):
    patient_id: Optional[str] = "anonymous"
    age: int
    gender: str                          # "M" | "F"
    symptoms: List[str]                  # ["memory loss", "confusion"]
    duration: Optional[str] = "Unknown"
    existing_conditions: Optional[List[str]] = []
    medications: Optional[List[str]] = []
    report_text: Optional[str] = None
    audience: Optional[str] = "patient"  # "patient" | "neurologist" | "gp" | "researcher"


class GapItem(BaseModel):
    criterion: str
    type: str
    explanation: str
    action: str


class TrialMatch(BaseModel):
    trial_id: str
    trial_title: str
    matched_condition: str
    overall_status: str        # "likely_eligible" | "needs_more_info" | "likely_ineligible"
    confidence: float
    start_date: str = "N/A"
    completion_date: str = "N/A"
    strengths: List[str]
    gaps: List[GapItem]
    summary_patient: str
    summary_clinician: str


class DiagnosisItem(BaseModel):
    disease: str
    confidence: float


class NeuroMatchResponse(BaseModel):
    patient_id: str
    audience: str
    diagnosis: List[DiagnosisItem]
    diagnosis_text: str
    trials_analyzed: int
    likely_eligible: int
    needs_more_info: int
    likely_ineligible: int
    top_matches: List[TrialMatch]
    disclaimer: str


# ── ENDPOINTS ──

@app.get("/")
async def root():
    return {"message": "NeuroMatch API is running", "status": "active", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/analyze", response_model=NeuroMatchResponse)
async def analyze(request: PatientRequest):
    """
    Main endpoint. Takes patient profile, runs full ML pipeline,
    returns diagnosis + matched trials + gap analysis.
    """
    # Build patient profile dict for pipeline
    patient_profile = {
        "patient_id": request.patient_id,
        "symptoms": request.symptoms,
        "duration": request.duration,
        "age": request.age,
        "gender": request.gender,
        "existing_conditions": request.existing_conditions or [],
        "medications": request.medications or [],
        "report_text": request.report_text,
        "predicted_diagnosis": [],
        "matched_trials": []
    }

    try:
        result = await run_neuromatch(patient_profile, audience=request.audience)
    except Exception as e:
        print(f"PIPELINE CRASH: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    if "error" in result:
        print(f"PIPELINE ERROR: {result['error']}")
        raise HTTPException(status_code=400, detail=result["error"])

    # Extract top 5 matches (exclude likely_ineligible from top display)
    gap_analysis = result.get("gap_analysis", [])
    top_matches = []
    for gap in gap_analysis[:5]:
        llm = gap.get("llm_analysis", {})
        gaps_raw = llm.get("gaps", [])
        gaps_formatted = [
            GapItem(
                criterion=g.get("criterion", ""),
                type=g.get("type", "missing_info"),
                explanation=g.get("explanation", ""),
                action=g.get("action", "")
            )
            for g in gaps_raw
        ]
        top_matches.append(TrialMatch(
            trial_id=gap.get("trial_id", ""),
            trial_title=gap.get("trial_title", ""),
            matched_condition=gap.get("matched_condition", ""),
            overall_status=gap.get("overall_status", "needs_more_info"),
            confidence=gap.get("confidence", 0.0),
            start_date=gap.get("start_date", "N/A"),
            completion_date=gap.get("completion_date", "N/A"),
            strengths=llm.get("strengths", []),
            gaps=gaps_formatted,
            summary_patient=llm.get("summary_patient", ""),
            summary_clinician=llm.get("summary_clinician", "")
        ))

    # Build diagnosis list
    diagnoses = [
        DiagnosisItem(disease=d["disease"], confidence=d["confidence"])
        for d in patient_profile.get("predicted_diagnosis", [])
    ]

    summary = result.get("summary", {})

    return NeuroMatchResponse(
        patient_id=request.patient_id,
        audience=request.audience,
        diagnosis=diagnoses,
        diagnosis_text=result.get("diagnosis_formatted", ""),
        trials_analyzed=result.get("trials_analyzed", 0),
        likely_eligible=summary.get("likely_eligible", 0),
        needs_more_info=summary.get("needs_more_info", 0),
        likely_ineligible=summary.get("likely_ineligible", 0),
        top_matches=top_matches,
        disclaimer="This is not a medical diagnosis. Please consult a neurologist."
    )


@app.get("/export")
async def export_report(
    trial_title: str, 
    match_score: float, 
    summary: str, 
    patient_summary: str
):
    """Generates a PDF report for the patient to share with their doctor."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        generate_pdf_report(
            tmp.name,
            patient_summary,
            trial_title,
            match_score,
            summary
        )
        return FileResponse(
            tmp.name, 
            media_type="application/pdf", 
            filename=f"NeuroMatch_Report_{trial_title[:20]}.pdf"
        )


@app.get("/conditions")
async def get_conditions():
    """Returns the list of neurological conditions NeuroMatch supports."""
    return {
        "conditions": [
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
    }


# ── RUN ──

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
