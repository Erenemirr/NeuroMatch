"""
NeuroMatch — API Layer
FastAPI endpoints that connect the ML pipeline to the frontend.
Replaces the placeholder logic in the backend's main.py.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import sys
import os
import tempfile

# Add ml directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ml"))

from pipeline import run_neuromatch, format_pipeline_output
try:
    from report_generator import generate_neuromatch_report
except ImportError:
    # Fallback to the old generator if teammate's file is missing or in different place
    from src.pdf_generator import generate_pdf_report as generate_neuromatch_report

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
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    # Extract all matches — frontend handles filtering/display
    gap_analysis = result.get("gap_analysis", [])
    top_matches = []
    for gap in gap_analysis:
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


@app.post("/report")
async def generate_report(request: PatientRequest):
    """
    Runs the full pipeline and returns a downloadable PDF report.
    """
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
        # Must await the async run_neuromatch
        result = await run_neuromatch(patient_profile, audience=request.audience)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    # Generate PDF to a temp file
    try:
        fd, pdf_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        generate_neuromatch_report(result, pdf_path, audience=request.audience)
        patient_id = request.patient_id or "anonymous"
        filename = f"NeuroMatch_Report_{patient_id}.pdf"
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation error: {str(e)}")


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
    # Using 127.0.0.1 for better compatibility on Windows local machines
    # reload=False because heavy model loading during startup can interfere with the reloader's handshake
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
