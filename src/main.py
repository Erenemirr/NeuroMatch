from fastapi import FastAPI, HTTPException
from .schemas import PatientProfile, AnalysisResponse, TrialResult
from .embedder import SymptomEmbedder
from .matcher import MatchingEngine
from .pdf_generator import generate_pdf_report
from fastapi.responses import FileResponse
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="NeuroMatch Lite API")

import pandas as pd

# Initialize components
embedder = SymptomEmbedder()
matcher = MatchingEngine()

# Load real dataset from Eren's repo
try:
    df = pd.read_csv("https://raw.githubusercontent.com/Erenemirr/NeuroMatch/main/data/datasets/symptom_disease/symptom_Description.csv")
    df = df.dropna()
    REAL_TRIALS = [{"id": f"DIS-{i}", "title": row["Disease"], "description": row["Description"]} for i, row in df.iterrows()]
except Exception as e:
    print(f"Error loading dataset: {e}")
    REAL_TRIALS = [
        {"id": "NCT01", "title": "Multiple Sclerosis Study", "description": "Relapsing-remitting MS with visual impairment"},
        {"id": "NCT02", "title": "Parkinson's Early Stage", "description": "Recent tremors and motor difficulty in adults"},
    ]

@app.get("/")
async def root():
    return {"message": "Welcome to NeuroMatch Lite API", "status": "active"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_patient(profile: PatientProfile):
    """
    Main endpoint to analyze patient symptoms and match with trials.
    """
    try:
        # 1. Get embedding for symptoms using real data
        matches = embedder.find_matches(profile.symptoms, REAL_TRIALS)
        
        # 2. Run Matching Engine for eligibility
        results = []
        for match in matches:
            trial_data = match['profile']
            match_score = match['score']
            
            # This is where we take Eren's parser output (placeholder for now)
            criteria_results = matcher.evaluate_criteria(profile, trial_data)
            
            results.append(TrialResult(
                trial_id=trial_data['id'],
                title=trial_data['title'],
                match_score=match_score,
                criteria_status=criteria_results['checks'],
                summary=criteria_results['summary'],
                next_steps=criteria_results['next_steps']
            ))

        return AnalysisResponse(
            patient_summary=f"Analysis for {profile.age}yo patient focusing on {profile.symptoms[:50]}...",
            matches=results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/export")
async def export_report(trial_title: str, match_score: float, summary: str, patient_summary: str):
    """
    Dynamically generates a PDF for a specific trial match and returns it.
    """
    # Reconstruct the match data for the generator
    pdf_data = {
        "patient_summary": patient_summary,
        "matches": [
            {
                "title": trial_title,
                "match_score": match_score,
                "summary": summary,
                "criteria_status": [], # Simplified for the export link
                "next_steps": ["Consult with your doctor", "Review ClinicalTrials.gov"]
            }
        ]
    }
    
    # Create a temporary file
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd) # Close immediately so ReportLab can open and write to it on Windows
    try:
        generate_pdf_report(pdf_data, path)
        return FileResponse(path, filename=f"NeuroMatch_{trial_title.replace(' ', '_')}.pdf", media_type="application/pdf")
    except Exception as e:
        print(f"PDF Generation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
