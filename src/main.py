from fastapi import FastAPI, HTTPException
from .schemas import PatientProfile, AnalysisResponse, TrialResult
from .embedder import SymptomEmbedder
from .matcher import MatchingEngine
from .pdf_generator import generate_pdf_report
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="NeuroMatch Lite API")

# Initialize components
embedder = SymptomEmbedder()
matcher = MatchingEngine()

@app.get("/")
async def root():
    return {"message": "Welcome to NeuroMatch Lite API", "status": "active"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_patient(profile: PatientProfile):
    """
    Main endpoint to analyze patient symptoms and match with trials.
    """
    try:
        # 1. Get embedding for symptoms
        # (For now, using a placeholder list of trials/diseases)
        placeholder_trials = [
            {"id": "NCT01", "title": "Multiple Sclerosis Study", "description": "Relapsing-remitting MS with visual impairment"},
            {"id": "NCT02", "title": "Parkinson's Early Stage", "description": "Recent tremors and motor difficulty in adults"},
        ]
        
        matches = embedder.find_matches(profile.symptoms, placeholder_trials)
        
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

@app.get("/export/{trial_id}")
async def export_report(trial_id: str):
    # Placeholder for PDF generation trigger
    return {"message": f"Generating PDF for {trial_id}..."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
