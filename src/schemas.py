from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class Criterion(BaseModel):
    name: str
    is_met: bool = False
    details: str = ""
    category: str = "inclusion" # inclusion or exclusion

class PatientProfile(BaseModel):
    age: int
    gender: str
    symptoms: str
    medical_history: List[str] = []
    user_type: str = "patient" # patient or professional

class TrialResult(BaseModel):
    trial_id: str
    title: str
    match_score: float
    criteria_status: List[Criterion]
    summary: str
    next_steps: List[str]

class AnalysisResponse(BaseModel):
    patient_summary: str
    matches: List[TrialResult]
