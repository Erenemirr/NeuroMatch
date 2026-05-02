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
    symptoms: List[str]
    duration: Optional[str] = ""
    existing_conditions: List[str] = []
    medications: List[str] = []
    user_type: str = "patient"

class TrialResult(BaseModel):
    trial_id: str
    title: str
    matchScore: int
    phase: str = "N/A"
    criteria_status: List[Criterion]
    summary: str
    rationale: str
    patientBenefit: str
    doctorInsight: str
    next_steps: List[str]

class AnalysisResponse(BaseModel):
    patient_summary: str
    matches: List[TrialResult]
