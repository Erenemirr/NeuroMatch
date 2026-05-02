from openai import OpenAI
import os
from .schemas import PatientProfile, Criterion
import json

class MatchingEngine:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if self.api_key:
            self.client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=self.api_key
            )
            self.model = "llama-3.3-70b-versatile"
        else:
            self.client = None
            self.model = None

    def evaluate_criteria(self, profile: PatientProfile, trial_data: dict):
        """
        Uses Groq (Llama 3) to compare patient profile against trial eligibility criteria.
        Outputs data aligned with the frontend's expected schema.
        """
        if not self.client:
            # Fallback to simple mock logic if no API key
            return self._mock_evaluation(profile, trial_data)

        prompt = f"""
        Compare the following Patient Profile with the Clinical Trial criteria.
        
        Patient Profile:
        Age: {profile.age}
        Gender: {profile.gender}
        Symptoms: {", ".join(profile.symptoms)}
        Duration: {profile.duration}
        Existing Conditions: {", ".join(profile.existing_conditions)}
        Medications: {", ".join(profile.medications)}
        
        Trial: {trial_data['title']}
        Description: {trial_data['description']}
        
        Tasks:
        1. Identify 3 key eligibility criteria.
        2. For each, determine if the patient meets it (True/False).
        3. Match Score: Provide a percentage (0-100) representing how well the patient fits.
        4. Provide the Trial Phase (e.g., Phase I, Phase II, Phase III).
        5. Rationale: Technical reason for the match.
        6. Patient Benefit: Simple, empathetic explanation of why this trial might help them.
        7. Doctor Insight: Precise clinical explanation of the trial's significance for this patient.
        8. Suggest 2 next steps.
        
        Respond ONLY in JSON format:
        {{
            "checks": [
                {{"name": "Criterion Name", "is_met": true/false, "details": "Explanation", "category": "inclusion"}},
                ...
            ],
            "matchScore": 85,
            "phase": "Phase III",
            "summary": "Overall summary of match",
            "rationale": "Technical explanation for match score",
            "patientBenefit": "Empathetic explanation for the patient",
            "doctorInsight": "Clinical significance for the healthcare professional",
            "next_steps": ["Step 1", "Step 2"]
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a clinical trial eligibility specialist."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" }
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Groq/Llama Error: {e}")
            return self._mock_evaluation(profile, trial_data)

    def _mock_evaluation(self, profile: PatientProfile, trial_data: dict):
        """Fallback mock logic aligned with frontend schema."""
        return {
            "checks": [
                {"name": "Age Compatibility", "is_met": profile.age > 18, "details": "Trial requires adults.", "category": "inclusion"},
                {"name": "Symptom Match", "is_met": True, "details": "Symptoms align with neurological focus.", "category": "inclusion"}
            ],
            "matchScore": 75,
            "phase": "Phase II",
            "summary": "Potential match found.",
            "rationale": "Matches basic criteria.",
            "patientBenefit": "This trial could help manage your current symptoms.",
            "doctorInsight": "Patient meets primary inclusion criteria for this neuro-focused study.",
            "next_steps": ["Consult with your neurologist", "Review trial details"]
        }
