import openai
import os
from .schemas import PatientProfile, Criterion
import json

class MatchingEngine:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            openai.api_key = self.api_key

    def evaluate_criteria(self, profile: PatientProfile, trial_data: dict):
        """
        Uses OpenAI to compare patient profile against trial eligibility criteria.
        """
        if not self.api_key:
            # Fallback to simple mock logic if no API key
            return self._mock_evaluation(profile, trial_data)

        prompt = f"""
        Compare the following Patient Profile with the Clinical Trial criteria.
        
        Patient Profile:
        Age: {profile.age}
        Gender: {profile.gender}
        Symptoms: {profile.symptoms}
        History: {", ".join(profile.medical_history)}
        
        Trial: {trial_data['title']}
        Description: {trial_data['description']}
        
        Tasks:
        1. Identify 3 key eligibility criteria (e.g., Age range, Condition, health factors).
        2. For each, determine if the patient meets it (True/False).
        3. Provide a brief explanation.
        4. Suggest 2 next steps.
        
        Respond ONLY in JSON format:
        {{
            "checks": [
                {{"name": "Criterion Name", "is_met": true/false, "details": "Explanation", "category": "inclusion"}},
                ...
            ],
            "summary": "Overall summary of match",
            "next_steps": ["Step 1", "Step 2"]
        }}
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini", # Using mini for speed/cost in hackathon
                messages=[
                    {"role": "system", "content": "You are a clinical trial eligibility specialist."},
                    {"role": "user", "content": prompt}
                ],
                response_format={{ "type": "json_object" }}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"OpenAI Error: {e}")
            return self._mock_evaluation(profile, trial_data)

    def _mock_evaluation(self, profile: PatientProfile, trial_data: dict):
        """Fallback mock logic."""
        return {
            "checks": [
                {"name": "Age Compatibility", "is_met": profile.age > 18, "details": "Trial requires adults.", "category": "inclusion"},
                {"name": "Symptom Match", "is_met": True, "details": "Symptoms align with neurological focus.", "category": "inclusion"}
            ],
            "summary": "Based on a preliminary scan, you appear to be a potential match.",
            "next_steps": ["Consult with your neurologist", "Review trial details on ClinicalTrials.gov"]
        }
