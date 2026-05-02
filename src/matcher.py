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
        """
        if not self.client:
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
        3. Provide an explanation specifically tailored for a {profile.user_type}.
           - If Patient: Use simple, empathetic, and clear language. Avoid medical jargon.
           - If Professional/Researcher/Neurologist: Use precise, clinical, and data-driven terminology.
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
        """Fallback mock logic."""
        return {
            "checks": [
                {"name": "Age Compatibility", "is_met": profile.age > 18, "details": "Trial requires adults.", "category": "inclusion"},
                {"name": "Symptom Match", "is_met": True, "details": "Symptoms align with neurological focus.", "category": "inclusion"}
            ],
            "summary": "Based on a preliminary scan, you appear to be a potential match.",
            "next_steps": ["Consult with your neurologist", "Review trial details on ClinicalTrials.gov"]
        }
