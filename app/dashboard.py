import streamlit as st
import requests
import json
import sys
import os
import io
import urllib.parse

# Add the parent directory (project root) to sys.path so 'src' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pdf_generator import generate_pdf_report

@st.cache_data
def get_pdf_data(pdf_data):
    buffer = io.BytesIO()
    generate_pdf_report(pdf_data, buffer)
    return buffer.getvalue()

# Set up page config
st.set_page_config(
    page_title="NeuroMatch Lite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium look
st.markdown("""
<style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .main-header {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        background: -webkit-linear-gradient(45deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #1e293b;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    .match-score {
        font-size: 2rem;
        font-weight: bold;
        color: #10b981;
    }
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# App Layout
st.markdown('<h1 class="main-header">🧠 NeuroMatch Lite</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Clinical Trial Matching for Neurological Symptoms</p>', unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000"

with st.sidebar:
    st.header("Patient Profile")
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    gender = st.selectbox("Gender", ["Female", "Male", "Other", "Prefer not to say"])
    user_type = st.radio("I am a:", ["Patient", "Healthcare Professional"])
    
    st.subheader("Medical History")
    history_input = st.text_area("List any prior conditions (comma separated)", "High blood pressure")
    
    st.markdown("---")
    st.info("Your data is processed locally and securely matched using AI semantic search.")

st.subheader("Describe Your Symptoms")
symptoms = st.text_area(
    "Please describe your neurological symptoms in your own words:",
    "I've been experiencing persistent tremors in my right hand, stiffness in my legs when walking, and a general slowing down of my movements over the past six months.",
    height=150
)

if st.button("🔍 Find Matching Clinical Trials"):
    with st.spinner("Analyzing profile and searching clinical trials..."):
        # Prepare payload
        symptoms_list = [s.strip() for s in symptoms.split(",") if s.strip()]
        medical_history = [item.strip() for item in history_input.split(",") if item.strip()]
        payload = {
            "age": age,
            "gender": gender.lower(),
            "symptoms": symptoms_list,
            "existing_conditions": medical_history,
            "audience": user_type.lower()
        }
        
        try:
            # Call FastAPI Backend
            response = requests.post(f"{API_URL}/analyze", json=payload)
            response.raise_for_status()
            data = response.json()
            
            st.success("Analysis Complete!")
            
            # Display Results
            st.markdown(f"### Diagnosis: {data.get('diagnosis_text', 'Analysis Complete')}")
            st.markdown("---")
            
            if not data.get('top_matches'):
                st.warning("No highly relevant trials found at this time.")
            else:
                for match in data['top_matches']:
                    with st.container():
                        st.markdown(f'<div class="card">', unsafe_allow_html=True)
                        cols = st.columns([3, 1])
                        with cols[0]:
                            st.markdown(f"### {match.get('trial_title', 'N/A')}")
                            st.markdown(f"**Trial ID:** `{match.get('trial_id', 'N/A')}`")
                            st.markdown(f"📅 **Trial Timeline:** `{match.get('start_date', 'N/A')}` — `{match.get('completion_date', 'N/A')}`")
                        with cols[1]:
                            confidence = match.get('confidence', 0)
                            # Handle confidence being either 0-1 or 0-100
                            score = int(confidence * 100) if confidence <= 1 else int(confidence)
                            st.markdown(f"<div style='text-align: right;'><span class='match-score'>{score}%</span><br>Match Score</div>", unsafe_allow_html=True)
                        
                        if user_type == "Patient":
                            st.info(f"**Summary:** {match.get('summary_patient', 'No summary available.')}")
                        else:
                            st.warning(f"**Clinician Insight:** {match.get('summary_clinician', 'No technical summary available.')}")
                        
                        # Strengths & Gaps
                        st.markdown("#### Eligibility Analysis")
                        
                        if match.get('strengths'):
                            st.markdown("**✅ Strengths:**")
                            for strength in match['strengths']:
                                st.markdown(f"- {strength}")
                        
                        if match.get('gaps'):
                            st.markdown("**⚠️ Potential Gaps / Requirements:**")
                            for gap in match['gaps']:
                                st.markdown(f"- **{gap['criterion']}**: {gap['explanation']} (*Action: {gap['action']}*)")
                        
                        # Next Steps
                        with st.expander("Recommended Next Steps"):
                            st.markdown(f"**Overall Status:** `{match.get('overall_status', 'N/A').replace('_', ' ').title()}`")
                            st.markdown("---")
                            st.markdown("- Consult with your neurologist about this specific trial.")
                            st.markdown("- Review the official listing on ClinicalTrials.gov.")

                        # PDF Download & Official Link
                        confidence = match.get('confidence', 0)
                        score_float = confidence if confidence <= 1 else confidence / 100
                        params = {
                            "trial_title": match.get('trial_title', 'Trial'),
                            "match_score": score_float,
                            "summary": match.get('summary_patient', '') if user_type == "Patient" else match.get('summary_clinician', ''),
                            "patient_summary": data.get('patient_summary', 'Analysis Complete')
                        }
                        export_url = f"{API_URL}/export?{urllib.parse.urlencode(params)}"
                        
                        btn_cols = st.columns(2)
                        with btn_cols[0]:
                            st.markdown(f'<a href="{export_url}" target="_blank" class="download-btn">📥 Download Clinical Report (PDF)</a>', unsafe_allow_html=True)
                        with btn_cols[1]:
                            official_url = f"https://clinicaltrials.gov/study/{match.get('trial_id')}"
                            st.markdown(f'<a href="{official_url}" target="_blank" class="download-btn" style="background: #4A5568;">🔗 View Official Listing</a>', unsafe_allow_html=True)
                                
                        st.markdown('</div>', unsafe_allow_html=True)
                        
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Cannot connect to the backend server. Make sure your FastAPI server is running!")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
