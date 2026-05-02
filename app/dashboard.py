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
        medical_history = [item.strip() for item in history_input.split(",") if item.strip()]
        payload = {
            "age": age,
            "gender": gender.lower(),
            "symptoms": symptoms,
            "medical_history": medical_history,
            "user_type": user_type.lower()
        }
        
        try:
            # Call FastAPI Backend
            response = requests.post(f"{API_URL}/analyze", json=payload)
            response.raise_for_status()
            data = response.json()
            
            st.success("Analysis Complete!")
            
            # Display Results
            st.markdown(f"### {data.get('patient_summary')}")
            st.markdown("---")
            
            if not data.get('matches'):
                st.warning("No highly relevant trials found at this time.")
            else:
                for match in data['matches']:
                    with st.container():
                        st.markdown(f'<div class="card">', unsafe_allow_html=True)
                        cols = st.columns([3, 1])
                        with cols[0]:
                            st.markdown(f"### {match['title']}")
                            st.markdown(f"**Trial ID:** `{match['trial_id']}`")
                        with cols[1]:
                            st.markdown(f"<div style='text-align: right;'><span class='match-score'>{int(match['match_score'] * 100)}%</span><br>Match Score</div>", unsafe_allow_html=True)
                        
                        st.markdown(match['summary'])
                        
                        # Eligibility Criteria
                        st.markdown("#### Eligibility Assessment")
                        for criteria in match['criteria_status']:
                            icon = "✅" if criteria['is_met'] else "❌"
                            color = "green" if criteria['is_met'] else "red"
                            st.markdown(f"- {icon} **{criteria['name']}**: {criteria['details']}")
                        
                        # Next Steps
                        with st.expander("Recommended Next Steps"):
                            for step in match['next_steps']:
                                st.markdown(f"- {step}")

                        # PDF Download Link (Prevents IDM auto-trigger)
                        params = {
                            "trial_title": match['title'],
                            "match_score": match['match_score'],
                            "summary": match['summary'],
                            "patient_summary": data.get('patient_summary')
                        }
                        export_url = f"{API_URL}/export?{urllib.parse.urlencode(params)}"
                        
                        st.link_button("📄 Download PDF Report", export_url)
                                
                        st.markdown('</div>', unsafe_allow_html=True)
                        
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Cannot connect to the backend server. Make sure your FastAPI server is running!")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
