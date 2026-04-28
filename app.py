import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Activity Predictor | AI Studio",
    page_icon="✨",
    layout="centered"
)

# --- Premium Aesthetics (Tailwind-Inspired) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    * { 
        font-family: 'Outfit', sans-serif; 
    }
    
    /* Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Hide top header line */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Main Glassmorphic Card */
    .main-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px;
        padding: 40px;
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.05);
    }
    
    /* Typography */
    h1 {
        font-weight: 700 !important;
        background: linear-gradient(to right, #a855f7, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem !important;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Inputs Styling */
    .stNumberInput > label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    
    div[data-baseweb="input"] {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }
    div[data-baseweb="input"] input {
        color: #f8fafc !important;
    }
    
    /* Vibrant Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        padding: 16px;
        border-radius: 14px;
        font-weight: 600;
        font-size: 1.1rem;
        letter-spacing: 0.05em;
        transition: all 0.3s ease;
        text-transform: uppercase;
        margin-top: 10px;
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 25px -5px rgba(99, 102, 241, 0.5);
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
    }
    
    /* Error Card */
    .error-card {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.2);
        border-radius: 16px;
        padding: 24px;
        color: #fca5a5;
        display: flex;
        align-items: flex-start;
        gap: 16px;
        backdrop-filter: blur(10px);
        margin-bottom: 24px;
    }
    .error-icon {
        font-size: 24px;
    }
    .error-content h3 {
        color: #f87171;
        margin: 0 0 8px 0;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .error-content p {
        margin: 0 0 12px 0;
        line-height: 1.5;
        font-size: 0.95rem;
    }
    .error-code {
        background: rgba(0, 0, 0, 0.3);
        padding: 8px 12px;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.85rem;
        color: #e2e8f0;
    }
    
    /* Result Section */
    .prediction-title {
        font-size: 1rem;
        color: #94a3b8;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-top: 24px;
        margin-bottom: 8px;
        font-weight: 500;
    }
    
    .prediction-value {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(to right, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 16px;
        line-height: 1.1;
    }
    
    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
        margin: 2rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants & Asset Loading ---
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

@st.cache_resource
def load_model():
    assets = {}
    files = ['kmeans.pkl', 'scaler.pkl', 'feature_names.pkl', 'cluster_activity_mapping.pkl', 'important_features.pkl']
    try:
        for f in files:
            with open(os.path.join(MODEL_DIR, f), 'rb') as file:
                assets[f.split('.')[0]] = pickle.load(file)
        return assets
    except:
        return None

assets = load_model()

# --- Simplified Feature Mapping ---
FEATURE_MAP = {
    'tBodyAcc-mean()-X': 'Horizontal Sway',
    'tBodyAcc-mean()-Z': 'Vertical Bounce',
    'tGravityAcc-mean()-X': 'Side Leaning (Lying)',
    'tGravityAcc-mean()-Y': 'Forward/Back Tilt',
    'tBodyGyro-mean()-X': 'Rolling Speed',
    'tBodyGyro-mean()-Z': 'Spinning Speed',
    'tBodyAccMag-mean()': 'Total Energy',
    'tGravityAccMag-mean()': 'Body Weight Feel',
    'fBodyAcc-mean()-X': 'Vibration Speed',
    'fBodyAcc-std()-Y': 'Stability Score'
}

# --- Main App ---
st.title("Activity Predictor ✨")
st.markdown("<div class='subtitle'>Advanced ML Analysis of Smartphone Sensor Data</div>", unsafe_allow_html=True)

if not assets:
    st.markdown("""
    <div class="error-card">
        <div class="error-icon">⚠️</div>
        <div class="error-content">
            <h3>Models Not Found</h3>
            <p>The pre-trained machine learning models are missing from the <code>models/</code> directory. To fix this on Windows, open your terminal/command prompt and run:</p>
            <div class="error-code">python analysis.py</div>
            <p style="margin-top: 12px; font-size: 0.85rem; color: #94a3b8;">Make sure your dataset is inside the <code>data/</code> folder before running the analysis script.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    with st.form("activity_form"):
        col1, col2 = st.columns(2)
        user_inputs = {}
        
        # We only use the 10 important features as inputs for simplicity
        # The rest will be set to 0 (neutral) for the full model prediction
        important_features = assets['important_features']
        
        for i, feat in enumerate(important_features):
            with col1 if i % 2 == 0 else col2:
                simple_name = FEATURE_MAP.get(feat, feat.replace('-', ' ').title())
                user_inputs[feat] = st.number_input(
                    simple_name, 
                    value=0.0, 
                    format="%.4f",
                    help=f"Original Technical Key: {feat}"
                )
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Analyze Movement Data")
        
        if submitted:
            # Prepare full feature vector (561 features)
            full_input = np.zeros((1, len(assets['feature_names'])))
            for feat, val in user_inputs.items():
                if feat in assets['feature_names']:
                    idx = assets['feature_names'].index(feat)
                    full_input[0, idx] = val
            
            # Predict
            scaled_input = assets['scaler'].transform(full_input)
            prediction_cluster = assets['kmeans'].predict(scaled_input)[0]
            activity_label = assets['cluster_activity_mapping'].get(prediction_cluster, f"Cluster {prediction_cluster}")
            
            # Success Animation
            time.sleep(0.5)
            
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(f'<div class="prediction-title">Detected Activity</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="prediction-value">{activity_label}</div>', unsafe_allow_html=True)
            
            if "WALKING" in activity_label.upper():
                st.info("💡 Insight: High frequency movement detected. This matches active cardio profiles.")
            elif "SITTING" in activity_label.upper() or "LAYING" in activity_label.upper():
                st.info("💡 Insight: Low impact/static position detected. This matches sedentary profiles.")
            else:
                st.info("💡 Insight: Moderate activity detected.")
            
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br><p style='text-align: center; color: #475569; font-size: 0.85rem; margin-top: 2rem;'>Powered by K-Means Clustering & Scikit-Learn</p>", unsafe_allow_html=True)
