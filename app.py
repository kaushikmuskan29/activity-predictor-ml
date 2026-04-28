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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { 
        font-family: 'Inter', sans-serif; 
    }
    
    /* Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0a192f 0%, #312e81 50%, #4c1d95 100%);
        color: #f8fafc;
    }
    
    /* Hide top header line */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Main Glassmorphic Card */
    .main-card {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 40px;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 2rem;
    }
    
    /* Typography */
    h1 {
        font-weight: 700 !important;
        background: linear-gradient(to right, #a855f7, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem !important;
        text-align: center;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 300;
        text-align: center;
    }
    
    /* Sections */
    .section-header {
        color: #a5b4fc;
        font-size: 1.25rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(165, 180, 252, 0.2);
        padding-bottom: 0.5rem;
    }
    .section-header:first-of-type {
        margin-top: 0;
    }

    /* Sliders Styling */
    .stSlider label {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    
    /* Vibrant Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #a855f7 0%, #ec4899 100%);
        color: white;
        border: none;
        padding: 16px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        letter-spacing: 0.05em;
        transition: all 0.3s ease;
        text-transform: uppercase;
        margin-top: 2rem;
        box-shadow: 0 10px 15px -3px rgba(236, 72, 153, 0.3), 0 4px 6px -2px rgba(236, 72, 153, 0.15);
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 20px 25px -5px rgba(236, 72, 153, 0.5), 0 10px 10px -5px rgba(236, 72, 153, 0.2);
        background: linear-gradient(135deg, #9333ea 0%, #db2777 100%);
        color: white;
        border-color: transparent !important;
    }
    
    .stButton>button:focus:not(:focus-visible) {
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
    .error-icon { font-size: 24px; }
    .error-content h3 { color: #f87171; margin: 0 0 8px 0; font-size: 1.2rem; font-weight: 600; }
    .error-content p { margin: 0 0 12px 0; line-height: 1.5; font-size: 0.95rem; }
    .error-code {
        background: rgba(0, 0, 0, 0.3);
        padding: 8px 12px;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.85rem;
        color: #e2e8f0;
    }
    
    /* Result Section */
    .result-card {
        background: rgba(15, 23, 42, 0.6);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        animation: glow 2s ease-in-out infinite alternate;
        margin-top: 2rem;
        box-shadow: 0 0 15px rgba(0,0,0,0.2);
    }

    @keyframes glow {
        from { box-shadow: 0 0 10px rgba(168, 85, 247, 0.2); border-color: rgba(168, 85, 247, 0.2); }
        to { box-shadow: 0 0 25px rgba(168, 85, 247, 0.5); border-color: rgba(168, 85, 247, 0.5); }
    }

    .prediction-title {
        font-size: 1.1rem;
        color: #cbd5e1;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
    }

    .prediction-value {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
    }

    .status-normal { color: #4ade80; text-shadow: 0 0 20px rgba(74, 222, 128, 0.4); }
    .status-moderate { color: #fb923c; text-shadow: 0 0 20px rgba(251, 146, 60, 0.4); }
    .status-risky { color: #f87171; text-shadow: 0 0 20px rgba(248, 113, 113, 0.4); }

    .insight-text {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-top: 1rem;
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
        user_inputs = {}
        
        st.markdown('<div class="section-header">🏃 Movement Metrics</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        metrics = ['tBodyAcc-mean()-X', 'tBodyAcc-mean()-Z', 'fBodyAcc-mean()-X']
        for i, feat in enumerate(metrics):
            with col1 if i % 2 == 0 else col2:
                name = FEATURE_MAP.get(feat, feat)
                user_inputs[feat] = st.slider(name, min_value=-2.0, max_value=2.0, value=0.0, step=0.01, help=f"Sensor: {feat}")

        st.markdown('<div class="section-header">🔄 Motion Behavior</div>', unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        behavior = ['tGravityAcc-mean()-X', 'tGravityAcc-mean()-Y', 'tBodyGyro-mean()-X', 'tBodyGyro-mean()-Z']
        for i, feat in enumerate(behavior):
            with col3 if i % 2 == 0 else col4:
                name = FEATURE_MAP.get(feat, feat)
                user_inputs[feat] = st.slider(name, min_value=-2.0, max_value=2.0, value=0.0, step=0.01, help=f"Sensor: {feat}")

        st.markdown('<div class="section-header">⚖️ Stability & Energy</div>', unsafe_allow_html=True)
        col5, col6 = st.columns(2)
        stability = ['tBodyAccMag-mean()', 'tGravityAccMag-mean()', 'fBodyAcc-std()-Y']
        for i, feat in enumerate(stability):
            with col5 if i % 2 == 0 else col6:
                name = FEATURE_MAP.get(feat, feat)
                user_inputs[feat] = st.slider(name, min_value=-2.0, max_value=2.0, value=0.0, step=0.01, help=f"Sensor: {feat}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Analyze Activity")
        
        if submitted:
            # Prepare full feature vector (561 features)
            full_input = np.zeros((1, len(assets['feature_names'])))
            for feat, val in user_inputs.items():
                if feat in assets['feature_names']:
                    idx = assets['feature_names'].index(feat)
                    full_input[0, idx] = val
            
            with st.spinner("Analyzing Movement Data..."):
                time.sleep(1) # Smooth loading effect
                
                # Predict
                scaled_input = assets['scaler'].transform(full_input)
                prediction_cluster = assets['kmeans'].predict(scaled_input)[0]
                activity_label = assets['cluster_activity_mapping'].get(prediction_cluster, f"Cluster {prediction_cluster}")
                
                # Setup Display Properties
                activity_upper = activity_label.upper()
                if "WALKING" in activity_upper:
                    icon = "🏃"
                    status_class = "status-risky" if "UPSTAIRS" in activity_upper or "DOWNSTAIRS" in activity_upper else "status-moderate"
                    insight = "High frequency movement detected. Matches active cardio profiles."
                elif "SITTING" in activity_upper or "LAYING" in activity_upper:
                    icon = "🧍"
                    status_class = "status-normal"
                    insight = "Low impact/static position detected. Matches sedentary profiles."
                else:
                    icon = "🚶"
                    status_class = "status-normal"
                    insight = "Moderate activity detected."
                
                st.markdown(f'''
                <div class="result-card">
                    <div class="prediction-title">Detected Activity</div>
                    <div class="prediction-value {status_class}">
                        {icon} <span>{activity_label}</span>
                    </div>
                    <div class="insight-text">💡 {insight}</div>
                </div>
                ''', unsafe_allow_html=True)
            
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br><p style='text-align: center; color: #475569; font-size: 0.85rem; margin-top: 2rem;'>Powered by K-Means Clustering & Scikit-Learn</p>", unsafe_allow_html=True)
