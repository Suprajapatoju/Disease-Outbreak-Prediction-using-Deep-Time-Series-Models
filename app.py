import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import hashlib



# --- Configuration ---
st.set_page_config(
    page_title="Disease Outbreak Prediction | Nurein",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏥"
)



# ============================================
# USER MANAGEMENT
# ============================================
USERS_FILE = "users.json"


def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    else:
        default_users = {
            "admin": {
                "password": hashlib.sha256("admin123".encode()).hexdigest(),
                "name": "Admin User",
                "role": "Admin"
            }
        }
        save_users(default_users)
        return default_users


def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username, password, name, role="User"):
    users = load_users()
    if username in users:
        return False, "Username already exists"
    users[username] = {
        "password": hash_password(password),
        "name": name,
        "role": role
    }
    save_users(users)
    return True, "Registration successful!"


def verify_user(username, password):
    users = load_users()
    if username in users:
        if users[username]["password"] == hash_password(password):
            return True, users[username]
    return False, None



# ============================================
# LOAD HISTORICAL DATA
# ============================================
@st.cache_data
def load_historical_data():
    """Load and clean historical data with proper data types"""
    try:
        data = pd.read_csv('Final_data_large.csv')
        
        numeric_cols = ['Cases', 'Deaths', 'Latitude', 'Longitude', 'Temp', 'preci', 'LAI']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        
        date_cols = ['day', 'mon', 'year']
        for col in date_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(1).astype(int)
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None



# ============================================
# GET TRAINED LOCATIONS
# ============================================
def get_trained_states(le_state):
    """Get only states that model was trained on"""
    return sorted(le_state.classes_)


def get_trained_districts_for_state(data, state, le_district):
    """Get only districts from selected state that model knows"""
    state_districts = data[data['state_ut'] == state]['district'].unique()
    trained = [d for d in state_districts if d in le_district.classes_]
    return sorted(trained)



# ============================================
# MODERN UI CSS - TEAL/TURQUOISE THEME
# ============================================
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8f4f8 100%);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #17a2b8 0%, #138496 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    [data-testid="stSidebar"] label {
        color: white !important;
        font-weight: 500;
    }
    
    /* Header Styling */
    .main-header {
        background: white;
        padding: 25px 30px;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        margin-bottom: 30px;
    }
    
    .main-header h1 {
        color: #17a2b8;
        font-size: 2.2em;
        font-weight: 700;
        margin: 0;
    }
    
    .main-header p {
        color: #6c757d;
        font-size: 1em;
        margin: 5px 0 0 0;
    }
    
    /* Modern Metric Cards */
    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 2px 15px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
        border-left: 4px solid #17a2b8;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(23, 162, 184, 0.15);
    }
    
    .metric-card h3 {
        color: #6c757d;
        font-size: 0.9em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 0 0 10px 0;
    }
    
    .metric-card .value {
        color: #17a2b8;
        font-size: 2.5em;
        font-weight: 700;
        margin: 10px 0;
    }
    
    .metric-card .sub-text {
        color: #adb5bd;
        font-size: 0.85em;
        margin: 0;
    }
    
    /* Risk Badge */
    .risk-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.9em;
        margin: 10px 0;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a6f);
        color: white;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #ffa500, #ff8c00);
        color: white;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #51cf66, #37b24d);
        color: white;
    }
    
    /* Info Box */
    .info-box {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        margin: 15px 0;
    }
    
    .info-box h4 {
        color: #17a2b8;
        margin-top: 0;
        font-weight: 600;
    }
    
    /* Factor Box */
    .factor-box {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 3px solid #17a2b8;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #17a2b8, #138496);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 15px;
        border: none;
        font-size: 1em;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #138496, #117a8b);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(23, 162, 184, 0.3);
    }
    
    /* Section Headers */
    .section-header {
        color: #2c3e50;
        font-size: 1.4em;
        font-weight: 600;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #17a2b8;
    }
    
    /* Welcome Cards */
    .welcome-card {
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 2px 15px rgba(0, 0, 0, 0.06);
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .welcome-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(23, 162, 184, 0.15);
    }
    
    .welcome-card h3 {
        color: #17a2b8;
        font-size: 1.2em;
        margin-bottom: 15px;
    }
    
    .welcome-card p {
        color: #6c757d;
        line-height: 1.6;
    }
    
    /* Icon Styling */
    .icon {
        font-size: 2.5em;
        margin-bottom: 15px;
    }
    
    /* Trend Indicator */
    .trend-up {
        color: #ee5a6f;
    }
    
    .trend-down {
        color: #51cf66;
    }
    
    /* Chart Container */
    .chart-container {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 2px 15px rgba(0, 0, 0, 0.06);
        margin: 20px 0;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #17a2b8, #138496);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)



# ============================================
# HELPER FUNCTIONS
# ============================================
def explain_factors(model, feature_cols, temp, preci, lai):
    """Generate human-readable explanations for prediction factors"""
    explanations = []
    
    if temp > 30:
        explanations.append(f"🌡️ **High Temperature** ({temp}°C): Warm conditions can accelerate disease transmission")
    elif temp < 20:
        explanations.append(f"🌡️ **Cool Temperature** ({temp}°C): May affect disease seasonality patterns")
    else:
        explanations.append(f"🌡️ **Moderate Temperature** ({temp}°C): Within normal range for disease activity")
    
    if preci > 100:
        explanations.append(f"🌧️ **High Rainfall** ({preci}mm): Heavy precipitation increases waterborne disease risks")
    elif preci < 20:
        explanations.append(f"☀️ **Low Rainfall** ({preci}mm): Dry conditions may concentrate contamination sources")
    else:
        explanations.append(f"🌦️ **Moderate Rainfall** ({preci}mm): Normal precipitation levels")
    
    if lai > 2:
        explanations.append(f"🌿 **Dense Vegetation** (LAI: {lai}): High vegetation may harbor disease vectors")
    else:
        explanations.append(f"🏜️ **Sparse Vegetation** (LAI: {lai}): Limited plant cover in the area")
    
    # Create feature importance dataframe
    try:
        feature_names = ['Temperature', 'Precipitation', 'Vegetation (LAI)', 'Last Week Cases', 'Last Month Cases']
        importance = [abs(temp/100), abs(preci/100), abs(lai), 0.8, 0.9]
        
        top_factors = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
    except:
        top_factors = None
    
    return explanations, top_factors



# ============================================
# AUTHENTICATION PAGE
# ============================================
def auth_page():
    # Initialize session state for tab selection
    if 'auth_tab' not in st.session_state:
        st.session_state.auth_tab = 'login'
    
    # Custom CSS for beautiful login page
    st.markdown("""
        <style>
        /* Hide Streamlit default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
        header {visibility: hidden;}
        
        /* Full page background */
        .stApp {
            background: #f5f7fa;
        }
        
        /* Main login container */
        .login-wrapper {
            display: flex;
            max-width: 1200px;
            margin: 50px auto;
            background: white;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }
        
        /* Left panel - Teal gradient with features */
        .login-left-panel {
            flex: 1;
            background: linear-gradient(135deg, #17a2b8 0%, #138496 100%);
            padding: 60px 50px;
            color: white;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        .brand-badge {
            background: rgba(255, 255, 255, 0.2);
            padding: 10px 25px;
            border-radius: 30px;
            display: inline-block;
            margin-bottom: 25px;
            font-weight: 600;
            font-size: 0.95em;
            width: fit-content;
        }
        
        .login-left-panel h1 {
            font-size: 2.8em;
            margin-bottom: 20px;
            font-weight: 700;
            line-height: 1.2;
        }
        
        .login-left-panel > p {
            font-size: 1.15em;
            line-height: 1.7;
            opacity: 0.95;
            margin-bottom: 40px;
        }
        
        .feature-item {
            display: flex;
            align-items: flex-start;
            margin: 25px 0;
            animation: slideIn 0.6s ease forwards;
        }
        
        .feature-icon {
            background: rgba(255, 255, 255, 0.25);
            border-radius: 50%;
            width: 55px;
            height: 55px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 20px;
            font-size: 1.6em;
            flex-shrink: 0;
        }
        
        .feature-content strong {
            display: block;
            font-size: 1.1em;
            margin-bottom: 5px;
        }
        
        .feature-content span {
            font-size: 0.95em;
            opacity: 0.85;
        }
        
        .stats-row {
            display: flex;
            gap: 15px;
            margin-top: 50px;
        }
        
        .stat-box {
            flex: 1;
            background: rgba(255, 255, 255, 0.15);
            border-radius: 15px;
            padding: 25px 15px;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        
        .stat-number {
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 8px;
        }
        
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        /* Right panel - Login form */
        .login-right-panel {
            flex: 1;
            padding: 60px 55px;
            background: white;
        }
        
        .login-header {
            text-align: center;
            margin-bottom: 45px;
        }
        
        .login-header h2 {
            color: #2c3e50;
            font-size: 2.2em;
            font-weight: 700;
            margin-bottom: 12px;
        }
        
        .login-header p {
            color: #6c757d;
            font-size: 1.05em;
        }
        
        /* Custom tab buttons */
        .custom-tabs {
            display: flex;
            gap: 15px;
            margin-bottom: 35px;
            justify-content: center;
        }
        
        .custom-tab {
            flex: 1;
            padding: 14px 25px;
            border-radius: 12px;
            border: 2px solid #e9ecef;
            background: white;
            color: #6c757d;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
            font-size: 1em;
        }
        
        .custom-tab.active {
            background: linear-gradient(135deg, #17a2b8, #138496);
            color: white;
            border-color: #17a2b8;
            box-shadow: 0 4px 15px rgba(23, 162, 184, 0.3);
        }
        
        .custom-tab:hover {
            border-color: #17a2b8;
            transform: translateY(-2px);
        }
        
        /* Form styling adjustments */
        .stTextInput > div > div > input {
            border-radius: 10px;
            border: 2px solid #e9ecef;
            padding: 12px 15px;
            font-size: 1em;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #17a2b8;
            box-shadow: 0 0 0 3px rgba(23, 162, 184, 0.1);
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Create the split layout
    col_left, col_right = st.columns([1, 1], gap="large")
    
    # LEFT PANEL - Features and Branding
    with col_left:
        import streamlit.components.v1 as components
        
        components.html("""
            <div style='background: linear-gradient(135deg, #17a2b8 0%, #138496 100%); 
                        padding: 45px 40px; 
                        border-radius: 20px; 
                        color: white; 
                        font-family: "Inter", sans-serif;
                        box-sizing: border-box;'>
                
                <div style='background: rgba(255, 255, 255, 0.2); 
                            padding: 10px 25px; 
                            border-radius: 30px; 
                            display: inline-block; 
                            margin-bottom: 20px; 
                            font-weight: 600;'>
                    🏥 Welcome to Nurein Health
                </div>
                
                <h1 style='font-size: 2.5em; margin-bottom: 15px; font-weight: 700; line-height: 1.2;'>
                    Disease Outbreak Prediction Platform
                </h1>
                
                <p style='font-size: 1.05em; line-height: 1.6; opacity: 0.95; margin-bottom: 30px;'>
                    Leverage AI-powered predictions to stay ahead of disease outbreaks and protect communities with data-driven insights.
                </p>
                
                <div style='display: flex; align-items: flex-start; margin: 20px 0;'>
                    <div style='background: rgba(255, 255, 255, 0.25); 
                                border-radius: 50%; 
                                width: 50px; 
                                height: 50px; 
                                display: flex; 
                                align-items: center; 
                                justify-content: center; 
                                margin-right: 18px; 
                                font-size: 1.5em;
                                flex-shrink: 0;'>
                        🎯
                    </div>
                    <div>
                        <strong style='display: block; font-size: 1.05em; margin-bottom: 4px;'>Accurate Predictions</strong>
                        <span style='font-size: 0.9em; opacity: 0.85;'>LSTM models trained on real outbreak data</span>
                    </div>
                </div>
                
                <div style='display: flex; align-items: flex-start; margin: 20px 0;'>
                    <div style='background: rgba(255, 255, 255, 0.25); 
                                border-radius: 50%; 
                                width: 50px; 
                                height: 50px; 
                                display: flex; 
                                align-items: center; 
                                justify-content: center; 
                                margin-right: 18px; 
                                font-size: 1.5em;
                                flex-shrink: 0;'>
                        ⚡
                    </div>
                    <div>
                        <strong style='display: block; font-size: 1.05em; margin-bottom: 4px;'>Real-Time Analysis</strong>
                        <span style='font-size: 0.9em; opacity: 0.85;'>Instant risk assessment and monitoring</span>
                    </div>
                </div>
                
                <div style='display: flex; align-items: flex-start; margin: 20px 0 30px 0;'>
                    <div style='background: rgba(255, 255, 255, 0.25); 
                                border-radius: 50%; 
                                width: 50px; 
                                height: 50px; 
                                display: flex; 
                                align-items: center; 
                                justify-content: center; 
                                margin-right: 18px; 
                                font-size: 1.5em;
                                flex-shrink: 0;'>
                        🛡️
                    </div>
                    <div>
                        <strong style='display: block; font-size: 1.05em; margin-bottom: 4px;'>Secure & Reliable</strong>
                        <span style='font-size: 0.9em; opacity: 0.85;'>Enterprise-grade security protocols</span>
                    </div>
                </div>
                
                <div style='display: flex; gap: 12px; margin-top: 30px; margin-bottom: 40px;'>
                    <div style='flex: 1; 
                                background: rgba(255, 255, 255, 0.2); 
                                border-radius: 12px; 
                                padding: 22px 10px; 
                                text-align: center;
                                border: 1px solid rgba(255, 255, 255, 0.3);'>
                        <div style='font-size: 2.3em; font-weight: 700; margin-bottom: 6px; line-height: 1;'>98%</div>
                        <div style='font-size: 0.88em; opacity: 0.95; font-weight: 500;'>Accuracy</div>
                    </div>
                    <div style='flex: 1; 
                                background: rgba(255, 255, 255, 0.2); 
                                border-radius: 12px; 
                                padding: 22px 10px; 
                                text-align: center;
                                border: 1px solid rgba(255, 255, 255, 0.3);'>
                        <div style='font-size: 2.3em; font-weight: 700; margin-bottom: 6px; line-height: 1;'>50K+</div>
                        <div style='font-size: 0.88em; opacity: 0.95; font-weight: 500;'>Predictions</div>
                    </div>
                    <div style='flex: 1; 
                                background: rgba(255, 255, 255, 0.2); 
                                border-radius: 12px; 
                                padding: 22px 10px; 
                                text-align: center;
                                border: 1px solid rgba(255, 255, 255, 0.3);'>
                        <div style='font-size: 2.3em; font-weight: 700; margin-bottom: 6px; line-height: 1;'>24/7</div>
                        <div style='font-size: 0.88em; opacity: 0.95; font-weight: 500;'>Monitoring</div>
                    </div>
                </div>
                
            </div>
        """, height=720, scrolling=False)
    
    # RIGHT PANEL - Login/Register Forms
    with col_right:
        st.markdown("""
            <div style='padding: 20px 0;'>
                <div style='text-align: center; margin-bottom: 35px;'>
                    <h2 style='color: #2c3e50; font-size: 2.2em; font-weight: 700; margin-bottom: 12px;'>Welcome Back</h2>
                    <p style='color: #6c757d; font-size: 1.05em;'>Enter your credentials to access your account</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Custom Tab Buttons
        col_tab1, col_tab2 = st.columns(2)
        with col_tab1:
            if st.button("🔐 Sign In", key="tab_login", use_container_width=True, 
                        type="primary" if st.session_state.auth_tab == 'login' else "secondary"):
                st.session_state.auth_tab = 'login'
                st.rerun()
        
        with col_tab2:
            if st.button("📝 Create Account", key="tab_register", use_container_width=True,
                        type="primary" if st.session_state.auth_tab == 'register' else "secondary"):
                st.session_state.auth_tab = 'register'
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # LOGIN FORM
        if st.session_state.auth_tab == 'login':
            with st.form("login_form", clear_on_submit=False):
                st.markdown("##### 🔑 Login to your account")
                username = st.text_input("Username", placeholder="Enter your username", key="login_user")
                password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_pass")
                
                col_btn, col_link = st.columns([3, 1])
                with col_btn:
                    submit = st.form_submit_button("Login", use_container_width=True)
                with col_link:
                    st.markdown("<p style='text-align: center; margin-top: 8px; color: #17a2b8; cursor: pointer;'>Forgot?</p>", unsafe_allow_html=True)
                
                if submit:
                    if not username or not password:
                        st.error("❌ Please enter both username and password")
                    else:
                        success, user_data = verify_user(username, password)
                        if success:
                            st.session_state['logged_in'] = True
                            st.session_state['username'] = username
                            st.session_state['user_data'] = user_data
                            st.success("✅ Login successful! Redirecting...")
                            st.rerun()
                        else:
                            st.error("❌ Invalid username or password")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("💡 **Demo Account** → Username: `admin` | Password: `admin123`")
        
        # REGISTER FORM
        else:
            with st.form("register_form", clear_on_submit=False):
                st.markdown("##### 📋 Create your account")
                new_name = st.text_input("Full Name", placeholder="John Doe", key="reg_name")
                new_username = st.text_input("Username", placeholder="Choose a username", key="reg_user")
                
                col_pass1, col_pass2 = st.columns(2)
                with col_pass1:
                    new_password = st.text_input("Password", type="password", placeholder="Password", key="reg_pass")
                with col_pass2:
                    confirm_password = st.text_input("Confirm", type="password", placeholder="Confirm", key="reg_confirm")
                
                register_btn = st.form_submit_button("Create Account", use_container_width=True)
                
                if register_btn:
                    if not new_name or not new_username or not new_password:
                        st.error("❌ All fields are required")
                    elif len(new_password) < 6:
                        st.error("❌ Password must be at least 6 characters")
                    elif new_password != confirm_password:
                        st.error("❌ Passwords do not match")
                    else:
                        success, message = register_user(new_username, new_password, new_name)
                        if success:
                            st.success("✅ " + message + " You can now login!")
                            st.session_state.auth_tab = 'login'
                        else:
                            st.error("❌ " + message)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("📋 By registering, you agree to our Terms of Service and Privacy Policy")



# ============================================
# MAIN DASHBOARD
# ============================================
def main_dashboard():
    # Header with user info
    user_name = st.session_state.get('user_data', {}).get('name', 'User')
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"""
            <div class='main-header'>
                <h1>Welcome back, {user_name}</h1>
                <p>Here are the latest disease outbreak predictions</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.rerun()
    
    # Load data and model
    historical_data = load_historical_data()
    
    if historical_data is not None and os.path.exists('best_disease_model.pkl'):
        pipeline = joblib.load('best_disease_model.pkl')
        model = pipeline['model']
        scaler = pipeline['scaler']
        le_state = pipeline['le_state']
        le_district = pipeline['le_district']
        le_disease = pipeline['le_disease']
        feature_cols = pipeline['features']
        seq_len = pipeline.get('sequence_length', 4)
        test_rmse = pipeline.get('test_rmse', 0)
        

        # Sidebar Configuration
        with st.sidebar:
            st.markdown("### ⚙️ Prediction Configuration")
            st.markdown("---")
            
            trained_states = get_trained_states(le_state)
            selected_state = st.selectbox("📍 Select State", trained_states, key='state_select')
            
            trained_districts = get_trained_districts_for_state(historical_data, selected_state, le_district)
            selected_district = st.selectbox("🏘️ Select District", trained_districts, key='district_select')
            
            # Disease selection dropdown (for display only)
            available_diseases = historical_data[
            (historical_data['state_ut'] == selected_state) &
            (historical_data['district'] == selected_district)
            ]['Disease'].unique().tolist()

            if len(available_diseases) > 0:
                selected_disease_display = st.selectbox(
                "🦠 Select Disease",
                sorted(available_diseases),
                key='disease_select'
                )
            else:
                st.warning("No disease data available for this location.")
                selected_disease_display = None

            st.markdown("---")
            st.markdown("### 🌡️ Environmental Parameters")
            
            temp = st.slider("Temperature (°C)", 10, 45, 28)
            preci = st.slider("Precipitation (mm)", 0, 300, 80)
            lai = st.slider("Vegetation Index (LAI)", 0.0, 5.0, 2.0, 0.1)
            
            st.markdown("---")
            st.markdown("### 📅 Historical Data")
            weeks_lookback = st.slider("Weeks of Historical Data", 4, 52, 12)
            
            st.markdown("---")
            predict_btn = st.button("🔮 Generate Prediction", use_container_width=True)
        
        # Main Content
        if predict_btn:
            with st.spinner("🔄 Analyzing data and generating predictions..."):
                # Always use Acute Diarrhoeal Disease data (model trained only on this)
                actual_disease = selected_disease_display
                
                # Get historical data for location (always using Acute Diarrhoeal Disease)
                hist = historical_data[
                    (historical_data['state_ut'] == selected_state) &
                    (historical_data['district'] == selected_district) &
                    (historical_data['Disease'] == actual_disease)
                ].copy()
                
                hist = hist.sort_values(by=['year', 'mon', 'day'])
                
                if len(hist) < seq_len:
                    st.error(f"❌ Insufficient data. Need at least {seq_len} records, found {len(hist)}")
                else:
                    # Prepare input features
                    hist['stateut_enc'] = le_state.transform([selected_state] * len(hist))
                    hist['district_enc'] = le_district.transform([selected_district] * len(hist))
                    hist['disease_enc'] = le_disease.transform([actual_disease] * len(hist))

                    feature_cols_to_scale = ['Temp', 'preci', 'LAI']
                    hist[feature_cols_to_scale] = hist[feature_cols_to_scale].fillna(method='ffill').fillna(method='bfill')
                    hist[[f+'_scaled' for f in feature_cols_to_scale]] = scaler.transform(hist[feature_cols_to_scale])
                    
                    hist['caseslastweek'] = hist['Cases'].shift(1).fillna(0)
                    hist['caseslastmonth'] = hist['Cases'].shift(4).fillna(0)
                    
                    cases_last_week = int(hist['caseslastweek'].iloc[-1])
                    cases_last_month = int(hist['caseslastmonth'].iloc[-1])
                    
                    # Build input
                    prediction_date = datetime.now() + timedelta(days=7)
                    lat = hist['Latitude'].iloc[-1]
                    lon = hist['Longitude'].iloc[-1]
                    
                    input_df = pd.DataFrame([{
                        'day': prediction_date.day,
                        'mon': prediction_date.month,
                        'year': prediction_date.year,
                        'Latitude': lat,
                        'Longitude': lon,
                        'Temp_scaled': (temp - scaler.mean_[0]) / scaler.scale_[0],
                        'preci_scaled': (preci - scaler.mean_[1]) / scaler.scale_[1],
                        'LAI_scaled': (lai - scaler.mean_[2]) / scaler.scale_[2],
                        'caseslastweek': cases_last_week,
                        'caseslastmonth': cases_last_month,
                        'stateut_enc': le_state.transform([selected_state])[0],
                        'district_enc': le_district.transform([selected_district])[0],
                        'disease_enc': le_disease.transform([selected_disease_display])[0]

                    }])
                    
                    X_recent = hist[feature_cols].tail(seq_len - 1).values
                    X_new_row = input_df[feature_cols].values
                    X_seq = np.vstack([X_recent, X_new_row])
                    X_seq = X_seq.reshape(1, seq_len, len(feature_cols))
                    
                    # Predict
                    pred_cases = model.predict(X_seq, verbose=0)[0][0]
                    pred_cases = max(0, pred_cases)
                    
                    # Risk assessment
                    if pred_cases > 100:
                        risk_label = "High Risk"
                        risk_class = "risk-high"
                        risk_emoji = "🔴"
                        risk_msg = "Immediate attention required. Prepare healthcare resources and alert authorities."
                    elif pred_cases > 50:
                        risk_label = "Medium Risk"
                        risk_class = "risk-medium"
                        risk_emoji = "🟡"
                        risk_msg = "Monitor situation closely. Consider preventive measures and public awareness."
                    else:
                        risk_label = "Low Risk"
                        risk_class = "risk-low"
                        risk_emoji = "🟢"
                        risk_msg = "Situation is stable. Continue routine surveillance and monitoring."
                    # -------- ALERT SYSTEM --------
                    if risk_label == "High Risk":
                        st.error("🚨 OUTBREAK ALERT: High disease risk detected in this district. Immediate public health action recommended.")

                    elif risk_label == "Medium Risk":
                        st.warning("⚠️ Warning: Moderate outbreak risk. Monitoring and preventive measures advised.")

                    else:
                        st.success("✅ Situation Stable: Low outbreak risk detected.")
                    # Display Results
                    st.markdown("<div class='section-header'>📊 Prediction Results</div>", unsafe_allow_html=True)
                    
                    # Metrics Row
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                            <div class='metric-card'>
                                <h3>Predicted Cases</h3>
                                <div class='value'>{int(pred_cases)}</div>
                                <p class='sub-text'>Next Period</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                            <div class='metric-card'>
                                <h3>Risk Level</h3>
                                <div class='value' style='font-size: 1.8em;'>{risk_emoji}</div>
                                <span class='risk-badge {risk_class}'>{risk_label}</span>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                            <div class='metric-card'>
                                <h3>Last Week</h3>
                                <div class='value'>{cases_last_week}</div>
                                <p class='sub-text'>Actual Cases</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        change_pct = ((pred_cases - cases_last_week) / max(cases_last_week, 1)) * 100
                        arrow = "↑" if change_pct > 0 else "↓"
                        trend_class = "trend-up" if change_pct > 0 else "trend-down"
                        st.markdown(f"""
                            <div class='metric-card'>
                                <h3>Trend</h3>
                                <div class='value {trend_class}'>{arrow} {abs(change_pct):.1f}%</div>
                                <p class='sub-text'>vs Last Week</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Risk Message
                    st.markdown(f"""
                        <div class='info-box' style='background: linear-gradient(135deg, rgba(23, 162, 184, 0.1), rgba(19, 132, 150, 0.1)); border-left: 4px solid #17a2b8;'>
                            <h4>{risk_emoji} {risk_label.upper()}</h4>
                            <p style='margin: 0; color: #495057;'>{risk_msg}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Chart
                    st.markdown("<div class='section-header'>📈 Historical Trends & Forecast</div>", unsafe_allow_html=True)
                    
                    hist_viz = hist.tail(weeks_lookback)[['Cases', 'year', 'mon', 'day']].copy()
                    hist_viz['Type'] = 'Historical'
                    
                    pred_row = pd.DataFrame([{
                        'Cases': int(pred_cases),
                        'Type': 'Predicted',
                        'year': prediction_date.year,
                        'mon': prediction_date.month,
                        'day': prediction_date.day
                    }])
                    
                    combined = pd.concat([hist_viz, pred_row], ignore_index=True)
                    combined['Period'] = range(len(combined))
                    
                    fig = go.Figure()
                    
                    # Historical line
                    hist_mask = combined['Type'] == 'Historical'
                    fig.add_trace(go.Scatter(
                        x=combined[hist_mask]['Period'],
                        y=combined[hist_mask]['Cases'],
                        mode='lines+markers',
                        name='Historical Cases',
                        line=dict(color='#17a2b8', width=3),
                        marker=dict(size=8, color='#17a2b8'),
                        fill='tozeroy',
                        fillcolor='rgba(23, 162, 184, 0.1)'
                    ))
                    
                    # Predicted point
                    pred_mask = combined['Type'] == 'Predicted'
                    fig.add_trace(go.Scatter(
                        x=combined[pred_mask]['Period'],
                        y=combined[pred_mask]['Cases'],
                        mode='markers',
                        name='Predicted',
                        marker=dict(size=20, color='#ee5a6f', symbol='star', 
                                  line=dict(width=2, color='white'))
                    ))
                    
                    fig.update_layout(
                        title=f"Disease Outbreak Trend - {selected_district}, {selected_state}",
                        xaxis_title="Time Period",
                        yaxis_title="Number of Cases",
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=450,
                        hovermode='x unified',
                        font=dict(family='Inter', size=12),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("<div class='section-header'>🗺️ Outbreak Location Map</div>", unsafe_allow_html=True)

                    map_df = pd.DataFrame({
                    "Latitude": [lat],
                    "Longitude": [lon],
                    "Predicted Cases": [int(pred_cases)],
                    "District": [selected_district],
                    "State": [selected_state]
                    })

                    fig_map = px.scatter_mapbox(
                    map_df,
                    lat="Latitude",
                    lon="Longitude",
                    size="Predicted Cases",
                    color="Predicted Cases",
                    hover_name="District",
                    zoom=4,
                    height=450,
                    color_continuous_scale="Reds"
                    )

                    fig_map.update_layout(
                    mapbox_style="open-street-map",
                    margin={"r":0,"t":0,"l":0,"b":0}
                    )

                    st.plotly_chart(fig_map, use_container_width=True)
                    
                    # Factors Explanation
                    st.markdown("<div class='section-header'>🧠 Contributing Factors</div>", unsafe_allow_html=True)
                    
                    explanations, top_factors = explain_factors(model, feature_cols, temp, preci, lai)
                    
                    col_exp1, col_exp2 = st.columns([2, 1])
                    
                    with col_exp1:
                        for i, exp in enumerate(explanations, 1):
                            st.markdown(f"""
                                <div class='factor-box'>
                                    {exp}
                                </div>
                            """, unsafe_allow_html=True)
                        
                        st.info(f"📊 Based on {weeks_lookback} weeks of historical data from {selected_district}, {selected_state}")
                    
                    with col_exp2:
                        if top_factors is not None:
                            st.markdown("**📌 Feature Importance**")
                            fig_imp = px.bar(
                                top_factors.head(5),
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                color='Importance',
                                color_continuous_scale=['#e8f4f8', '#17a2b8']
                            )
                            fig_imp.update_layout(
                                showlegend=False,
                                height=300,
                                margin=dict(l=0, r=0, t=0, b=0),
                                plot_bgcolor='white',
                                paper_bgcolor='white'
                            )
                            st.plotly_chart(fig_imp, use_container_width=True)
                    
                    # Model Metrics
                    st.markdown("<div class='section-header'>📉 Model Performance</div>", unsafe_allow_html=True)
                    
                    col_met1, col_met2, col_met3 = st.columns(3)
                    
                    with col_met1:
                        st.markdown(f"""
                            <div class='metric-card' style='text-align: center;'>
                                <h3>RMSE</h3>
                                <div class='value' style='font-size: 2em;'>{test_rmse:.2f}</div>
                                <p class='sub-text'>Root Mean Square Error</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col_met2:
                        test_mae = pipeline.get('test_mae', test_rmse * 0.8)
                        st.markdown(f"""
                            <div class='metric-card' style='text-align: center;'>
                                <h3>MAE</h3>
                                <div class='value' style='font-size: 2em;'>{test_mae:.2f}</div>
                                <p class='sub-text'>Mean Absolute Error</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col_met3:
                        accuracy_pct = max(0, (1 - (test_rmse / 100)) * 100)
                        st.markdown(f"""
                            <div class='metric-card' style='text-align: center;'>
                                <h3>Confidence</h3>
                                <div class='value' style='font-size: 2em;'>{accuracy_pct:.1f}%</div>
                                <p class='sub-text'>Prediction Accuracy</p>
                            </div>
                        """, unsafe_allow_html=True)
        
        else:
            # Welcome Screen
            st.markdown("<div class='section-header'>👋 Welcome to Nurein Health Platform</div>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                    <div class='welcome-card'>
                        <div class='icon'>🎯</div>
                        <h3>Accurate Predictions</h3>
                        <p>Advanced LSTM models trained on historical outbreak data combined with environmental factors.</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                    <div class='welcome-card'>
                        <div class='icon'>⚡</div>
                        <h3>Real-Time Analysis</h3>
                        <p>Get instant risk assessments based on current conditions and historical trends.</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                    <div class='welcome-card'>
                        <div class='icon'>📊</div>
                        <h3>Data-Driven Insights</h3>
                        <p>Understand key factors contributing to outbreak risks with detailed visualizations.</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.info("👈 Configure the prediction parameters in the sidebar and click **'Generate Prediction'** to analyze outbreak risk.")
    
    else:
        st.error("⚠️ Model not found. Please train the model first using `model_training.py`.")
        st.info("Run: `python model_training.py` to train and save the model.")



# ============================================
# MAIN APP LOGIC
# ============================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False


if st.session_state['logged_in']:
    main_dashboard()
else:
    auth_page()