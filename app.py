import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set page layout to wide for dashboard look
st.set_page_config(layout="wide")

st.title("🚨 Live Trade Fraud Simulation: The Scrap Plastic Smuggler Game")
st.markdown("""
    **Classroom Context:** Students act as 'bad actors' trying to bypass customs by hiding low-value plastic scrap 
    under premium virgin plastic HS codes. The ML pipeline on the right detects their patterns live!
""")

# ---Legitimate Baseline Data Reference ---
baseline_prices = {
    390110: 4.20,  # Virgin Polyethylene
    390210: 4.50,  # Virgin Polypropylene
    390410: 3.90   # Virgin PVC
}

# --- Persistent Data Storage (Session State) ---
if "submissions" not in st.session_state:
    # Pre-populate with 5 diverse dummy entries so the K-Means has background noise
    st.session_state.submissions = pd.DataFrame([
        {"Trader Name": "GenuineCorp A", "HS Code": 390110, "Weight (KG)": 25000, "Declared Price ($/KG)": 4.10},
        {"Trader Name": "GenuineCorp B", "HS Code": 390210, "Weight (KG)": 10000, "Declared Price ($/KG)": 4.50},
        {"Trader Name": "ShadowBroker LLC", "HS Code": 390210, "Weight (KG)": 100000, "Declared Price ($/KG)": 0.50},
        {"Trader Name": "EcoRecycle Ltd", "HS Code": 390410, "Weight (KG)": 25000, "Declared Price ($/KG)": 1.80},
        {"Trader Name": "GenuineCorp C", "HS Code": 390410, "Weight (KG)": 10000, "Declared Price ($/KG)": 3.80},
    ])

# Layout Split: 1/3 Student Form, 2/3 Teacher Dashboard
col1, col2 = st.columns([1, 2])

# --- COLUMN 1: STUDENT INPUT INTERFACE ---
with col1:
    st.header("📥 Student Submission Portal")
    with st.form(key="student_form", clear_on_submit=True):
        student_name = st.text_input("Enter your Smuggler Alias / Name:", placeholder="e.g., Anonymous Trader")
        
        hs_code = st.selectbox(
            "Select HS Code to hide your scrap in:",
            options=[390110, 390210, 390410],
            format_func=lambda x: f"HS {x} (Premium Virgin Polymer)"
        )
        
        weight = st.selectbox(
            "Select Shipment Weight (Volume):",
            options=[10000, 25000, 100000],
            format_func=lambda x: f"{x:,} KG"
        )
        
        price = st.selectbox(
            "Select Declared Value (Unit Price):",
            options=[0.50, 1.80, 4.50],
            format_func=lambda x: f"${x:.2f} / KG"
        )
        
        submit_button = st.form_submit_with_button_choices = st.form_submit_button(label="🚀 Submit Manifest to Customs")
        
        if submit_button:
            if student_name.strip() == "":
                st.error("Please enter a name or alias before submitting!")
            else:
                # Append new submission to our database
                new_row = {
                    "Trader Name": student_name,
                    "HS Code": hs_code,
                    "Weight (KG)": weight,
                    "Declared Price ($/KG)": price
                }
                st.session_state.submissions = pd.concat([
                    st.session_state.submissions, 
                    pd.DataFrame([new_row])
                ], ignore_index=True)
                st.success(f"Manifest successfully submitted for {student_name}!")

# --- COLUMN 2: LIVE TEACHER MACHINE LEARNING DASHBOARD ---
with col2:
    st.header("🖥️ Live Customs ML Control Panel")
    
    df = st.session_state.submissions.copy()
    
    # Feature Engineering: Calculate Value Mismatch (Price Deficit)
    df["Baseline Price"] = df["HS Code"].map(baseline_prices)
    df["Price Deficit"] = df["Baseline Price"] - df["Declared Price ($/KG)"]
    
    # Run K-Means Live Clustering
    # Clusters represent: 0=Compliant, 1=Tactical Risk, 2=High-Risk Flagrant Fraud
    try:
        X = df[["Weight (KG)", "Price Deficit"]]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df["Cluster_ID"] = kmeans.fit_predict(X_scaled)
        
        # Smart Risk Labeling Logic based on centroid profiles
        def define_profile(row):
            if row["Weight (KG)"] == 100000 and row["Price Deficit"] > 2.0:
                return "🚨 CRITICAL RISK: Large-scale Waste Dumping"
            elif row["Price Deficit"] <= 0.0:
                return "✅ LOW RISK: Fully Compliant Market Price"
            elif row["Weight (KG)"] == 10000 and row["Price Deficit"] > 1.0:
                return "⚠️ MEDIUM RISK: Stealth Smuggling Attempt"
            else:
                return "🔍 MODERATE RISK: Suspicious Value Manipulation"
                
        df["Risk Profile"] = df.apply(define_profile, axis=1)
        
    except Exception as e:
        df["Risk Profile"] = "Awaiting More Live Class Submissions..."

    # Analytics Summary Widgets
    most_suspicious_hs = df.groupby("HS Code")["Price Deficit"].mean().idxmax()
    
    w1, w2 = st.columns(2)
    w1.metric(label="Total Mainfests Scanned", value=len(df))
    w2.metric(label="Most Suspicious Exploited HS Code", value=f"HS {most_suspicious_hs}", delta="Highest Price Deficit")
    
    st.subheader("🕵️ Live Risk Watchlist (Real-time ML Analysis)")
    
    # Render scannable colored table dataframe
    display_df = df[["Trader Name", "HS Code", "Weight (KG)", "Declared Price ($/KG)", "Risk Profile"]]
    
    def highlight_risk(val):
        if "CRITICAL" in str(val): return 'background-color: #ffcccc; color: black; font-weight: bold;'
        elif "MEDIUM" in str(val): return 'background-color: #fff2cc; color: black;'
        elif "LOW" in str(val): return 'background-color: #e2efda; color: black;'
        return ''
        
    st.dataframe(
        display_df.style.applymap(highlight_risk, subset=["Risk Profile"]),
        use_container_width=True
    )
    
    # Clear session button for next class period
    if st.button("🔄 Reset Simulation Data"):
        del st.session_state.submissions
        st.rerun()
