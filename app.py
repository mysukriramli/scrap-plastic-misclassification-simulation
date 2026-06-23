import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set page layout to wide for dashboard look
st.set_page_config(layout="wide")

# The exact logo text and context headers you wanted to protect
st.title("🚨 Live Trade Fraud Simulation: The Scrap Plastic Smuggler Game")
st.markdown("""
    **Classroom Context:** Students act as 'bad actors' trying to bypass customs by hiding low-value plastic scrap 
    under premium virgin plastic HS codes. The ML pipeline on the right detects their patterns live!
""")

# --- Legitimate Baseline Data Reference ---
baseline_prices = {
    390110: 4.20,  # Virgin Polyethylene
    390210: 4.50,  # Virgin Polypropylene
    390410: 3.90   # Virgin PVC
}

# --- Persistent Data Storage (Session State) ---
# NOW STARTS COMPLETELY EMPTY AND NEUTRAL
if "submissions" not in st.session_state:
    st.session_state.submissions = pd.DataFrame(columns=[
        "Trader Name", "HS Code", "Weight (KG)", "Declared Price ($/KG)"
    ])

# Layout Split: 1/3 Student Form, 2/3 Teacher Dashboard
col1, col2 = st.columns([1, 2])

# --- COLUMN 1: STUDENT INPUT INTERFACE & QR CODE ---
with col1:
    st.header("📥 Student Submission Portal")
    
    st.markdown("### 📲 Scan to Join the Game Live!")
    app_url = "https://scrap-plastic-misclassification-simulation-jghvmoc7yc3tvmrhyul.streamlit.app/"
    qr_api_url = f"https://api.qrserver.com/v1/create-qr-code/?size=200x200&data={app_url}"
    st.image(qr_api_url, caption="Scan with your phone camera to play", width=200)
    st.markdown("---")
    
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
        
        submit_button = st.form_submit_button(label="🚀 Submit Manifest to Customs")
        
        if submit_button:
            if student_name.strip() == "":
                st.error("Please enter a name or alias before submitting!")
            else:
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
    
    # Custom profile rules engine that works cleanly for empty or growing datasets
    def define_profile(row):
        base_p = baseline_prices.get(int(row["HS Code"]), 4.0)
        deficit = base_p - float(row["Declared Price ($/KG)"])
        if int(row["Weight (KG)"]) == 100000 and deficit > 2.0:
            return "🚨 CRITICAL RISK: Large-scale Waste Dumping"
        elif deficit <= 0.0:
            return "✅ LOW RISK: Fully Compliant Market Price"
        elif int(row["Weight (KG)"]) == 10000 and deficit > 1.0:
            return "⚠️ MEDIUM RISK: Stealth Smuggling Attempt"
        else:
            return "🔍 MODERATE RISK: Suspicious Value Manipulation"

    # Process metrics and ML calculations dynamically
    if not df.empty:
        df["Baseline Price"] = df["HS Code"].map(baseline_prices)
        df["Price Deficit"] = df["Baseline Price"] - df["Declared Price ($/KG)"]
        
        # ML K-Means starts tracking automatically once 3 or more manifests are logged
        if len(df) >= 3:
            try:
                X = df[["Weight (KG)", "Price Deficit"]]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                df["Cluster_ID"] = kmeans.fit_predict(X_scaled)
            except:
                pass
                
        df["Risk Profile"] = df.apply(define_profile, axis=1)
        most_suspicious_hs = f"HS {df.groupby('HS Code')['Price Deficit'].mean().idxmax()}"
        total_manifests = len(df)
    else:
        most_suspicious_hs = "None"
        total_manifests = 0

    # Display Metrics Panel
    w1, w2 = st.columns(2)
    w1.metric(label="Total Manifests Scanned", value=total_manifests)
    w2.metric(label="Most Suspicious Exploited HS Code", value=most_suspicious_hs, 
              delta="Highest Price Deficit" if total_manifests > 0 else None)
    
    st.subheader("🕵️ Live Risk Watchlist (Real-time ML Analysis)")
    
    # Display table logic
    if not df.empty:
        display_df = df[["Trader Name", "HS Code", "Weight (KG)", "Declared Price ($/KG)", "Risk Profile"]]
        
        def highlight_risk(val):
            if "CRITICAL" in str(val): return 'background-color: #ffcccc; color: black; font-weight: bold;'
            elif "MEDIUM" in str(val): return 'background-color: #fff2cc; color: black;'
            elif "LOW" in str(val): return 'background-color: #e2efda; color: black;'
            return ''
            
        st.dataframe(
            display_df.style.map(highlight_risk, subset=["Risk Profile"]),
            use_container_width=True
        )
    else:
        st.info("Awaiting the first digital customs manifest submission from students...")

    # --- CAMOUFLAGED TEACHER RESET KEY (ONLY YOU KNOW) ---
    # Creates vertical separation and hides inside a tiny dot expander at the very bottom
    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
    with st.expander("·", expanded=False):
        if st.button("🔄 Reset Simulation Data"):
            del st.session_state.submissions
            st.rerun()
