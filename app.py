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

# --- Expanded Legitimate Baseline Data Reference ---
baseline_prices = {
    390110: 4.20,  # Virgin Polyethylene
    390210: 4.50,  # Virgin Polypropylene
    390319: 3.80,  # Virgin Polystyrene
    390410: 3.90,  # Virgin PVC
    390729: 5.20,  # Specialty Polyether Resins
    392020: 4.80,  # PP Plates/Sheets
    392690: 6.50   # Technical Finished Plastics (Catch-All Category)
}

hs_descriptions = {
    390110: "Premium Virgin Polyethylene",
    390210: "Premium Virgin Polypropylene",
    390319: "Standard Virgin Polystyrene",
    390410: "Virgin PVC Industrial Resin",
    390729: "Specialty Polyether Resins",
    392020: "Rigid PP Plates & Sheets",
    392690: "Finished Technical Plastic Articles"
}

# --- GLOBAL SHARED DATABASE MANAGER ---
class GlobalDataManager:
    def __init__(self):
        self.df = pd.DataFrame(columns=[
            "Trader Name", "HS Code", "Weight (KG)", "Declared Price ($/KG)"
        ])
    
    def add_submission(self, row):
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        
    def clear_database(self):
        self.df = pd.DataFrame(columns=[
            "Trader Name", "HS Code", "Weight (KG)", "Declared Price ($/KG)"
        ])

@st.cache_resource
def get_global_database():
    return GlobalDataManager()

# Initialize global shared cloud database connection
db = get_global_database()

# Initialize local session state for the reveal mechanism toggle
if "reveal_results" not in st.session_state:
    st.session_state.reveal_results = False

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
            options=list(baseline_prices.keys()),
            format_func=lambda x: f"HS {x} ({hs_descriptions[x]})"
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
                db.add_submission(new_row)
                st.success(f"Manifest successfully submitted for {student_name}!")

# --- COLUMN 2: LIVE TEACHER MACHINE LEARNING DASHBOARD ---
with col2:
    st.header("🖥️ Live Customs ML Control Panel")
    
    # Action Deck Buttons
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("🔄 Sync Incoming Submissions", type="primary", use_container_width=True):
            st.rerun()
    with c_btn2:
        if st.session_state.reveal_results:
            if st.button("🔒 Hide Results From Class", use_container_width=True):
                st.session_state.reveal_results = False
                st.rerun()
        else:
            if st.button("🔓 Reveal Results to Class", use_container_width=True):
                st.session_state.reveal_results = True
                st.rerun()

    df = db.df.copy()
    total_manifests = len(df)
    
    # Core calculations run in background to keep data structured
    if not df.empty:
        df["Baseline Price"] = df["HS Code"].map(baseline_prices)
        df["Price Deficit"] = df["Baseline Price"] - df["Declared Price ($/KG)"]
        
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
        
        df["Risk Profile"] = df.apply(define_profile, axis=1)

    # STATE A: RESULTS HIDDEN
    if not st.session_state.reveal_results:
        st.info(f"📥 Neutral Data Collection Mode Active. Total manifests captured silently: **{total_manifests}**.")
        
    # STATE B: REVEAL UNLOCKED
    else:
        if not df.empty:
            # Multi-HS Code Metric Ranking Panel
            st.subheader("📊 Suspect Code Cluster Deviations")
            
            grouped_deficits = df.groupby("HS Code")["Price Deficit"].mean().sort_values(ascending=False)
            
            metric_cols = st.columns(min(len(grouped_deficits), 4))
            for idx, (code, avg_def) in enumerate(grouped_deficits.items()):
                if idx < len(metric_cols):
                    metric_cols[idx].metric(
                        label=f"Exploited HS {code}", 
                        value=f"${avg_def:.2f} Deficit",
                        delta="High Value Deficit", 
                        delta_color="inverse"
                    )
            
            # --- REAL-TIME MULTI-CODE PRICE VISUALIZATION ---
            st.subheader("📈 Declared Price Contamination vs. Real Market Value")
            
            chart_data = []
            for code in baseline_prices.keys():
                sub_df = df[df["HS Code"] == code]
                avg_dec = sub_df["Declared Price ($/KG)"].mean() if not sub_df.empty else 0
                chart_data.append({
                    "HS Code": f"HS {code}",
                    "Students Declared Avg": avg_dec,
                    "True Market Price": baseline_prices[code]
                })
            
            chart_df = pd.DataFrame(chart_data).set_index("HS Code")
            st.bar_chart(chart_df, use_container_width=True)
            
            # --- AUTOMATED REGULATORY COMMENTARY BLOCK ---
            st.subheader("🤖 Automated Customs Fraud Intelligence Reports")
            
            has_alerts = False
            for code in baseline_prices.keys():
                sub_df = df[df["HS Code"] == code]
                if not sub_df.empty:
                    avg_def = sub_df["Price Deficit"].mean()
                    total_vol = sub_df["Weight (KG)"].sum()
                    exploiters_count = len(sub_df)
                    
                    if avg_def > 2.0 and total_vol >= 100000:
                        has_alerts = True
                        st.error(f"""
                        **🔴 CRITICAL ANOMALY IN CODE: HS {code} ({hs_descriptions[code]})**
                        * **Fraud Signature:** High-Volume Price Contamination.
                        * **Analysis:** {exploiters_count} trade entities are dumping large volumes ({total_vol:,} KG) at artificial, scrap-level prices (Avg Deficit: ${avg_def:.2f}/KG). This matches the inverse price-volume signature—where falling unit values run parallel to volume spikes, exposing calculated evasion of the solid waste environmental tax.
                        """)
                    
                    elif avg_def > 1.0:
                        has_alerts = True
                        st.warning(f"""
                        **🟡 SUSPICIOUS ACTIVITY IN CODE: HS {code} ({hs_descriptions[code]})**
                        * **Fraud Signature:** Tactical Value Manipulation.
                        * **Analysis:** Declared import entries are consistently tracking below premium resin index baselines. While cargo volume stays discrete to prevent immediate border inspections, this pattern suggests deliberate under-invoicing or hidden contamination vectors.
                        """)
            
            if not has_alerts:
                st.success("🟢 **System Clean:** No definitive price contamination signatures detected across tracked HS families yet.")
                
            # --- FULL LIVE RISK TABLE WATCHLIST ---
            st.subheader("🕵️ Live Risk Watchlist (Real-time ML Analysis)")
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
            st.info("The database is currently neutral. Awaiting student submissions before analysis can be unlocked.")

    # --- CAMOUFLAGED TEACHER RESET KEY (Hidden inside the dot expander) ---
    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
    with st.expander("·", expanded=False):
        if st.button("🗑️ Reset All Global Data"):
            db.clear_database()
            st.session_state.reveal_results = False
            st.rerun()
