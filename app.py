import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# --- Define Brand Elements based on the Comms Audit spreadsheet ---
brand_elements = [
    "Electric Green", "Dark Green", "Type", "Tagline", "Symbol",
    "Hacek", "Wordmark", "Facets", "Sonic"
]

# --- App UI and Logic ---
st.set_page_config(
    layout="wide",
    page_title="Škoda Brand Intelligence Dashboard",
    page_icon="📊"
)

# --- A more professional title and introduction ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Škoda Brand Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("This strategic tool synthesizes **Comms Audit** and **Quant Research** data to provide interactive insights into the performance and equity of Škoda's key brand assets.")

# --- Helper function to create a downloadable Excel file ---
def to_excel(df):
    output = BytesIO()
    try:
        import xlsxwriter
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
    except ImportError:
        writer = pd.ExcelWriter(output, engine='openpyxl')
    
    # Write the transposed data to Excel to match the screen display
    df.T.to_excel(writer, index=True, sheet_name='Analysis')
    writer.close()
    processed_data = output.getvalue()
    return processed_data

# --- Data Room: Single File Uploader ---
st.markdown("### 📥 1. Data Room: Upload Your Comms Audit File")
comms_audit_file = st.file_uploader(
    "Upload your Comms Audit Excel File (e.g., skoda ads overview.xlsx)",
    type=["xlsx", "csv"]
)

if comms_audit_file:
    # --- Calculation Engine: Process and merge the data ---
    try:
        # Load Comms Audit data
        audit_df = pd.read_excel(comms_audit_file) if comms_audit_file.name.endswith('xlsx') else pd.read_csv(comms_audit_file)
        
        # Data Cleaning for the Audit file
        audit_df['Spend'] = pd.to_numeric(audit_df['Spend'].astype(str).str.replace('€', '').str.replace(',', ''), errors='coerce').fillna(0)
        for col in brand_elements:
            if col in audit_df.columns:
                audit_df[col] = audit_df[col].astype(bool)
            else:
                audit_df[col] = False

        # --- DUMMY DATA GENERATION (CLICKABLE) ---
        survey_data = {
            'Element': brand_elements, 
            '% recognised': [0.80, 0.47, 0.78, 0.30, 0.22, 0.52, 0.59, 0.14, 0.29], 
            'Positive associations': [0.70, 0.39, 0.29, 0.45, 0.59, 0.35, 0.76, 0.33, 0.21], 
            'Negative associations': [0.30, 0.30, 0.51, 0.20, 0.11, 0.46, 0.15, 0.58, 0.78], 
            'Uniqueness': [0.51, 0.29, 0.90, 0.60, 0.94, 0.73, 0.46, 0.54, 0.53]
        }
        research_df = pd.DataFrame(survey_data).set_index('Element')
        
        with st.expander("Note: Using placeholder data for Quant Research metrics. Click to see the data."):
            st.warning("The data below is for demonstration purposes until the official Savanta file is provided.")
            st.dataframe(research_df.style.format("{:.1%}"))
        
        st.success("Comms Audit file loaded successfully! The dashboard is now active.")
        st.markdown("---")
        
        # --- Create Tabs for a more professional UI ---
        tab1, tab2, tab3 = st.tabs(["📊 Executive Summary", "❓ Strategic Deep Dive", "📄 Data Explorer"])

        with tab1:
            st.header("Executive Summary")
            
            # --- Global Filters ---
            st.markdown("Use these filters to explore the data for specific markets or placements.")
            
            available_markets = audit_df['Market'].unique()
            available_placements = audit_df['Placement'].unique()
            
            col1, col2 = st.columns(2)
            with col1:
                selected_market = st.selectbox("Filter by Market", options=['All'] + sorted(list(available_markets)), index=0)
            with col2:
                selected_placement = st.selectbox("Filter by Placement", options=['All'] + sorted(list(available_placements)), index=0)

            # Filter the DataFrame based on selections
            filtered_audit_df = audit_df.copy()
            if selected_market != 'All':
                filtered_audit_df = filtered_audit_df[filtered_audit_df['Market'] == selected_market]
            if selected_placement != 'All':
                filtered_audit_df = filtered_audit_df[filtered_audit_df['Placement'] == selected_placement]

            # --- Perform Calculations on the Filtered Data ---
            total_ads = len(filtered_audit_df)
            media_metrics = []
            for element in brand_elements:
                element_df = filtered_audit_df[filtered_audit_df[element] == True]
                media_metrics.append({
                    'Element': element,
                    '% Total Used': (len(element_df) / total_ads) if total_ads > 0 else 0,
                    'Total Investment': element_df['Spend'].sum(),
                    'Average Investment': element_df['Spend'].mean()
                })
            media_df = pd.DataFrame(media_metrics).set_index('Element')
            
            master_df = media_df.join(research_df)

            # --- Display the Combined Analysis Table (Heatmap) ---
            st.markdown("#### Combined Analysis Table")
            st.caption("This table synthesizes the media audit with survey data. Use the filters above to drill down into specific segments.")
            
            df_for_display = master_df.T
            styler = df_for_display.fillna(0).style

            # ----- THIS IS THE ROBUST HEATMAP FIX -----
            # Define which rows are "good" (high is green) and "bad" (low is green)
            good_metric_rows = ['% recognised', 'Positive associations', 'Uniqueness']
            bad_metric_rows = ['Negative associations']

            # Apply the normal colormap to the "good" metrics
            styler = styler.background_gradient(cmap='RdYlGn', axis=1, subset=(pd.IndexSlice[good_metric_rows], slice(None)))
            # Apply the REVERSED colormap to the "bad" metrics
            styler = styler.background_gradient(cmap='RdYlGn_r', axis=1, subset=(pd.IndexSlice[bad_metric_rows], slice(None)))
            # -------------------------------------------
            
    
