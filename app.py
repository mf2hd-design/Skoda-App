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
st.set_page_config(layout="wide")
st.title("Škoda Brand Asset Strategic Analyzer")
st.markdown("This tool synthesizes your **Comms Audit** and **Quant Research** data to provide interactive strategic insights.")

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
st.header("1. Data Room: Upload Your Comms Audit File")
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
                # If a brand element column is missing from the file, create it and fill with False
                audit_df[col] = False

        # --- DUMMY DATA GENERATION ---
        st.info("Note: Using placeholder data for Quant Research metrics until the official Savanta file is provided.")
        survey_data = {
            'Element': brand_elements, 
            '% recognised': [0.80, 0.47, 0.78, 0.30, 0.22, 0.52, 0.59, 0.14, 0.29], 
            'Positive associations': [0.70, 0.39, 0.29, 0.45, 0.59, 0.35, 0.76, 0.33, 0.21], 
            'Negative associations': [0.30, 0.30, 0.51, 0.20, 0.11, 0.46, 0.15, 0.58, 0.78], 
            'Uniqueness': [0.51, 0.29, 0.90, 0.60, 0.94, 0.73, 0.46, 0.54, 0.53]
        }
        research_df = pd.DataFrame(survey_data).set_index('Element')
        # ---------------------------

        st.success("Comms Audit file loaded successfully! The dashboard is now active.")
        st.subheader("Interactive Dashboard")

        # --- Interactive Dashboard ---
        # --- Global Filters ---
        st.markdown("### Global Filters")
        
        available_markets = audit_df['Market'].unique()
        available_mediums = audit_df['Medium'].unique()
        
        selected_market = st.selectbox("Filter by Market", options=['All'] + sorted(list(available_markets)), index=0)
        selected_medium = st.selectbox("Filter by Medium", options=['All'] + sorted(list(available_mediums)), index=0)

        # Filter the DataFrame based on selections
        filtered_audit_df = audit_df.copy()
        if selected_market != 'All':
            filtered_audit_df = filtered_audit_df[filtered_audit_df['Market'] == selected_market]
        if selected_medium != 'All':
            filtered_audit_df = filtered_audit_df[filtered_audit_df['Medium'] == selected_medium]

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
        
        # Merge with research data (now the dummy data)
        master_df = media_df.join(research_df)

        # --- Display the Combined Analysis Table (Heatmap) ---
        st.markdown("#### Combined Analysis Table")
        
        # ----- THIS IS THE ROBUST FIX -----
        # 1. Transpose the DataFrame so metrics become the rows
        df_for_display = master_df.T
        
        # 2. Create the Styler object from the transposed DataFrame
        styler = df_for_display.fillna(0).style

        # 3. Define the rows for each format type (these are now the index of the DataFrame)
        heatmap_rows = ['% recognised', 'Positive associations', 'Negative associations', 'Uniqueness']
        percent_rows = ['% Total Used', '% recognised', 'Positive associations', 'Negative associations', 'Uniqueness']
        currency_rows = ['Total Investment', 'Average Investment']
        
        # 4. Apply styling and formatting
        styler = styler.background_gradient(cmap='RdYlGn', axis=1, subset=(pd.IndexSlice[heatmap_rows], slice(None)))
        styler = styler.format("{:.1%}", subset=(pd.IndexSlice[percent_rows], slice(None)))
        styler = styler.format("€{:,.2f}", subset=(pd.IndexSlice[currency_rows], slice(None)))
        
        st.dataframe(styler)
        # ---------------------------------
        
        excel_file = to_excel(master_df.fillna(0))
        st.download_button(
            label="📥 Export Filtered Analysis to Excel",
            data=excel_file,
            file_name=f"skoda_analysis_{selected_market}_{selected_medium}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # --- Display the Brand Equity Matrix ---
        st.markdown("#### Brand Equity Matrix")
        plot_data = master_df.reset_index().rename(columns={'% recognised': 'Recognition (Fame)', 'Average Investment': 'avg_spend', 'index': 'Element'})
        plot_data['avg_spend'] = plot_data['avg_spend'].fillna(0)

        fig_matrix = px.scatter(
            plot_data, x="Uniqueness", y="Recognition (Fame)", 
            size="avg_spend", color="Positive associations", 
            text="Element", size_max=60, hover_name="Element", 
            color_continuous_scale='RdYlGn', 
            title="Fame vs. Uniqueness (Size by Avg Spend)"
        )
        st.plotly_chart(fig_matrix, use_container_width=True)

        # --- ROI Proxy Chart ---
        st.markdown("#### ROI Proxy: Positive Associations per €1M Invested")
        master_df['ROI_Proxy'] = (master_df['Positive associations'] / master_df['Total Investment']) * 1_000_000
        master_df['ROI_Proxy'].fillna(0, inplace=True)
        roi_df = master_df[master_df['Total Investment'] > 0]['ROI_Proxy'].sort_values(ascending=False).reset_index()

        fig_roi = px.bar(
            roi_df,
            x='Element',
            y='ROI_Proxy',
            title="Which elements are most efficient at generating positive associations?",
            labels={'ROI_Proxy': 'Positive Association Score per €1M Invested'}
        )
        st.plotly_chart(fig_roi, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        st.error("Please ensure your file is a valid Excel/CSV with the expected column names (e.g., 'Market', 'Spend', etc.).")
