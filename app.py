import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# --- Define Brand Elements based on your provided data ---
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
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=True, sheet_name='Analysis')
    writer.close()
    processed_data = output.getvalue()
    return processed_data

# --- Data Room: File Uploaders ---
st.header("1. Data Room: Upload Your Files")
col1, col2 = st.columns(2)

with col1:
    comms_audit_file = st.file_uploader(
        "Upload your Comms Audit Excel File",
        type=["xlsx", "csv"]
    )

with col2:
    quant_research_file = st.file_uploader(
        "Upload your Savanta Quant Research Excel File",
        type=["xlsx", "csv"]
    )

if comms_audit_file and quant_research_file:
    # --- Calculation Engine: Process and merge the data ---
    try:
        # Load Comms Audit data
        audit_df = pd.read_excel(comms_audit_file) if comms_audit_file.name.endswith('xlsx') else pd.read_csv(comms_audit_file)
        
        # Data Cleaning for the Audit file
        audit_df['Spend'] = pd.to_numeric(audit_df['Spend'].astype(str).str.replace('€', '').str.replace(',', ''), errors='coerce').fillna(0)
        for col in brand_elements:
            if col in audit_df.columns:
                audit_df[col] = audit_df[col].astype(bool)

        # Load Quant Research data
        research_df = pd.read_excel(quant_research_file) if quant_research_file.name.endswith('xlsx') else pd.read_csv(quant_research_file)
        # We'll assume the research file has a column named 'Element' and the survey metrics
        research_df.set_index('Element', inplace=True)

        st.success("Files loaded successfully! The dashboard is now active.")
        st.subheader("Interactive Dashboard")

        # --- Interactive Dashboard ---
        # --- Global Filters ---
        st.markdown("### Global Filters")
        
        # Get unique values for filters from the audit data
        available_markets = audit_df['Market'].unique()
        available_mediums = audit_df['Medium'].unique()
        
        selected_market = st.selectbox("Filter by Market (select one or more)", options=['All'] + list(available_markets), index=0)
        selected_medium = st.selectbox("Filter by Medium", options=['All'] + list(available_mediums), index=0)

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
        
        # Merge with research data
        master_df = media_df.join(research_df)

        # --- Display the Combined Analysis Table (Heatmap) ---
        st.markdown("#### Combined Analysis Table")
        styler = master_df.fillna(0).style
        heatmap_rows = ['% recognised', 'Positive associations', 'Negative associations', 'Uniqueness']
        styler = styler.background_gradient(cmap='RdYlGn', axis=1, subset=(pd.IndexSlice[heatmap_rows], slice(None)))
        styler = styler.format("{:.1%}", subset=pd.IndexSlice[['% Total Used'] + heatmap_rows, :])
        styler = styler.format("€{:,.2f}", subset=pd.IndexSlice[['Total Investment', 'Average Investment'], :])
        st.dataframe(styler)
        
        excel_file = to_excel(master_df.fillna(0))
        st.download_button(
            label="📥 Export Filtered Analysis to Excel",
            data=excel_file,
            file_name=f"skoda_analysis_{selected_market}_{selected_medium}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # --- Display the Brand Equity Matrix ---
        st.markdown("#### Brand Equity Matrix")
        plot_data = master_df.reset_index().rename(columns={'% recognised': 'Recognition (Fame)', 'Average Investment': 'avg_spend'})
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
        # Calculate the ROI proxy metric, handle division by zero
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
        st.error(f"An error occurred while processing the files: {e}")
        st.error("Please ensure your files are in the correct format with the expected column names.")
