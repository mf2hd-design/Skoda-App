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
        st.subheader("Interactive Dashboard")

        # --- Interactive Dashboard ---
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
        
        master_df = media_df.join(research_df)

        # --- Display the Combined Analysis Table (Heatmap) ---
        st.markdown("#### Combined Analysis Table")
        
        df_for_display = master_df.T
        styler = df_for_display.fillna(0).style

        heatmap_rows = ['% recognised', 'Positive associations', 'Negative associations', 'Uniqueness']
        styler = styler.background_gradient(cmap='RdYlGn', axis=1, subset=(pd.IndexSlice[heatmap_rows], slice(None)))
        
        percent_rows = ['% Total Used', '% recognised', 'Positive associations', 'Negative associations', 'Uniqueness']
        currency_rows = ['Total Investment', 'Average Investment']
        
        styler = styler.format("{:.1%}", subset=(pd.IndexSlice[percent_rows], slice(None)))
        styler = styler.format("€{:,.2f}", subset=(pd.IndexSlice[currency_rows], slice(None)))
        
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

        # --- NEW SECTION: ANSWERING KEY STRATEGIC QUESTIONS ---
        st.header("Answering Key Strategic Questions")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Where is our investment going?")
            investment_df = master_df[['Total Investment']].sort_values(by='Total Investment', ascending=True)
            fig_investment = px.bar(
                investment_df,
                x='Total Investment',
                y=investment_df.index,
                orientation='h',
                title="Total Spend by Brand Element",
                text_auto=True
            )
            st.plotly_chart(fig_investment, use_container_width=True)

        with col2:
            st.markdown("##### Which assets are 'safe bets' vs. 'risky'?")
            sentiment_df = master_df.reset_index().rename(columns={'index': 'Element'})
            fig_sentiment = px.scatter(
                sentiment_df,
                x='Negative associations',
                y='Positive associations',
                size='Total Investment',
                color='Element',
                hover_name='Element',
                title="Sentiment Analysis: Positive vs. Negative Associations"
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)

        st.markdown("##### Which assets are most efficient at driving Recognition (Fame)?")
        master_df['Recognition_ROI'] = (master_df['% recognised'] / master_df['Total Investment']) * 1_000_000
        master_df['Recognition_ROI'].fillna(0, inplace=True)
        recognition_roi_df = master_df[master_df['Total Investment'] > 0]['Recognition_ROI'].sort_values(ascending=False).reset_index()
        
        fig_recognition_roi = px.bar(
            recognition_roi_df,
            x='Element',
            y='Recognition_ROI',
            title="Recognition Efficiency",
            labels={'Recognition_ROI': 'Recognition % Points per €1M Invested'}
        )
        st.plotly_chart(fig_recognition_roi, use_container_width=True)
        # --------------------------------------------------------

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        st.error("Please ensure your file is a valid Excel/CSV with the expected column names (e.g., 'Market', 'Spend', etc.).")
