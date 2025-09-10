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
    
    df.T.to_excel(writer, index=True, sheet_name='Analysis')
    writer.close()
    return output.getvalue()

# --- Data Room: Single File Uploader ---
st.markdown("### 📥 1. Data Room: Upload Your Comms Audit File")
st.info("For the 'Color Area Analysis' chart, please add columns to your file named `Electric_Green_Area_%` and `Dark_Green_Area_%`.")
comms_audit_file = st.file_uploader(
    "Upload your Comms Audit Excel File (e.g., skoda ads overview.xlsx)",
    type=["xlsx", "csv"]
)

if comms_audit_file:
    try:
        audit_df = pd.read_excel(comms_audit_file) if comms_audit_file.name.endswith('xlsx') else pd.read_csv(comms_audit_file)
        
        # Data Cleaning and Preparation
        audit_df['Spend'] = pd.to_numeric(audit_df['Spend'].astype(str).str.replace('€', '').str.replace(',', ''), errors='coerce').fillna(0)
        for col in brand_elements:
            if col not in audit_df.columns: audit_df[col] = False
            else: audit_df[col] = audit_df[col].astype(bool)
        if 'Electric_Green_Area_%' not in audit_df.columns: audit_df['Electric_Green_Area_%'] = 0.0
        if 'Dark_Green_Area_%' not in audit_df.columns: audit_df['Dark_Green_Area_%'] = 0.0

        # --- DUMMY DATA GENERATION (WITH CORRECTED LIST LENGTHS) ---
        survey_data = {
            'Element': brand_elements,
            '% recognised': [0.80, 0.47, 0.78, 0.30, 0.22, 0.52, 0.59, 0.14, 0.29],
            'Positive associations': [0.70, 0.39, 0.29, 0.45, 0.59, 0.35, 0.76, 0.33, 0.21],
            'Negative associations': [0.30, 0.30, 0.51, 0.20, 0.11, 0.46, 0.15, 0.58, 0.78],
            'Uniqueness': [0.51, 0.29, 0.90, 0.60, 0.94, 0.73, 0.46, 0.54, 0.53],
            'adj_bold': [0.6, 0.8, 0.7, 0.4, 0.5, 0.3, 0.6, 0.4, 0.2],
            'adj_stylish': [0.5, 0.7, 0.6, 0.3, 0.6, 0.2, 0.5, 0.3, 0.4],
            'adj_modern': [0.4, 0.9, 0.5, 0.2, 0.6, 0.1, 0.7, 0.2, 0.3],
        }
        research_df = pd.DataFrame(survey_data).set_index('Element')
        # -----------------------------------------------------------
        
        with st.expander("Note: Using placeholder data for Quant Research metrics. Click to see the data."):
            st.dataframe(research_df.style.format("{:.1%}"))
        
        st.success("Comms Audit file loaded successfully! The dashboard is now active.")
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["📊 Executive Summary", "❓ Strategic Deep Dive", "📄 Data Explorer"])

        with tab1:
            st.header("Executive Summary")
            
            available_markets = audit_df['Market'].unique()
            available_placements = audit_df['Placement'].unique()
            
            col1, col2 = st.columns(2)
            with col1:
                selected_market = st.selectbox("Filter by Market", options=['All'] + sorted(list(available_markets)), index=0)
            with col2:
                selected_placement = st.selectbox("Filter by Placement", options=['All'] + sorted(list(available_placements)), index=0)

            filtered_audit_df = audit_df.copy()
            if selected_market != 'All': filtered_audit_df = filtered_audit_df[filtered_audit_df['Market'] == selected_market]
            if selected_placement != 'All': filtered_audit_df = filtered_audit_df[filtered_audit_df['Placement'] == selected_placement]

            total_ads = len(filtered_audit_df)
            media_metrics = []
            for element in brand_elements:
                element_df = filtered_audit_df[filtered_audit_df[element] == True]
                media_metrics.append({'Element': element, '% Total Used': (len(element_df) / total_ads) if total_ads > 0 else 0, 'Total Investment': element_df['Spend'].sum(), 'Average Investment': element_df['Spend'].mean()})
            media_df = pd.DataFrame(media_metrics).set_index('Element')
            master_df = media_df.join(research_df)
            
            st.markdown("#### Combined Analysis Table")
            st.caption("This table synthesizes the media audit with survey data. Use the filters above to drill down into specific segments.")
            df_for_display = master_df.drop(columns=[col for col in master_df.columns if col.startswith('adj_')])
            styler = df_for_display.T.fillna(0).style
            heatmap_rows = ['% recognised', 'Positive associations', 'Negative associations', 'Uniqueness']
            styler = styler.background_gradient(cmap='RdYlGn', axis=1, subset=(pd.IndexSlice[heatmap_rows], slice(None)))
            styler = styler.background_gradient(cmap='RdYlGn_r', axis=1, subset=(pd.IndexSlice[['Negative associations']], slice(None)))
            percent_rows = ['% Total Used', '% recognised', 'Positive associations', 'Negative associations', 'Uniqueness']
            currency_rows = ['Total Investment', 'Average Investment']
            styler = styler.format("{:.1%}", subset=(pd.IndexSlice[percent_rows], slice(None)))
            styler = styler.format("€{:,.2f}", subset=(pd.IndexSlice[currency_rows], slice(None)))
            st.dataframe(styler)
            
            excel_file = to_excel(df_for_display.fillna(0))
            st.download_button(label="📥 Export Filtered Analysis to Excel", data=excel_file, file_name=f"skoda_analysis_{selected_market}_{selected_placement}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            st.markdown("#### Brand Equity Matrix")
            st.caption("This chart plots each element's Fame vs. Uniqueness. The size of the bubble represents the average spend on ads containing that element.")
            plot_data = master_df.reset_index().rename(columns={'% recognised': 'Recognition (Fame)', 'Average Investment': 'avg_spend', 'index': 'Element'})
            plot_data['avg_spend'] = plot_data['avg_spend'].fillna(0)
            fig_matrix = px.scatter(plot_data, x="Uniqueness", y="Recognition (Fame)", size="avg_spend", color="Positive associations", text="Element", size_max=60, hover_name="Element", color_continuous_scale='RdYlGn', title="Fame vs. Uniqueness (Size by Avg Spend)")
            st.plotly_chart(fig_matrix, use_container_width=True)

        with tab2:
            st.header("Strategic Deep Dive")
            st.caption("These charts answer key strategic questions based on the full, unfiltered dataset to provide a global overview.")

            total_ads_all = len(audit_df)
            media_metrics_all = []
            for element in brand_elements:
                element_df_all = audit_df[audit_df[element] == True]
                media_metrics_all.append({'Element': element, 'Total Investment': element_df_all['Spend'].sum()})
            media_df_all = pd.DataFrame(media_metrics_all).set_index('Element')
            master_df_all = media_df_all.join(research_df)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Where is our investment going?")
                investment_df = master_df_all[['Total Investment']].sort_values(by='Total Investment', ascending=True)
                fig_investment = px.bar(investment_df, x='Total Investment', y=investment_df.index, orientation='h', title="Total Spend by Brand Element", text_auto=True)
                st.plotly_chart(fig_investment, use_container_width=True)

            with col2:
                st.markdown("##### Which assets are 'safe bets' vs. 'risky'?")
                sentiment_df = master_df_all.reset_index().rename(columns={'index': 'Element'})
                fig_sentiment = px.scatter(sentiment_df, x='Negative associations', y='Positive associations', size='Total Investment', color='Element', hover_name='Element', title="Sentiment Analysis")
                st.plotly_chart(fig_sentiment, use_container_width=True)
            
            st.markdown("---")
            st.markdown("##### What is the perceived personality of our key assets?")
            st.caption("Select one or more assets to compare their positive attribute scores from the survey.")
            adj_cols = {'adj_bold': 'Bold', 'adj_stylish': 'Stylish', 'adj_modern': 'Modern'}
            personality_df = master_df_all[adj_cols.keys()].rename(columns=adj_cols).reset_index()
            personality_melted = personality_df.melt(id_vars='Element', var_name='Adjective', value_name='Score')
            selected_elements = st.multiselect("Select elements to compare:", options=brand_elements, default=brand_elements[:2])
            
            if selected_elements:
                filtered_personality = personality_melted[personality_melted['Element'].isin(selected_elements)]
                fig_personality = px.bar(filtered_personality, x="Adjective", y="Score", color="Element", barmode="group", title="Asset Personality Profile Comparison")
                st.plotly_chart(fig_personality, use_container_width=True)
            
            st.markdown("---")
            st.markdown("##### How much visual real estate do our brand colors occupy?")
            st.caption("This chart shows the distribution of the percentage of ad area covered by our key brand colors (based on data in your spreadsheet).")
            color_area_df = audit_df[['Electric_Green_Area_%', 'Dark_Green_Area_%']].melt(var_name='Color', value_name='Area %')
            color_area_df = color_area_df[color_area_df['Area %'] > 0]
            
            if not color_area_df.empty:
                fig_color_area = px.box(color_area_df, x='Color', y='Area %', color='Color', title='Distribution of Color Area Coverage in Ads', points='all')
                st.plotly_chart(fig_color_area, use_container_width=True)
            else:
                st.warning("No color area data found. Please add `Electric_Green_Area_%` and `Dark_Green_Area_%` columns to your file to enable this analysis.")

        with tab3:
            st.header("Data Explorer")
            st.caption("This section shows the raw Comms Audit data you uploaded.")
            st.dataframe(audit_df)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        st.error("Please ensure your file is a valid Excel/CSV with the expected column names.")
