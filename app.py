import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# --- 1. FINALIZED BRAND ELEMENTS (from Savanta survey) ---
brand_elements = [
    "Emerald Green", "Electric Green", "Skoda Symbol", "Hacek", 
    "Facets", "Sonic tag", "Tagline: Let’s Explore"
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
comms_audit_file = st.file_uploader(
    "Upload your Comms Audit Excel File (e.g., skoda ads overview.xlsx)",
    type=["xlsx", "csv"]
)

if comms_audit_file:
    try:
        audit_df = pd.read_excel(comms_audit_file) if comms_audit_file.name.endswith('xlsx') else pd.read_csv(comms_audit_file)
        
        audit_df['Spend'] = pd.to_numeric(audit_df['Spend'].astype(str).str.replace('€', '').str.replace(',', ''), errors='coerce').fillna(0)
        # Rename survey columns to match our cleaner list
        column_rename_map = {"Emerald Green": "Emerald Green", "Electric Green": "Electric Green", "Symbol": "Skoda Symbol", "Hacek": "Hacek", "Facets": "Facets", "Sonic": "Sonic tag", "Tagline": "Tagline: Let’s Explore"}
        audit_df.rename(columns=column_rename_map, inplace=True)

        for col in brand_elements:
            if col not in audit_df.columns: audit_df[col] = False
            else: audit_df[col] = audit_df[col].astype(bool)

        # --- DUMMY DATA GENERATION (Now updated with new metrics) ---
        survey_data = {
            'Element': brand_elements, 
            '% recognised': [0.47, 0.80, 0.22, 0.52, 0.14, 0.29, 0.30], 
            'Positive associations': [0.39, 0.70, 0.59, 0.35, 0.33, 0.21, 0.45], 
            'Negative associations': [0.30, 0.30, 0.11, 0.46, 0.58, 0.78, 0.20], 
            'Uniqueness': [0.29, 0.51, 0.94, 0.73, 0.54, 0.53, 0.60],
            # 2. NEW DUMMY DATA for Personality Profile
            'adj_bold': [0.6, 0.8, 0.7, 0.4, 0.5, 0.3, 0.6],
            'adj_stylish': [0.5, 0.7, 0.6, 0.3, 0.6, 0.2, 0.5],
            'adj_modern': [0.4, 0.9, 0.5, 0.2, 0.6, 0.1, 0.7],
            'adj_playful': [0.3, 0.6, 0.4, 0.5, 0.4, 0.4, 0.5],
            'adj_adventurous': [0.4, 0.7, 0.6, 0.6, 0.5, 0.3, 0.8],
            # 3. NEW DUMMY DATA for Strength of Link
            'strength_rank': [4, 2, 1, 3, 5, 7, 6] # 1 is strongest, 7 is weakest
        }
        research_df = pd.DataFrame(survey_data).set_index('Element')
        
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
            
            df_for_display = master_df.drop(columns=[col for col in master_df.columns if col.startswith('adj_') or col == 'strength_rank'])
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

            # --- NEW CHART 1: Asset Personality Profile ---
            st.markdown("##### What is the perceived personality of our key assets?")
            st.caption("Select one or more assets to compare their positive attribute scores from the survey.")
            
            adj_cols = {
                'adj_bold': 'Bold', 'adj_stylish': 'Stylish', 'adj_modern': 'Modern', 
                'adj_playful': 'Playful', 'adj_adventurous': 'Adventurous'
            }
            personality_df = master_df_all[adj_cols.keys()].rename(columns=adj_cols).reset_index()
            personality_melted = personality_df.melt(id_vars='Element', var_name='Adjective', value_name='Score')
            
            selected_elements = st.multiselect("Select elements to compare:", options=brand_elements, default=brand_elements[:2])
            
            if selected_elements:
                filtered_personality = personality_melted[personality_melted['Element'].isin(selected_elements)]
                fig_personality = px.bar(
                    filtered_personality, 
                    x="Adjective", 
                    y="Score", 
                    color="Element", 
                    barmode="group",
                    title="Asset Personality Profile Comparison"
                )
                st.plotly_chart(fig_personality, use_container_width=True)

            # --- NEW CHART 2: Strength of Link ---
            st.markdown("##### Which assets are most strongly linked to the Škoda brand?")
            st.caption("This chart ranks the elements based on which ones consumers feel are most strongly 'Škoda'.")
            
            strength_df = master_df_all[['strength_rank']].sort_values(by='strength_rank', ascending=True)
            fig_strength = px.bar(
                strength_df,
                y=strength_df.index,
                x='strength_rank',
                orientation='h',
                title="Strength of Link to Škoda (Lower Rank is Stronger)",
                labels={'strength_rank': 'Strength Rank (1=Most Linked)', 'y': 'Brand Element'},
                text='strength_rank'
            )
            fig_strength.update_layout(xaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_strength, use_container_width=True)

            # ... Other charts from before ...
        
        with tab3:
            st.header("Data Explorer")
            st.caption("This section shows the raw Comms Audit data you uploaded.")
            st.dataframe(audit_df)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        st.error("Please ensure your file is a valid Excel/CSV with the expected column names.")
