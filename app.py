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
            # ----- THIS IS THE UPDATED FILTER -----
            available_placements = audit_df['Placement'].unique()
            # ------------------------------------
            
            col1, col2 = st.columns(2)
            with col1:
                selected_market = st.selectbox("Filter by Market", options=['All'] + sorted(list(available_markets)), index=0)
            with col2:
                # ----- THIS IS THE UPDATED FILTER -----
                selected_placement = st.selectbox("Filter by Placement", options=['All'] + sorted(list(available_placements)), index=0)
                # ------------------------------------

            # Filter the DataFrame based on selections
            filtered_audit_df = audit_df.copy()
            if selected_market != 'All':
                filtered_audit_df = filtered_audit_df[filtered_audit_df['Market'] == selected_market]
            # ----- THIS IS THE UPDATED FILTER LOGIC -----
            if selected_placement != 'All':
                filtered_audit_df = filtered_audit_df[filtered_audit_df['Placement'] == selected_placement]
            # ------------------------------------------

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
            heatmap_rows = ['% recognised', 'Positive associations', 'Negative associations', 'Uniqueness']
            styler = styler.background_gradient(cmap='RdYlGn', axis=1, subset=(pd.IndexSlice[heatmap_rows], slice(None)))
            percent_rows = ['% Total Used', '% recognised', 'Positive associations', 'Negative associations', 'Uniqueness']
            currency_rows = ['Total Investment', 'Average Investment']
            styler = styler.format("{:.1%}", subset=(pd.IndexSlice[percent_rows], slice(None)))
            styler = styler.format("€{:,.2f}", subset=(pd.IndexSlice[currency_rows], slice(None)))
            st.dataframe(styler)
            
            # ----- THIS IS THE UPDATED FILENAME FOR EXPORT -----
            excel_file = to_excel(master_df.fillna(0))
            st.download_button(label="📥 Export Filtered Analysis to Excel", data=excel_file, file_name=f"skoda_analysis_{selected_market}_{selected_placement}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            # ----------------------------------------------------
            
            # --- Display the Brand Equity Matrix ---
            st.markdown("#### Brand Equity Matrix")
            st.caption("This chart plots each element's Fame vs. Uniqueness. The size of the bubble represents the average spend on ads containing that element.")
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

            # Re-calculate master_df for the full dataset for this tab
            total_ads_all = len(audit_df)
            media_metrics_all = []
            for element in brand_elements:
                element_df_all = audit_df[audit_df[element] == True]
                media_metrics_all.append({
                    'Element': element,
                    'Total Investment': element_df_all['Spend'].sum(),
                })
            media_df_all = pd.DataFrame(media_metrics_all).set_index('Element')
            master_df_all = media_df_all.join(research_df)

            # ... Rest of the charts in this tab are the same as before ...
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("##### Where is our investment going?")
                investment_df = master_df_all[['Total Investment']].sort_values(by='Total Investment', ascending=True)
                fig_investment = px.bar(investment_df, x='Total Investment', y=investment_df.index, orientation='h', title="Total Spend by Brand Element", text_auto=True)
                st.plotly_chart(fig_investment, use_container_width=True)

            with col4:
                st.markdown("##### Which assets are 'safe bets' vs. 'risky'?")
                sentiment_df = master_df_all.reset_index().rename(columns={'index': 'Element'})
                fig_sentiment = px.scatter(sentiment_df, x='Negative associations', y='Positive associations', size='Total Investment', color='Element', hover_name='Element', title="Sentiment Analysis")
                st.plotly_chart(fig_sentiment, use_container_width=True)
            
            col5, col6 = st.columns(2)
            with col5:
                st.markdown("##### Are elements used consistently across markets?")
                market_usage = audit_df.groupby('Market')[brand_elements].mean().reset_index()
                market_usage_melted = market_usage.melt(id_vars='Market', value_vars=brand_elements, var_name='Element', value_name='% Used')
                fig_market = px.bar(market_usage_melted, x='Element', y='% Used', color='Market', barmode='group', title='Brand Element Usage by Market')
                st.plotly_chart(fig_market, use_container_width=True)

            with col6:
                st.markdown("##### Are assets used effectively across media types?")
                media_usage = audit_df.groupby('Medium')[brand_elements].mean().T
                fig_media_heatmap = px.imshow(media_usage, labels=dict(x="Medium Type", y="Brand Element", color="% Used"), text_auto=True, aspect="auto", title="Element Usage Frequency by Media Type")
                st.plotly_chart(fig_media_heatmap, use_container_width=True)

            st.markdown("---")
            col7, col8 = st.columns(2)
            with col7:
                st.markdown("##### Which assets are most efficient at driving Recognition?")
                master_df_all['Recognition_ROI'] = (master_df_all['% recognised'] / master_df_all['Total Investment']) * 1_000_000
                master_df_all['Recognition_ROI'].fillna(0, inplace=True)
                recognition_roi_df = master_df_all[master_df_all['Total Investment'] > 0]['Recognition_ROI'].sort_values(ascending=False).reset_index()
                fig_recognition_roi = px.bar(recognition_roi_df, x='Element', y='Recognition_ROI', title="Recognition Efficiency", labels={'Recognition_ROI': 'Recognition % Points per €1M Invested'})
                st.plotly_chart(fig_recognition_roi, use_container_width=True)

            with col8:
                st.markdown("##### Are we investing in our most unique assets?")
                uniqueness_df = master_df_all.reset_index().rename(columns={'index': 'Element'})
                fig_uniqueness_spend = px.scatter(
                    uniqueness_df, 
                    x='Uniqueness', 
                    y='Total Investment', 
                    size='Positive associations',
                    color='Element',
                    hover_name='Element',
                    title='Investment vs. Uniqueness'
                )
                st.plotly_chart(fig_uniqueness_spend, use_container_width=True)

        with tab3:
            st.header("Data Explorer")
            st.caption("This section shows the raw Comms Audit data you uploaded.")
            st.dataframe(audit_df)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        st.error("Please ensure your file is a valid Excel/CSV with the expected column names (e.g., 'Market', 'Spend', etc.).")
