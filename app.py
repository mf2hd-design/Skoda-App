import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# --- Define Brand Elements ---
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
st.markdown("This strategic tool combines **Comms Audit** media data with **Quantitative Research** insights (P045556 - Saffron Brand Assets) to provide comprehensive analysis of Škoda's brand elements.")

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

# --- RESEARCH DATA FROM P045556_ALL_Tables_20251016_Private.xlsx ---
# This data is integrated and always available
survey_data = {
    'Element': brand_elements,
    '% recognised': [0.38, 0.39, 0.37, 0.36, 0.64, 0.38, 0.44, 0.38, 0.4],
    'Positive associations': [0.47, 0.49, 0.46, 0.48, 0.49, 0.45, 0.48, 0.47, 0.5],
    'Negative associations': [0.24, 0.22, 0.24, 0.22, 0.22, 0.25, 0.23, 0.23, 0.22],
    'Uniqueness': [0.62, 0.61, 0.63, 0.64, 0.36, 0.62, 0.56, 0.62, 0.6],
    'adj_bold': [0.49, 0.51, 0.47, 0.48, 0.5, 0.46, 0.49, 0.5, 0.5],
    'adj_stylish': [0.46, 0.49, 0.47, 0.48, 0.5, 0.46, 0.49, 0.48, 0.49],
    'adj_modern': [0.5, 0.52, 0.49, 0.51, 0.55, 0.49, 0.54, 0.51, 0.55],
}
research_df = pd.DataFrame(survey_data).set_index('Element')

# --- Comms Audit File Upload ---
st.markdown("### 📥 Upload Your Comms Audit File")
st.info("**Research data is already integrated** from P045556 study. Upload your Comms Audit file to analyze media spend, brand element usage, and markets.")

comms_audit_file = st.file_uploader(
    "Upload your Comms Audit Excel/CSV File (e.g., skoda ads overview.xlsx)",
    type=["xlsx", "csv"]
)

if comms_audit_file:
    try:
        # Load the Comms Audit data
        audit_df = pd.read_excel(comms_audit_file) if comms_audit_file.name.endswith('xlsx') else pd.read_csv(comms_audit_file)

        # Data Cleaning and Preparation
        audit_df['Spend'] = pd.to_numeric(audit_df['Spend'].astype(str).str.replace('€', '').str.replace(',', ''), errors='coerce').fillna(0)
        for col in brand_elements:
            if col not in audit_df.columns:
                audit_df[col] = False
            else:
                audit_df[col] = audit_df[col].astype(bool)

        # Optional color area columns
        if 'Electric_Green_Area_%' not in audit_df.columns:
            audit_df['Electric_Green_Area_%'] = 0.0
        if 'Dark_Green_Area_%' not in audit_df.columns:
            audit_df['Dark_Green_Area_%'] = 0.0

        st.success("✓ Comms Audit file loaded successfully! The dashboard is now active.")

        with st.expander("View Quantitative Research Data (P045556 - Saffron Brand Assets)"):
            st.dataframe(research_df.style.format("{:.1%}"))

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["📊 Executive Summary", "❓ Strategic Deep Dive", "📄 Data Explorer"])

        with tab1:
            st.header("Executive Summary")

            # Filters
            available_markets = audit_df['Market'].unique()
            available_placements = audit_df['Placement'].unique()

            col1, col2 = st.columns(2)
            with col1:
                selected_market = st.selectbox("Filter by Market", options=['All'] + sorted(list(available_markets)), index=0)
            with col2:
                selected_placement = st.selectbox("Filter by Placement", options=['All'] + sorted(list(available_placements)), index=0)

            # Apply filters
            filtered_audit_df = audit_df.copy()
            if selected_market != 'All':
                filtered_audit_df = filtered_audit_df[filtered_audit_df['Market'] == selected_market]
            if selected_placement != 'All':
                filtered_audit_df = filtered_audit_df[filtered_audit_df['Placement'] == selected_placement]

            # Calculate media metrics
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

            # Combine with research data
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
            st.download_button(
                label="📥 Export Filtered Analysis to Excel",
                data=excel_file,
                file_name=f"skoda_analysis_{selected_market}_{selected_placement}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.markdown("#### Brand Equity Matrix")
            st.caption("This chart plots each element's Fame vs. Uniqueness. The size of the bubble represents the average spend on ads containing that element.")

            plot_data = master_df.reset_index().rename(columns={'% recognised': 'Recognition (Fame)', 'Average Investment': 'avg_spend', 'index': 'Element'})
            plot_data['avg_spend'] = plot_data['avg_spend'].fillna(0)

            fig_matrix = px.scatter(
                plot_data,
                x="Uniqueness",
                y="Recognition (Fame)",
                size="avg_spend",
                color="Positive associations",
                text="Element",
                size_max=60,
                hover_name="Element",
                color_continuous_scale='RdYlGn',
                title="Fame vs. Uniqueness (Size by Avg Spend)"
            )
            st.plotly_chart(fig_matrix, use_container_width=True)

        with tab2:
            st.header("Strategic Deep Dive")
            st.caption("These charts answer key strategic questions based on the full, unfiltered dataset to provide a global overview.")

            # Calculate metrics for all data (unfiltered)
            total_ads_all = len(audit_df)
            media_metrics_all = []
            for element in brand_elements:
                element_df_all = audit_df[audit_df[element] == True]
                media_metrics_all.append({
                    'Element': element,
                    'Total Investment': element_df_all['Spend'].sum()
                })
            media_df_all = pd.DataFrame(media_metrics_all).set_index('Element')
            master_df_all = media_df_all.join(research_df)

            st.markdown("##### Investment & Sentiment Analysis")
            col1, col2 = st.columns(2)

            with col1:
                st.caption("Where is our investment going?")
                investment_df = master_df_all[['Total Investment']].sort_values(by='Total Investment', ascending=True)
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
                st.caption("Which assets are 'safe bets' vs. 'risky'?")
                sentiment_df = master_df_all.reset_index().rename(columns={'index': 'Element'})
                fig_sentiment = px.scatter(
                    sentiment_df,
                    x='Negative associations',
                    y='Positive associations',
                    size='Total Investment',
                    color='Element',
                    hover_name='Element',
                    title="Sentiment Analysis"
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)

            st.markdown("---")
            st.markdown("##### Market & Media Analysis")

            # Market usage chart
            st.caption("Are elements used consistently across markets? Select markets below to compare.")
            all_available_markets = sorted(list(audit_df['Market'].unique()))
            selected_markets_for_chart = st.multiselect(
                "Select markets to compare:",
                options=all_available_markets,
                default=all_available_markets[:4] if len(all_available_markets) >= 4 else all_available_markets
            )

            if selected_markets_for_chart:
                market_usage = audit_df[audit_df['Market'].isin(selected_markets_for_chart)]
                market_usage = market_usage.groupby('Market')[brand_elements].mean().reset_index()
                market_usage_melted = market_usage.melt(id_vars='Market', value_vars=brand_elements, var_name='Element', value_name='% Used')
                fig_market = px.bar(
                    market_usage_melted,
                    x='Element',
                    y='% Used',
                    color='Market',
                    barmode='group',
                    title='Brand Element Usage by Market'
                )
                st.plotly_chart(fig_market, use_container_width=True)
            else:
                st.warning("Please select at least one market to display the chart.")

            st.caption("Are assets used effectively across media types?")
            media_usage = audit_df.groupby('Medium')[brand_elements].mean().T
            fig_media_heatmap = px.imshow(
                media_usage,
                labels=dict(x="Medium Type", y="Brand Element", color="% Used"),
                text_auto=True,
                aspect="auto",
                title="Element Usage Frequency by Media Type"
            )
            st.plotly_chart(fig_media_heatmap, use_container_width=True)

            st.markdown("---")
            st.markdown("##### ROI, Personality & Color Analysis")

            col5, col6 = st.columns(2)
            with col5:
                st.caption("Which assets are most efficient at driving Recognition?")
                master_df_all['Recognition_ROI'] = (master_df_all['% recognised'] / master_df_all['Total Investment']) * 1_000_000
                master_df_all['Recognition_ROI'].fillna(0, inplace=True)
                recognition_roi_df = master_df_all[master_df_all['Total Investment'] > 0]['Recognition_ROI'].sort_values(ascending=False).reset_index()
                fig_recognition_roi = px.bar(
                    recognition_roi_df,
                    x='Element',
                    y='Recognition_ROI',
                    title="Recognition Efficiency",
                    labels={'Recognition_ROI': 'Recognition % Points per €1M Invested'}
                )
                st.plotly_chart(fig_recognition_roi, use_container_width=True)

            with col6:
                st.caption("Are we investing in our most unique assets?")
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

            st.caption("What is the perceived personality of our key assets?")
            adj_cols = {'adj_bold': 'Bold', 'adj_stylish': 'Stylish', 'adj_modern': 'Modern'}
            personality_df = master_df_all[adj_cols.keys()].rename(columns=adj_cols).reset_index()
            personality_melted = personality_df.melt(id_vars='Element', var_name='Adjective', value_name='Score')
            selected_elements_personality = st.multiselect(
                "Select elements to compare for personality:",
                options=brand_elements,
                default=brand_elements[:2]
            )

            if selected_elements_personality:
                filtered_personality = personality_melted[personality_melted['Element'].isin(selected_elements_personality)]
                fig_personality = px.bar(
                    filtered_personality,
                    x="Adjective",
                    y="Score",
                    color="Element",
                    barmode="group",
                    title="Asset Personality Profile Comparison"
                )
                st.plotly_chart(fig_personality, use_container_width=True)

            st.caption("How much visual real estate do our brand colors occupy?")
            color_area_df = audit_df[['Electric_Green_Area_%', 'Dark_Green_Area_%']].melt(var_name='Color', value_name='Area %')
            color_area_df = color_area_df[color_area_df['Area %'] > 0]

            if not color_area_df.empty:
                fig_color_area = px.box(
                    color_area_df,
                    x='Color',
                    y='Area %',
                    color='Color',
                    title='Distribution of Color Area Coverage in Ads',
                    points='all'
                )
                st.plotly_chart(fig_color_area, use_container_width=True)
            else:
                st.warning("No color area data found. Add columns `Electric_Green_Area_%` and `Dark_Green_Area_%` to your file to see this chart.")

        with tab3:
            st.header("Data Explorer")
            st.caption("This section shows the raw Comms Audit data you uploaded.")
            st.dataframe(audit_df)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        st.error("Please ensure your file is a valid Excel/CSV with the expected column names (Market, Placement, Medium, Spend, and brand element columns).")

else:
    # Show instructions when no file is uploaded
    st.markdown("### 👋 Welcome!")
    st.markdown("""
    To get started, please upload your **Comms Audit** file above. Your file should include:

    **Required Columns:**
    - `Market` - The market/country (e.g., UK, Spain, Germany, Poland)
    - `Placement` - Ad placement type
    - `Medium` - Media type (e.g., TV, Digital, Print)
    - `Spend` - Investment amount (can include € symbol)
    - Brand element columns: One column for each of the 9 brand elements, with TRUE/FALSE or 1/0 values

    **Brand Elements:** Electric Green, Dark Green, Type, Tagline, Symbol, Hacek, Wordmark, Facets, Sonic

    **Optional Columns:**
    - `Electric_Green_Area_%` - Percentage of visual space occupied by electric green
    - `Dark_Green_Area_%` - Percentage of visual space occupied by dark green
    """)

    st.markdown("---")
    st.markdown("### 📊 Preview: Research Data Already Integrated")
    st.info("The quantitative research metrics from **P045556 - Saffron Brand Assets** are already integrated and will be combined with your Comms Audit data.")

    with st.expander("Preview Research Metrics"):
        st.dataframe(research_df.style.format("{:.1%}"))
