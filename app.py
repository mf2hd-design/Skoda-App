import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from comms_data import comms_audit_data

# --- Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Škoda Brand Intelligence Dashboard",
    page_icon="📊"
)

# --- Brand Elements ---
brand_elements = [
    "Electric Green", "Dark Green", "Type", "Tagline", "Symbol",
    "Hacek", "Wordmark", "Facets", "Sonic"
]

# --- Research Data from P045556 Study ---
# Recognition: Q02 (Have you seen/heard this element before?)
# Uniqueness: Q05 (Which brand do you think this belongs to? - % who said Škoda)
# Personality traits: Q04 NET T2B (Top 2 Box scores)
# Positive/Negative Associations: Q04 SUMMARY tables (NET: Left = Positive, NET: Right = Negative)
# Sentiment values are averages across 7 personality dimensions from Q04 SUMMARY tables
research_data = {
    'Electric Green': {'recognition': 0.38, 'uniqueness': 0.17, 'bold': 0.49, 'stylish': 0.46, 'modern': 0.50, 'simple': 0.50, 'human': 0.45, 'exciting': 0.45, 'playful': 0.44, 'positive': 0.470, 'negative': 0.236},
    'Dark Green': {'recognition': 0.39, 'uniqueness': 0.19, 'bold': 0.51, 'stylish': 0.49, 'modern': 0.52, 'simple': 0.53, 'human': 0.46, 'exciting': 0.49, 'playful': 0.45, 'positive': 0.479, 'negative': 0.231},
    'Type': {'recognition': 0.37, 'uniqueness': 0.17, 'bold': 0.47, 'stylish': 0.47, 'modern': 0.49, 'simple': 0.50, 'human': 0.44, 'exciting': 0.45, 'playful': 0.41, 'positive': 0.461, 'negative': 0.239},
    'Tagline': {'recognition': 0.36, 'uniqueness': 0.17, 'bold': 0.48, 'stylish': 0.48, 'modern': 0.51, 'simple': 0.50, 'human': 0.46, 'exciting': 0.51, 'playful': 0.45, 'positive': 0.501, 'negative': 0.214},
    'Symbol': {'recognition': 0.64, 'uniqueness': 0.38, 'bold': 0.50, 'stylish': 0.50, 'modern': 0.55, 'simple': 0.54, 'human': 0.46, 'exciting': 0.50, 'playful': 0.46, 'positive': 0.506, 'negative': 0.211},
    'Hacek': {'recognition': 0.38, 'uniqueness': 0.19, 'bold': 0.46, 'stylish': 0.46, 'modern': 0.49, 'simple': 0.55, 'human': 0.44, 'exciting': 0.44, 'playful': 0.42, 'positive': 0.489, 'negative': 0.223},
    'Wordmark': {'recognition': 0.44, 'uniqueness': 0.28, 'bold': 0.49, 'stylish': 0.49, 'modern': 0.54, 'simple': 0.52, 'human': 0.45, 'exciting': 0.48, 'playful': 0.45, 'positive': 0.493, 'negative': 0.216},
    'Facets': {'recognition': 0.38, 'uniqueness': 0.16, 'bold': 0.50, 'stylish': 0.48, 'modern': 0.51, 'simple': 0.51, 'human': 0.43, 'exciting': 0.46, 'playful': 0.46, 'positive': 0.466, 'negative': 0.240},
    'Sonic': {'recognition': 0.40, 'uniqueness': 0.17, 'bold': 0.50, 'stylish': 0.49, 'modern': 0.55, 'simple': 0.55, 'human': 0.46, 'exciting': 0.51, 'playful': 0.48, 'positive': 0.484, 'negative': 0.214},
}

# Recognition by Country
recognition_by_country = {
    'Electric Green': {'UK': 0.41, 'Spain': 0.38, 'Germany': 0.29, 'Poland': 0.44},
    'Dark Green': {'UK': 0.37, 'Spain': 0.38, 'Germany': 0.37, 'Poland': 0.41},
    'Type': {'UK': 0.45, 'Spain': 0.37, 'Germany': 0.30, 'Poland': 0.38},
    'Tagline': {'UK': 0.36, 'Spain': 0.40, 'Germany': 0.32, 'Poland': 0.36},
    'Symbol': {'UK': 0.54, 'Spain': 0.66, 'Germany': 0.61, 'Poland': 0.76},
    'Hacek': {'UK': 0.36, 'Spain': 0.38, 'Germany': 0.35, 'Poland': 0.41},
    'Wordmark': {'UK': 0.45, 'Spain': 0.47, 'Germany': 0.39, 'Poland': 0.48},
    'Facets': {'UK': 0.42, 'Spain': 0.39, 'Germany': 0.37, 'Poland': 0.35},
    'Sonic': {'UK': 0.39, 'Spain': 0.42, 'Germany': 0.39, 'Poland': 0.39},
}

# --- Load Comms Audit Data ---
audit_df = pd.DataFrame(comms_audit_data)

# --- Helper Functions ---
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

def calculate_metrics():
    """Calculate all metrics combining comms audit and research data"""
    metrics = []
    total_ads = len(audit_df)

    for element in brand_elements:
        element_df = audit_df[audit_df[element] == True]

        # Comms Audit metrics
        usage_pct = len(element_df) / total_ads if total_ads > 0 else 0
        total_investment = element_df['Spend'].sum()
        avg_investment = element_df['Spend'].mean() if len(element_df) > 0 else 0

        # Usage by medium
        usage_image = element_df[element_df['Medium'] == 'Image'].shape[0] / audit_df[audit_df['Medium'] == 'Image'].shape[0] if audit_df[audit_df['Medium'] == 'Image'].shape[0] > 0 else 0
        usage_video = element_df[element_df['Medium'] == 'Video'].shape[0] / audit_df[audit_df['Medium'] == 'Video'].shape[0] if audit_df[audit_df['Medium'] == 'Video'].shape[0] > 0 else 0

        # Research data
        research = research_data[element]

        # Recognition ROI
        recognition_roi = (research['recognition'] / total_investment * 1_000_000) if total_investment > 0 else 0
        
        # Net Sentiment
        net_sentiment = research['positive'] - research['negative']

        metrics.append({
            'Element': element,
            'Overall Usage': usage_pct,
            'Usage Image': usage_image,
            'Usage Video': usage_video,
            'Average Investment': avg_investment,
            'Total Investment': total_investment,
            'Recognition': research['recognition'],
            'Uniqueness': research['uniqueness'],
            'Positive': research['positive'],
            'Negative': research['negative'],
            'Net Sentiment': net_sentiment,
            'Recognition ROI': recognition_roi,
            'Bold': research['bold'],
            'Stylish': research['stylish'],
            'Modern': research['modern'],
        })

    return pd.DataFrame(metrics)

# Calculate master metrics
master_df = calculate_metrics()

# --- App Header ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Škoda Brand Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Navigation Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Executive Summary",
    "💚 Sentiment Analysis",
    "📈 Strategic Insights",
    "🎯 Non-Negotiables",
    "🔮 Future-Proofing",
    "🔍 Deep Dive Analysis",
    "📄 Data Explorer"
])

# ==================== TAB 1: EXECUTIVE SUMMARY ====================
with tab1:
    st.header("Executive Summary")
    st.caption("Combined view replicating Excel 'NEW Calculations ALL' sheet")

    # Key Headlines
    col1, col2, col3, col4 = st.columns(4)

    most_recognized = master_df.loc[master_df['Recognition'].idxmax()]
    most_unique = master_df.loc[master_df['Uniqueness'].idxmax()]
    highest_investment = master_df.loc[master_df['Total Investment'].idxmax()]
    best_roi = master_df.loc[master_df['Recognition ROI'].idxmax()]

    with col1:
        st.metric("Most Recognised Asset", most_recognized['Element'], f"{most_recognized['Recognition']:.0%}")
        st.info(f"**{most_recognized['Recognition']:.0%}** of consumers have seen or heard this element before, making it the most familiar Škoda brand asset. Strong recognition drives immediate brand attribution.")

    with col2:
        st.metric("Most Unique Asset", most_unique['Element'], f"{most_unique['Uniqueness']:.0%}")
        st.info(f"Rated **{most_unique['Uniqueness']:.0%}** for distinctiveness. High uniqueness means this element differentiates Škoda from competitors and builds long-term brand equity.")

    with col3:
        st.metric("Highest Investment", highest_investment['Element'], f"€{highest_investment['Total Investment']:,.0f}")
        st.info(f"**€{highest_investment['Total Investment']:,.0f}** invested across **{int(highest_investment['Overall Usage'] * 102)}** ads. This represents **{highest_investment['Overall Usage']:.0%}** of total campaign usage.")

    with col4:
        st.metric("Best Recognition ROI", best_roi['Element'], f"{best_roi['Recognition ROI']:.2f}")
        st.info(f"Delivers **{best_roi['Recognition ROI']:.2f}** recognition points per €1M spent. This asset achieves **{best_roi['Recognition']:.0%}** recognition with only **€{best_roi['Total Investment']:,.0f}** investment - the most efficient performer.")

    st.markdown("---")

    # Combined Analysis Table (matching Excel structure)
    st.markdown("#### Combined Analysis Table")
    st.caption("Synthesizes Comms Audit media metrics with Quantitative Research insights")

    display_df = master_df[[
        'Element', 'Overall Usage', 'Usage Image', 'Usage Video',
        'Average Investment', 'Total Investment',
        'Recognition', 'Uniqueness', 'Net Sentiment'
    ]].set_index('Element')

    # Style the dataframe
    styler = display_df.T.style

    # Heatmaps for research metrics
    research_rows = ['Recognition', 'Uniqueness', 'Net Sentiment']
    styler = styler.background_gradient(cmap='RdYlGn', axis=1, subset=(pd.IndexSlice[research_rows], slice(None)))

    # Format percentages and currency
    percent_rows = ['Overall Usage', 'Usage Image', 'Usage Video', 'Recognition', 'Uniqueness', 'Net Sentiment']
    currency_rows = ['Average Investment', 'Total Investment']
    styler = styler.format("{:.1%}", subset=(pd.IndexSlice[percent_rows], slice(None)))
    styler = styler.format("€{:,.2f}", subset=(pd.IndexSlice[currency_rows], slice(None)))

    st.dataframe(styler, use_container_width=True)

    # Export button
    excel_file = to_excel(display_df.fillna(0))
    st.download_button(
        label="📥 Export Analysis to Excel",
        data=excel_file,
        file_name="skoda_brand_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown("---")

    # Key Insights
    st.markdown("### 🎯 Key Strategic Findings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌟 Top Performers")
        top_recognition = master_df.nlargest(3, 'Recognition')
        for _, row in top_recognition.iterrows():
            st.success(f"**{row['Element']}**: {row['Recognition']:.0%} Recognition | Net Sentiment: {row['Net Sentiment']:+.0%}")

    with col2:
        st.markdown("#### ⚠️ Underutilized Assets")
        # Find assets with high performance but low usage
        master_df['performance_usage_gap'] = master_df['Recognition'] - master_df['Overall Usage']
        gap_assets = master_df.nlargest(3, 'performance_usage_gap')
        for _, row in gap_assets.iterrows():
            gap = row['performance_usage_gap'] * 100
            st.warning(f"**{row['Element']}**: {gap:+.0f}% gap (Recog: {row['Recognition']:.0%} vs Usage: {row['Overall Usage']:.0%})")

# ==================== TAB 2: SENTIMENT ANALYSIS ====================
with tab2:
    st.header("💚 Brand Sentiment Analysis")
    st.caption("Positive/Negative Associations from Q04 SUMMARY tables (NET: Left = Positive, NET: Right = Negative)")
    
    st.markdown("---")
    
    # Overview metrics
    col1, col2, col3 = st.columns(3)
    
    best_sentiment = master_df.loc[master_df['Net Sentiment'].idxmax()]
    most_positive = master_df.loc[master_df['Positive'].idxmax()]
    least_negative = master_df.loc[master_df['Negative'].idxmin()]
    
    with col1:
        st.metric("Best Net Sentiment", best_sentiment['Element'], f"{best_sentiment['Net Sentiment']:+.0%}")
        st.success(f"**{best_sentiment['Element']}** has {best_sentiment['Positive']:.0%} positive associations and only {best_sentiment['Negative']:.0%} negative associations")
    
    with col2:
        st.metric("Most Positive Asset", most_positive['Element'], f"{most_positive['Positive']:.0%}")
        st.info(f"**{most_positive['Positive']:.0%}** of consumers associate **{most_positive['Element']}** with positive brand attributes")
    
    with col3:
        st.metric("Least Negative Asset", least_negative['Element'], f"{least_negative['Negative']:.0%}")
        st.info(f"Only **{least_negative['Negative']:.0%}** negative associations for **{least_negative['Element']}** - the most universally appreciated asset")
    
    st.markdown("---")
    
    # Sentiment Comparison Chart
    st.markdown("### 📊 Positive vs Negative Associations")
    
    fig_sentiment = go.Figure()
    
    # Positive bars
    fig_sentiment.add_trace(go.Bar(
        name='Positive',
        x=master_df['Element'],
        y=master_df['Positive'],
        text=master_df['Positive'].apply(lambda x: f'{x:.0%}'),
        textposition='outside',
        marker_color='#4CAF50'
    ))
    
    # Negative bars (inverted)
    fig_sentiment.add_trace(go.Bar(
        name='Negative',
        x=master_df['Element'],
        y=-master_df['Negative'],
        text=master_df['Negative'].apply(lambda x: f'{x:.0%}'),
        textposition='outside',
        marker_color='#F44336'
    ))
    
    fig_sentiment.update_layout(
        title='Positive (Green) vs Negative (Red) Brand Associations',
        yaxis_title='Association Percentage',
        barmode='relative',
        height=500,
        yaxis=dict(tickformat='.0%'),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_sentiment, use_container_width=True)
    
    st.markdown("---")
    
    # Net Sentiment Analysis
    st.markdown("### 🎯 Net Sentiment Score (Positive - Negative)")
    
    # Sort by net sentiment
    sentiment_sorted = master_df.sort_values('Net Sentiment', ascending=True)
    
    fig_net = go.Figure(go.Bar(
        x=sentiment_sorted['Net Sentiment'],
        y=sentiment_sorted['Element'],
        orientation='h',
        text=sentiment_sorted['Net Sentiment'].apply(lambda x: f'{x:+.0%}'),
        textposition='outside',
        marker=dict(
            color=sentiment_sorted['Net Sentiment'],
            colorscale='RdYlGn',
            line=dict(color='rgba(0,0,0,0.3)', width=1)
        )
    ))
    
    fig_net.update_layout(
        title='Net Sentiment Score by Brand Element',
        xaxis_title='Net Sentiment (Positive - Negative)',
        height=500,
        xaxis=dict(tickformat='.0%'),
        showlegend=False
    )
    
    st.plotly_chart(fig_net, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed Sentiment Table
    st.markdown("### 📋 Detailed Sentiment Breakdown")
    
    sentiment_table = master_df[['Element', 'Positive', 'Negative', 'Net Sentiment']].sort_values('Net Sentiment', ascending=False)
    
    st.dataframe(
        sentiment_table.style.format({
            'Positive': '{:.1%}',
            'Negative': '{:.1%}',
            'Net Sentiment': '{:+.1%}'
        }).background_gradient(cmap='RdYlGn', subset=['Net Sentiment']),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Strategic Insights
    st.markdown("### 💡 Sentiment Insights & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Sentiment Winners")
        top_sentiment = master_df.nlargest(3, 'Net Sentiment')
        for idx, row in top_sentiment.iterrows():
            st.success(
                f"**{row['Element']}**\n\n"
                f"- Positive: {row['Positive']:.0%}\n"
                f"- Negative: {row['Negative']:.0%}\n"
                f"- Net: **{row['Net Sentiment']:+.0%}**\n\n"
                f"*Recommendation: Amplify usage to leverage positive sentiment*"
            )
    
    with col2:
        st.markdown("#### ⚠️ Sentiment Challenges")
        bottom_sentiment = master_df.nsmallest(3, 'Net Sentiment')
        for idx, row in bottom_sentiment.iterrows():
            st.warning(
                f"**{row['Element']}**\n\n"
                f"- Positive: {row['Positive']:.0%}\n"
                f"- Negative: {row['Negative']:.0%}\n"
                f"- Net: **{row['Net Sentiment']:+.0%}**\n\n"
                f"*Recommendation: Review application context and consider reducing exposure*"
            )
    
    st.markdown("---")
    
    # Sentiment vs Usage Analysis
    st.markdown("### 🔄 Sentiment vs Current Usage")
    st.caption("Are we using assets with the best sentiment?")
    
    fig_sentiment_usage = px.scatter(
        master_df,
        x='Net Sentiment',
        y='Overall Usage',
        size='Recognition',
        color='Net Sentiment',
        color_continuous_scale='RdYlGn',
        text='Element',
        title='Net Sentiment vs Current Usage (Size = Recognition)',
        labels={
            'Net Sentiment': 'Net Sentiment (Positive - Negative)',
            'Overall Usage': 'Current Usage in Communications'
        }
    )
    
    fig_sentiment_usage.update_traces(textposition='top center')
    fig_sentiment_usage.update_layout(height=600)
    fig_sentiment_usage.update_xaxes(tickformat='.0%')
    fig_sentiment_usage.update_yaxes(tickformat='.0%')
    
    st.plotly_chart(fig_sentiment_usage, use_container_width=True)
    
    # Identify misalignments
    st.markdown("#### 🎯 Usage-Sentiment Alignment")
    
    master_df['sentiment_usage_alignment'] = master_df['Net Sentiment'] - master_df['Overall Usage']
    misaligned = master_df[abs(master_df['sentiment_usage_alignment']) > 0.15].sort_values('sentiment_usage_alignment', ascending=False)
    
    if not misaligned.empty:
        for idx, row in misaligned.iterrows():
            if row['sentiment_usage_alignment'] > 0:
                st.success(
                    f"**{row['Element']}**: Strong positive sentiment ({row['Net Sentiment']:+.0%}) but low usage ({row['Overall Usage']:.0%}) "
                    f"→ **Opportunity to increase usage by {abs(row['sentiment_usage_alignment']):.0%}**"
                )
            else:
                st.warning(
                    f"**{row['Element']}**: Lower sentiment ({row['Net Sentiment']:+.0%}) but high usage ({row['Overall Usage']:.0%}) "
                    f"→ **Consider reducing usage by {abs(row['sentiment_usage_alignment']):.0%}**"
                )

# ==================== TAB 3: STRATEGIC INSIGHTS ====================
with tab3:
    st.header("📈 Strategic Insights")
    st.caption("Performance-Usage Gap Analysis & Investment Recommendations")

    # Key Metric: Performance-Usage Gap
    st.markdown("### 🎯 Performance-Usage Gap Analysis")
    st.caption("Identifies assets with strong performance (Recognition + Sentiment) but low usage")

    # Calculate comprehensive performance score
    master_df['performance_score'] = (
        (master_df['Recognition'] * 0.4) +
        (master_df['Uniqueness'] * 0.3) +
        (master_df['Net Sentiment'] * 0.3)
    )
    master_df['performance_usage_gap'] = master_df['performance_score'] - master_df['Overall Usage']

    # Visualize gap
    gap_sorted = master_df.sort_values('performance_usage_gap', ascending=True)

    fig_gap = go.Figure(go.Bar(
        x=gap_sorted['performance_usage_gap'],
        y=gap_sorted['Element'],
        orientation='h',
        text=gap_sorted['performance_usage_gap'].apply(lambda x: f'{x:+.0%}'),
        textposition='outside',
        marker=dict(
            color=gap_sorted['performance_usage_gap'],
            colorscale='RdYlGn',
            line=dict(color='rgba(0,0,0,0.3)', width=1)
        )
    ))

    fig_gap.update_layout(
        title='Performance-Usage Gap (Positive = Underutilized Opportunity)',
        xaxis_title='Gap (Performance Score - Current Usage)',
        height=500,
        xaxis=dict(tickformat='.0%', zeroline=True, zerolinewidth=2, zerolinecolor='black'),
        showlegend=False
    )

    st.plotly_chart(fig_gap, use_container_width=True)

    st.markdown("---")

    # Strategic Quadrant Analysis
    st.markdown("### 📊 Strategic Quadrant Analysis")
    st.caption("Positioning assets by Performance vs Investment")

    fig_quadrant = px.scatter(
        master_df,
        x='performance_score',
        y='Total Investment',
        size='Recognition',
        color='Net Sentiment',
        color_continuous_scale='RdYlGn',
        text='Element',
        title='Strategic Positioning: Performance vs Investment',
        labels={
            'performance_score': 'Performance Score (Recognition + Uniqueness + Sentiment)',
            'Total Investment': 'Total Media Investment (€)'
        }
    )

    fig_quadrant.update_traces(textposition='top center')
    fig_quadrant.update_layout(height=600)
    fig_quadrant.update_xaxes(tickformat='.0%')
    fig_quadrant.update_yaxes(tickformat='€,.0f')

    # Add quadrant lines
    median_performance = master_df['performance_score'].median()
    median_investment = master_df['Total Investment'].median()

    fig_quadrant.add_hline(y=median_investment, line_dash="dash", line_color="gray", opacity=0.5)
    fig_quadrant.add_vline(x=median_performance, line_dash="dash", line_color="gray", opacity=0.5)

    st.plotly_chart(fig_quadrant, use_container_width=True)

    # Quadrant Interpretations
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌟 Stars (High Performance, High Investment)")
        stars = master_df[
            (master_df['performance_score'] >= median_performance) &
            (master_df['Total Investment'] >= median_investment)
        ]
        for _, row in stars.iterrows():
            st.success(f"**{row['Element']}** - Performing well, investment justified")

        st.markdown("#### 💡 Rising Stars (High Performance, Low Investment)")
        rising = master_df[
            (master_df['performance_score'] >= median_performance) &
            (master_df['Total Investment'] < median_investment)
        ]
        for _, row in rising.iterrows():
            st.info(f"**{row['Element']}** - **OPPORTUNITY: Increase investment**")

    with col2:
        st.markdown("#### ⚠️ Question Marks (Low Performance, High Investment)")
        questions = master_df[
            (master_df['performance_score'] < median_performance) &
            (master_df['Total Investment'] >= median_investment)
        ]
        for _, row in questions.iterrows():
            st.warning(f"**{row['Element']}** - **CONCERN: Reduce investment**")

        st.markdown("#### 📉 Underperformers (Low Performance, Low Investment)")
        under = master_df[
            (master_df['performance_score'] < median_performance) &
            (master_df['Total Investment'] < median_investment)
        ]
        for _, row in under.iterrows():
            st.error(f"**{row['Element']}** - Low priority, maintain minimal usage")

    st.markdown("---")

    # Investment Efficiency
    st.markdown("### 💰 Investment Efficiency Analysis")
    
    # ROI table
    roi_data = master_df[['Element', 'Total Investment', 'Recognition', 'Recognition ROI', 'Net Sentiment']].sort_values('Recognition ROI', ascending=False)
    
    st.dataframe(
        roi_data.style.format({
            'Total Investment': '€{:,.0f}',
            'Recognition': '{:.1%}',
            'Recognition ROI': '{:.2f}',
            'Net Sentiment': '{:+.1%}'
        }).background_gradient(cmap='RdYlGn', subset=['Recognition ROI', 'Net Sentiment']),
        use_container_width=True
    )

    st.markdown("---")

    # Recommendations Summary
    st.markdown("### 🎯 Strategic Recommendations")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### ✅ INCREASE USAGE")
        increase = master_df[master_df['performance_usage_gap'] > 0.15].sort_values('performance_usage_gap', ascending=False)
        for _, row in increase.iterrows():
            st.success(
                f"**{row['Element']}**\n\n"
                f"- Performance: {row['performance_score']:.0%}\n"
                f"- Current Usage: {row['Overall Usage']:.0%}\n"
                f"- Gap: {row['performance_usage_gap']:+.0%}\n"
                f"- Net Sentiment: {row['Net Sentiment']:+.0%}"
            )

    with col2:
        st.markdown("#### 🔄 MAINTAIN CURRENT")
        maintain = master_df[abs(master_df['performance_usage_gap']) <= 0.15]
        for _, row in maintain.iterrows():
            st.info(
                f"**{row['Element']}**\n\n"
                f"- Well balanced\n"
                f"- Usage: {row['Overall Usage']:.0%}\n"
                f"- Performance: {row['performance_score']:.0%}"
            )

    with col3:
        st.markdown("#### ⚠️ REDUCE USAGE")
        reduce = master_df[master_df['performance_usage_gap'] < -0.15].sort_values('performance_usage_gap')
        for _, row in reduce.iterrows():
            st.warning(
                f"**{row['Element']}**\n\n"
                f"- Performance: {row['performance_score']:.0%}\n"
                f"- Current Usage: {row['Overall Usage']:.0%}\n"
                f"- Overused by: {abs(row['performance_usage_gap']):.0%}\n"
                f"- Net Sentiment: {row['Net Sentiment']:+.0%}"
            )

# ==================== TAB 4: NON-NEGOTIABLES ====================
with tab4:
    st.header("🎯 Non-Negotiable Brand Assets")
    st.caption("Must-use elements for 100% brand consistency")

    st.markdown("---")

    # Define non-negotiables based on criteria
    non_negotiable_criteria = (
        (master_df['Recognition'] >= 0.45) &
        (master_df['Uniqueness'] >= 0.25) &
        (master_df['Net Sentiment'] >= 0.20)
    )

    non_negotiables = master_df[non_negotiable_criteria].sort_values('Recognition', ascending=False)

    if not non_negotiables.empty:
        st.markdown("### ✅ Mandatory Assets")
        st.success(f"**{len(non_negotiables)}** brand elements meet the criteria for non-negotiable status")

        for _, row in non_negotiables.iterrows():
            with st.expander(f"**{row['Element']}** - Must appear in 100% of communications", expanded=True):
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Recognition", f"{row['Recognition']:.0%}")
                with col2:
                    st.metric("Uniqueness", f"{row['Uniqueness']:.0%}")
                with col3:
                    st.metric("Net Sentiment", f"{row['Net Sentiment']:+.0%}")
                with col4:
                    st.metric("Current Usage", f"{row['Overall Usage']:.0%}")

                # Usage recommendation
                if row['Overall Usage'] < 0.90:
                    st.warning(f"⚠️ **Action Required**: Currently only used in {row['Overall Usage']:.0%} of communications. Must increase to 100%.")
                else:
                    st.success(f"✅ **On Track**: Currently used in {row['Overall Usage']:.0%} of communications.")

                # Performance summary
                st.markdown("**Why this is non-negotiable:**")
                reasons = []
                if row['Recognition'] >= 0.60:
                    reasons.append(f"✓ Exceptional recognition ({row['Recognition']:.0%})")
                elif row['Recognition'] >= 0.45:
                    reasons.append(f"✓ Strong recognition ({row['Recognition']:.0%})")

                if row['Uniqueness'] >= 0.35:
                    reasons.append(f"✓ Highly unique ({row['Uniqueness']:.0%})")
                elif row['Uniqueness'] >= 0.25:
                    reasons.append(f"✓ Good uniqueness ({row['Uniqueness']:.0%})")

                if row['Net Sentiment'] >= 0.25:
                    reasons.append(f"✓ Very positive sentiment ({row['Net Sentiment']:+.0%})")
                elif row['Net Sentiment'] >= 0.20:
                    reasons.append(f"✓ Positive sentiment ({row['Net Sentiment']:+.0%})")

                for reason in reasons:
                    st.markdown(reason)

    st.markdown("---")

    # Strategic amplifiers
    st.markdown("### 🎨 Strategic Amplifiers")
    st.caption("High-value assets to use frequently (target: 60-80% of communications)")

    amplifier_criteria = (
        (master_df['Recognition'] >= 0.35) &
        (master_df['Uniqueness'] >= 0.15) &
        (master_df['Net Sentiment'] >= 0.15) &
        (~non_negotiable_criteria)
    )

    amplifiers = master_df[amplifier_criteria].sort_values('performance_score', ascending=False)

    if not amplifiers.empty:
        for _, row in amplifiers.iterrows():
            with st.expander(f"**{row['Element']}** - Target 60-80% usage"):
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Recognition", f"{row['Recognition']:.0%}")
                with col2:
                    st.metric("Performance Score", f"{row['performance_score']:.0%}")
                with col3:
                    st.metric("Net Sentiment", f"{row['Net Sentiment']:+.0%}")
                with col4:
                    current_usage = row['Overall Usage']
                    target_min = 0.60
                    target_max = 0.80
                    if current_usage < target_min:
                        delta = f"+{target_min - current_usage:.0%}"
                        delta_color = "off"
                    elif current_usage > target_max:
                        delta = f"-{current_usage - target_max:.0%}"
                        delta_color = "inverse"
                    else:
                        delta = "On target"
                        delta_color = "normal"
                    st.metric("Current Usage", f"{current_usage:.0%}", delta)

    st.markdown("---")

    # Assets to reduce
    st.markdown("### ⚠️ Assets to Reduce")
    st.caption("Lower-performing elements to use sparingly (<40% of communications)")

    reduce_criteria = (
        (master_df['performance_score'] < 0.35) |
        (master_df['Net Sentiment'] < 0.15)
    )

    reduce_assets = master_df[reduce_criteria].sort_values('Overall Usage', ascending=False)

    if not reduce_assets.empty:
        for _, row in reduce_assets.iterrows():
            with st.expander(f"**{row['Element']}** - Reduce to <40% usage"):
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Current Usage", f"{row['Overall Usage']:.0%}")
                with col2:
                    st.metric("Performance Score", f"{row['performance_score']:.0%}")
                with col3:
                    st.metric("Net Sentiment", f"{row['Net Sentiment']:+.0%}")
                with col4:
                    if row['Overall Usage'] > 0.40:
                        st.metric("Reduce By", f"{row['Overall Usage'] - 0.40:.0%}", delta=f"-{row['Overall Usage'] - 0.40:.0%}", delta_color="inverse")
                    else:
                        st.metric("Status", "✅ Within target")

                # Explanation
                st.markdown("**Reasons for reduction:**")
                if row['performance_score'] < 0.35:
                    st.markdown(f"• Lower overall performance score ({row['performance_score']:.0%})")
                if row['Recognition'] < 0.35:
                    st.markdown(f"• Lower recognition ({row['Recognition']:.0%})")
                if row['Net Sentiment'] < 0.15:
                    st.markdown(f"• Weaker sentiment ({row['Net Sentiment']:+.0%})")
                if row['Overall Usage'] > 0.60:
                    st.markdown(f"• Currently overused ({row['Overall Usage']:.0%})")

# ==================== TAB 5: FUTURE-PROOFING ====================
with tab5:
    st.header("🔮 Future-Proofing Strategy")
    st.caption("Investment priorities and market-specific recommendations")

    st.markdown("---")

    # Investment Priority Matrix
    st.markdown("### 💰 Investment Priority Matrix")
    st.caption("Based on Recognition ROI and Net Sentiment")

    fig_priority = px.scatter(
        master_df,
        x='Recognition ROI',
        y='Net Sentiment',
        size='Recognition',
        color='Overall Usage',
        color_continuous_scale='Blues',
        text='Element',
        title='Investment Priorities: ROI vs Sentiment (Size = Recognition, Color = Current Usage)',
        labels={
            'Recognition ROI': 'Recognition ROI (Recognition per €1M)',
            'Net Sentiment': 'Net Brand Sentiment',
            'Overall Usage': 'Current Usage'
        }
    )

    fig_priority.update_traces(textposition='top center')
    fig_priority.update_layout(height=600)
    fig_priority.update_yaxes(tickformat='.0%')

    st.plotly_chart(fig_priority, use_container_width=True)

    # Priority recommendations
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 High Priority Investments")
        st.caption("High ROI + High Sentiment + Low Usage")

        high_priority = master_df[
            (master_df['Recognition ROI'] > master_df['Recognition ROI'].median()) &
            (master_df['Net Sentiment'] > 0.20) &
            (master_df['Overall Usage'] < 0.50)
        ].sort_values('Recognition ROI', ascending=False)

        for _, row in high_priority.iterrows():
            st.success(
                f"**{row['Element']}**\n\n"
                f"- ROI: {row['Recognition ROI']:.2f}\n"
                f"- Sentiment: {row['Net Sentiment']:+.0%}\n"
                f"- Current Usage: {row['Overall Usage']:.0%}\n\n"
                f"*Recommended Action: Increase investment and usage*"
            )

    with col2:
        st.markdown("#### ⚠️ Optimize or Reduce")
        st.caption("Lower ROI or Negative Sentiment")

        low_priority = master_df[
            (master_df['Recognition ROI'] <= master_df['Recognition ROI'].median()) |
            (master_df['Net Sentiment'] < 0.20)
        ].sort_values('Recognition ROI')

        for _, row in low_priority.iterrows():
            st.warning(
                f"**{row['Element']}**\n\n"
                f"- ROI: {row['Recognition ROI']:.2f}\n"
                f"- Sentiment: {row['Net Sentiment']:+.0%}\n"
                f"- Current Usage: {row['Overall Usage']:.0%}\n\n"
                f"*Recommended Action: Review application or reduce investment*"
            )

    st.markdown("---")

    # Market-Specific Opportunities
    st.markdown("### 🌍 Market-Specific Strategies")

    tab_uk, tab_spain, tab_germany, tab_poland = st.tabs(["🇬🇧 UK", "🇪🇸 Spain", "🇩🇪 Germany", "🇵🇱 Poland"])

    for tab, country in [(tab_uk, 'UK'), (tab_spain, 'Spain'), (tab_germany, 'Germany'), (tab_poland, 'Poland')]:
        with tab:
            st.markdown(f"### {country} Market Analysis")

            # Create country-specific data
            country_data = []
            for element in brand_elements:
                country_recognition = recognition_by_country[element][country]
                overall_recognition = research_data[element]['recognition']
                country_data.append({
                    'Element': element,
                    'Country Recognition': country_recognition,
                    'Overall Recognition': overall_recognition,
                    'Country vs Overall': country_recognition - overall_recognition,
                    'Net Sentiment': research_data[element]['positive'] - research_data[element]['negative']
                })

            country_df = pd.DataFrame(country_data)

            # Strongest performers
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🌟 Strongest Assets")
                top_country = country_df.nlargest(3, 'Country Recognition')
                for _, row in top_country.iterrows():
                    st.success(
                        f"**{row['Element']}**: {row['Country Recognition']:.0%}\n\n"
                        f"vs Overall: {row['Country vs Overall']:+.0%} | Sentiment: {row['Net Sentiment']:+.0%}"
                    )

            with col2:
                st.markdown("#### 📈 Growth Opportunities")
                growth = country_df.sort_values('Country vs Overall').head(3)
                for _, row in growth.iterrows():
                    st.info(
                        f"**{row['Element']}**: {row['Country Recognition']:.0%}\n\n"
                        f"Below avg by: {abs(row['Country vs Overall']):.0%} | Opportunity to strengthen"
                    )

    st.markdown("---")

    # Timeline & Phasing
    st.markdown("### 📅 Implementation Timeline")

    st.markdown("""
    #### Phase 1: Immediate (0-3 months)
    - Audit all current communications for non-negotiable assets
    - Establish brand asset approval checklist
    - Brief creative teams on new guidelines

    #### Phase 2: Short-term (3-6 months)
    - Increase usage of high-ROI, high-sentiment assets
    - Reduce reliance on overused, lower-performing elements
    - Implement market-specific strategies

    #### Phase 3: Medium-term (6-12 months)
    - Monitor performance metrics and adjust
    - Conduct mid-year brand health check
    - Refine guidelines based on results

    #### Phase 4: Long-term (12+ months)
    - Annual comprehensive review
    - Update research and benchmark against competitors
    - Evolve strategy based on market changes
    """)

# ==================== TAB 6: DEEP DIVE ANALYSIS ====================
with tab6:
    st.header("🔍 Deep Dive Analysis")
    st.caption("Detailed charts and visualizations")

    # Recognition Deep Dive
    st.markdown("### 📊 Recognition Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Recognition bar chart
        recognition_sorted = master_df.sort_values('Recognition', ascending=True)

        fig_recognition = go.Figure(go.Bar(
            x=recognition_sorted['Recognition'],
            y=recognition_sorted['Element'],
            orientation='h',
            text=recognition_sorted['Recognition'].apply(lambda x: f'{x:.0%}'),
            textposition='outside',
            marker=dict(
                color=recognition_sorted['Recognition'],
                colorscale='Greens',
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            )
        ))

        fig_recognition.update_layout(
            title='Brand Element Recognition',
            xaxis_title='Recognition %',
            height=500,
            xaxis=dict(tickformat='.0%'),
            showlegend=False
        )

        st.plotly_chart(fig_recognition, use_container_width=True)

    with col2:
        st.markdown("#### Key Insights:")

        # Highest recognition
        highest = master_df.loc[master_df['Recognition'].idxmax()]
        st.success(f"**Highest**: {highest['Element']} at {highest['Recognition']:.0%}")

        # Lowest recognition
        lowest = master_df.loc[master_df['Recognition'].idxmin()]
        st.error(f"**Lowest**: {lowest['Element']} at {lowest['Recognition']:.0%}")

        # Average
        avg_recognition = master_df['Recognition'].mean()
        st.info(f"**Average**: {avg_recognition:.0%}")

        # Recognition distribution
        above_avg = len(master_df[master_df['Recognition'] >= avg_recognition])
        st.metric("Above Average", f"{above_avg} of {len(master_df)}")

    st.markdown("---")

    # Uniqueness Analysis
    st.markdown("### 🎯 Uniqueness Analysis")

    fig_uniqueness = px.bar(
        master_df.sort_values('Uniqueness'),
        x='Uniqueness',
        y='Element',
        orientation='h',
        text=master_df.sort_values('Uniqueness')['Uniqueness'].apply(lambda x: f'{x:.0%}'),
        title='Brand Element Uniqueness',
        color='Uniqueness',
        color_continuous_scale='Blues'
    )

    fig_uniqueness.update_traces(textposition='outside')
    fig_uniqueness.update_layout(height=500)
    fig_uniqueness.update_xaxes(tickformat='.0%')

    st.plotly_chart(fig_uniqueness, use_container_width=True)

    st.markdown("---")

    # Usage Analysis
    st.markdown("### 📺 Usage Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig_usage = px.bar(
            master_df.sort_values('Overall Usage'),
            x='Overall Usage',
            y='Element',
            orientation='h',
            text=master_df.sort_values('Overall Usage')['Overall Usage'].apply(lambda x: f'{x:.0%}'),
            title='Overall Usage in Communications',
            color='Overall Usage',
            color_continuous_scale='Oranges'
        )
        fig_usage.update_traces(textposition='outside')
        fig_usage.update_layout(height=500)
        fig_usage.update_xaxes(tickformat='.0%')
        st.plotly_chart(fig_usage, use_container_width=True)

    with col2:
        # Usage by medium
        medium_data = []
        for element in brand_elements:
            element_df = audit_df[audit_df[element] == True]
            image_usage = len(element_df[element_df['Medium'] == 'Image'])
            video_usage = len(element_df[element_df['Medium'] == 'Video'])

            medium_data.append({
                'Element': element,
                'Image': image_usage,
                'Video': video_usage
            })

        medium_df = pd.DataFrame(medium_data)

        fig_medium = go.Figure()
        fig_medium.add_trace(go.Bar(name='Image', x=medium_df['Element'], y=medium_df['Image'], marker_color='#90CAF9'))
        fig_medium.add_trace(go.Bar(name='Video', x=medium_df['Element'], y=medium_df['Video'], marker_color='#4CAF50'))

        fig_medium.update_layout(
            barmode='stack',
            title='Usage by Medium',
            xaxis_title='Brand Element',
            yaxis_title='Number of Ads',
            height=500
        )

        st.plotly_chart(fig_medium, use_container_width=True)

    st.markdown("---")

    # Personality Deep Dive
    st.markdown("### 🎨 Brand Personality Deep Dive")

    # Allow comparison of elements
    comparison_elements = st.multiselect(
        "Select elements to compare:",
        brand_elements,
        default=brand_elements[:3]
    )

    if comparison_elements:
        personality_data = []
        for element in comparison_elements:
            research = research_data[element]
            personality_data.append({
                'Element': element,
                'Bold': research['bold'],
                'Stylish': research['stylish'],
                'Modern': research['modern'],
                'Simple': research['simple'],
                'Human': research['human'],
                'Exciting': research['exciting'],
                'Playful': research['playful']
            })

        personality_df = pd.DataFrame(personality_data).melt(
            id_vars='Element',
            var_name='Attribute',
            value_name='Score'
        )

        fig_personality = px.bar(
            personality_df,
            x='Attribute',
            y='Score',
            color='Element',
            barmode='group',
            text=personality_df['Score'].apply(lambda x: f'{x:.0%}'),
            title='Brand Personality Attributes (7 Dimensions)'
        )

        fig_personality.update_layout(height=500)
        fig_personality.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig_personality, use_container_width=True)

    st.markdown("---")

    # Market/Country Recognition Analysis
    st.markdown("### 🌍 Recognition by Market")
    st.caption("See how brand elements perform across different countries")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Heatmap of recognition by country
        heatmap_data = []
        for element in brand_elements:
            row_data = {'Element': element}
            for country in ['UK', 'Spain', 'Germany', 'Poland']:
                row_data[country] = recognition_by_country[element][country]
            heatmap_data.append(row_data)

        heatmap_df = pd.DataFrame(heatmap_data).set_index('Element')

        fig_heatmap = px.imshow(
            heatmap_df,
            labels=dict(x="Country", y="Brand Element", color="Recognition"),
            text_auto='.0%',
            aspect="auto",
            color_continuous_scale='RdYlGn',
            title="Brand Element Recognition by Country"
        )
        fig_heatmap.update_layout(height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with col2:
        st.markdown("#### Key Findings:")

        # Find strongest market per element
        st.markdown("**Strongest Markets:**")
        for element in brand_elements[:5]:  # Show top 5
            countries_sorted = sorted(
                recognition_by_country[element].items(),
                key=lambda x: x[1],
                reverse=True
            )
            best_country = countries_sorted[0]
            st.success(f"**{element}**: {best_country[0]} ({best_country[1]:.0%})")

        st.markdown("**Market Opportunities:**")
        # Find elements with biggest market variations
        variations = []
        for element in brand_elements:
            values = list(recognition_by_country[element].values())
            variation = max(values) - min(values)
            variations.append((element, variation))

        variations_sorted = sorted(variations, key=lambda x: x[1], reverse=True)
        for element, var in variations_sorted[:3]:
            st.warning(f"**{element}**: {var:.0%} variation - consider market-specific strategies")

# ==================== TAB 7: DATA EXPLORER ====================
with tab7:
    st.header("📄 Data Explorer")
    st.caption("Raw data access and detailed views")

    tab_a, tab_b, tab_c = st.tabs(["Comms Audit Data", "Research Data", "Combined Metrics"])

    with tab_a:
        st.markdown("### Comms Audit Data (102 Ads)")
        st.dataframe(audit_df, use_container_width=True)

        csv = audit_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Comms Audit CSV",
            data=csv,
            file_name="skoda_comms_audit.csv",
            mime="text/csv"
        )

    with tab_b:
        st.markdown("### Research Data (P045556 - Saffron Brand Assets)")

        research_view = st.radio(
            "Select view:",
            ["Core Metrics", "Sentiment Analysis", "Extended Personality (7 Dimensions)", "Recognition by Country"],
            horizontal=True
        )

        if research_view == "Core Metrics":
            research_display = []
            for element, data in research_data.items():
                research_display.append({
                    'Element': element,
                    'Recognition': data['recognition'],
                    'Uniqueness': data['uniqueness'],
                    'Net Sentiment': data['positive'] - data['negative']
                })
            research_display_df = pd.DataFrame(research_display)

            st.dataframe(research_display_df.style.format({
                'Recognition': '{:.1%}',
                'Uniqueness': '{:.1%}',
                'Net Sentiment': '{:+.1%}'
            }).background_gradient(cmap='RdYlGn', subset=['Net Sentiment']), use_container_width=True)

        elif research_view == "Sentiment Analysis":
            sentiment_display = []
            for element, data in research_data.items():
                sentiment_display.append({
                    'Element': element,
                    'Positive Associations': data['positive'],
                    'Negative Associations': data['negative'],
                    'Net Sentiment': data['positive'] - data['negative']
                })
            sentiment_display_df = pd.DataFrame(sentiment_display)

            st.dataframe(sentiment_display_df.style.format({
                'Positive Associations': '{:.1%}',
                'Negative Associations': '{:.1%}',
                'Net Sentiment': '{:+.1%}'
            }).background_gradient(cmap='RdYlGn', subset=['Net Sentiment']), use_container_width=True)

        elif research_view == "Extended Personality (7 Dimensions)":
            personality_display = []
            for element, data in research_data.items():
                personality_display.append({
                    'Element': element,
                    'Bold': data['bold'],
                    'Stylish': data['stylish'],
                    'Modern': data['modern'],
                    'Simple': data['simple'],
                    'Human': data['human'],
                    'Exciting': data['exciting'],
                    'Playful': data['playful']
                })
            personality_display_df = pd.DataFrame(personality_display)

            st.dataframe(personality_display_df.style.format({
                'Bold': '{:.1%}',
                'Stylish': '{:.1%}',
                'Modern': '{:.1%}',
                'Simple': '{:.1%}',
                'Human': '{:.1%}',
                'Exciting': '{:.1%}',
                'Playful': '{:.1%}'
            }), use_container_width=True)

        else:  # Recognition by Country
            country_display = []
            for element in brand_elements:
                row_data = {'Element': element}
                row_data.update(recognition_by_country[element])
                country_display.append(row_data)
            country_display_df = pd.DataFrame(country_display)

            st.dataframe(country_display_df.style.format({
                'UK': '{:.1%}',
                'Spain': '{:.1%}',
                'Germany': '{:.1%}',
                'Poland': '{:.1%}'
            }), use_container_width=True)

    with tab_c:
        st.markdown("### Combined Metrics")
        
        # Add filter options
        col1, col2 = st.columns(2)
        with col1:
            min_recognition = st.slider("Minimum Recognition", 0.0, 1.0, 0.0, 0.05)
        with col2:
            min_sentiment = st.slider("Minimum Net Sentiment", -0.5, 0.5, -0.5, 0.05)
        
        filtered_df = master_df[
            (master_df['Recognition'] >= min_recognition) &
            (master_df['Net Sentiment'] >= min_sentiment)
        ]
        
        st.dataframe(filtered_df, use_container_width=True)

        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Combined Metrics CSV",
            data=csv,
            file_name="skoda_combined_metrics.csv",
            mime="text/csv"
        )

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
<p><b>Škoda Brand Intelligence Dashboard</b> | Enhanced with Sentiment Analysis | Powered by Saffron</p>
<p style='font-size: 0.9em; margin-top: 10px;'>
Data Sources: P045556 Brand Assets Study | Q04 SUMMARY Tables (Positive/Negative Associations) | Comms Audit (102 Ads)
</p>
</div>
""", unsafe_allow_html=True)
