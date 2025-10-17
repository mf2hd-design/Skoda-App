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
research_data = {
    'Electric Green': {'recognition': 0.38, 'uniqueness': 0.62, 'positive': 0.47, 'negative': 0.24, 'bold': 0.49, 'stylish': 0.46, 'modern': 0.50},
    'Dark Green': {'recognition': 0.39, 'uniqueness': 0.61, 'positive': 0.49, 'negative': 0.22, 'bold': 0.51, 'stylish': 0.49, 'modern': 0.52},
    'Type': {'recognition': 0.37, 'uniqueness': 0.63, 'positive': 0.46, 'negative': 0.24, 'bold': 0.47, 'stylish': 0.47, 'modern': 0.49},
    'Tagline': {'recognition': 0.36, 'uniqueness': 0.64, 'positive': 0.48, 'negative': 0.22, 'bold': 0.48, 'stylish': 0.48, 'modern': 0.51},
    'Symbol': {'recognition': 0.64, 'uniqueness': 0.36, 'positive': 0.49, 'negative': 0.22, 'bold': 0.50, 'stylish': 0.50, 'modern': 0.55},
    'Hacek': {'recognition': 0.38, 'uniqueness': 0.62, 'positive': 0.45, 'negative': 0.25, 'bold': 0.46, 'stylish': 0.46, 'modern': 0.49},
    'Wordmark': {'recognition': 0.44, 'uniqueness': 0.56, 'positive': 0.48, 'negative': 0.23, 'bold': 0.49, 'stylish': 0.49, 'modern': 0.54},
    'Facets': {'recognition': 0.38, 'uniqueness': 0.62, 'positive': 0.47, 'negative': 0.23, 'bold': 0.50, 'stylish': 0.48, 'modern': 0.51},
    'Sonic': {'recognition': 0.40, 'uniqueness': 0.60, 'positive': 0.50, 'negative': 0.22, 'bold': 0.50, 'stylish': 0.49, 'modern': 0.55},
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

        metrics.append({
            'Element': element,
            'Overall Usage': usage_pct,
            'Usage Image': usage_image,
            'Usage Video': usage_video,
            'Average Investment': avg_investment,
            'Total Investment': total_investment,
            'Recognition': research['recognition'],
            'Uniqueness': research['uniqueness'],
            'Positive Associations': research['positive'],
            'Negative Associations': research['negative'],
            'POS/NEG': 'POS' if research['positive'] > research['negative'] else 'NEG',
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

st.markdown("""
<div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 20px;'>
<h3>Comprehensive Brand Asset Analysis</h3>
<p>Combining <b>Comms Audit Data</b> (102 ads across 4 markets) with <b>Quantitative Research</b> (P045556 - Saffron Brand Assets)</p>
<p><i>Objective: Determine which Škoda brand assets are the most iconic and build long-term recognition</i></p>
</div>
""", unsafe_allow_html=True)

# --- Navigation Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Executive Summary",
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
    with col2:
        st.metric("Most Unique Asset", most_unique['Element'], f"{most_unique['Uniqueness']:.0%}")
    with col3:
        st.metric("Highest Investment", highest_investment['Element'], f"€{highest_investment['Total Investment']:,.0f}")
    with col4:
        st.metric("Best Recognition ROI", best_roi['Element'], f"{best_roi['Recognition ROI']:.2f}")

    st.markdown("---")

    # Combined Analysis Table (matching Excel structure)
    st.markdown("#### Combined Analysis Table")
    st.caption("Synthesizes Comms Audit media metrics with Quantitative Research insights")

    display_df = master_df[[
        'Element', 'Overall Usage', 'Usage Image', 'Usage Video',
        'Average Investment', 'Total Investment',
        'Recognition', 'Uniqueness', 'Positive Associations', 'Negative Associations', 'POS/NEG'
    ]].set_index('Element')

    # Style the dataframe
    styler = display_df.T.style

    # Heatmaps for research metrics
    research_rows = ['Recognition', 'Uniqueness', 'Positive Associations', 'Negative Associations']
    styler = styler.background_gradient(cmap='RdYlGn', axis=1, subset=(pd.IndexSlice[research_rows], slice(None)))
    styler = styler.background_gradient(cmap='RdYlGn_r', axis=1, subset=(pd.IndexSlice[['Negative Associations']], slice(None)))

    # Format percentages and currency
    percent_rows = ['Overall Usage', 'Usage Image', 'Usage Video', 'Recognition', 'Uniqueness', 'Positive Associations', 'Negative Associations']
    currency_rows = ['Average Investment', 'Total Investment']
    styler = styler.format("{:.1%}", subset=(pd.IndexSlice[percent_rows], slice(None)))
    styler = styler.format("€{:,.2f}", subset=(pd.IndexSlice[currency_rows], slice(None)))

    st.dataframe(styler, use_container_width=True)

    # Export button
    excel_file = to_excel(display_df.fillna(0))
    st.download_button(
        label="📥 Export Analysis to Excel",
        data=excel_file,
        file_name="skoda_combined_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown("---")

    # Brand Equity Matrix
    st.markdown("#### Brand Equity Matrix: Fame vs. Uniqueness")
    st.caption("Bubble size represents total investment. Color shows positive association strength.")

    fig_matrix = px.scatter(
        master_df,
        x="Uniqueness",
        y="Recognition",
        size="Total Investment",
        color="Positive Associations",
        text="Element",
        size_max=60,
        hover_data=['Total Investment', 'Average Investment', 'Overall Usage'],
        color_continuous_scale='RdYlGn',
        title="Fame vs. Uniqueness (Size by Total Investment)"
    )
    fig_matrix.update_traces(textposition='top center')
    fig_matrix.update_layout(height=600)
    st.plotly_chart(fig_matrix, use_container_width=True)

# ==================== TAB 2: STRATEGIC INSIGHTS ====================
with tab2:
    st.header("Strategic Insights Dashboard")
    st.caption("Advanced analytics to identify opportunities and optimize brand asset usage")

    # Recognition ROI Analysis
    st.markdown("### 💡 Recognition ROI: Efficiency Analysis")
    st.info("**Insight:** Which assets deliver the best recognition per Euro spent?")

    col1, col2 = st.columns([2, 1])

    with col1:
        roi_df = master_df.sort_values('Recognition ROI', ascending=True)
        fig_roi = px.bar(
            roi_df,
            y='Element',
            x='Recognition ROI',
            orientation='h',
            title='Recognition Efficiency (Recognition % per €1M Invested)',
            text=roi_df['Recognition ROI'].apply(lambda x: f'{x:.2f}'),
            color='Recognition ROI',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_roi, use_container_width=True)

    with col2:
        st.markdown("#### Key Findings:")
        top_3_roi = roi_df.nlargest(3, 'Recognition ROI')
        for idx, row in top_3_roi.iterrows():
            st.success(f"**{row['Element']}**: {row['Recognition ROI']:.2f} points per €1M")

        st.markdown("#### Recommendations:")
        st.write("• Increase investment in high-ROI assets")
        st.write("• Re-evaluate low-ROI assets")
        st.write("• Balance reach with efficiency")

    st.markdown("---")

    # Efficiency Quadrant Analysis
    st.markdown("### 📊 Asset Performance Quadrants")
    st.info("**Insight:** Categorize assets by recognition and uniqueness performance")

    # Calculate quadrants
    median_recognition = master_df['Recognition'].median()
    median_uniqueness = master_df['Uniqueness'].median()

    def get_quadrant(row):
        if row['Recognition'] >= median_recognition and row['Uniqueness'] >= median_uniqueness:
            return 'Stars ⭐'
        elif row['Recognition'] >= median_recognition and row['Uniqueness'] < median_uniqueness:
            return 'Workhorses 🐴'
        elif row['Recognition'] < median_recognition and row['Uniqueness'] >= median_uniqueness:
            return 'Hidden Gems 💎'
        else:
            return 'Question Marks ❓'

    master_df['Quadrant'] = master_df.apply(get_quadrant, axis=1)

    fig_quadrant = px.scatter(
        master_df,
        x='Uniqueness',
        y='Recognition',
        color='Quadrant',
        text='Element',
        size='Total Investment',
        size_max=50,
        title='Asset Performance Quadrants',
        color_discrete_map={
            'Stars ⭐': '#4CAF50',
            'Workhorses 🐴': '#2196F3',
            'Hidden Gems 💎': '#FF9800',
            'Question Marks ❓': '#F44336'
        }
    )

    # Add median lines
    fig_quadrant.add_hline(y=median_recognition, line_dash="dash", line_color="gray", annotation_text="Median Recognition")
    fig_quadrant.add_vline(x=median_uniqueness, line_dash="dash", line_color="gray", annotation_text="Median Uniqueness")
    fig_quadrant.update_traces(textposition='top center')
    fig_quadrant.update_layout(height=600)

    st.plotly_chart(fig_quadrant, use_container_width=True)

    # Quadrant breakdown
    col1, col2, col3, col4 = st.columns(4)

    quadrant_counts = master_df['Quadrant'].value_counts()

    with col1:
        stars = master_df[master_df['Quadrant'] == 'Stars ⭐']['Element'].tolist()
        st.success(f"**Stars ⭐** ({len(stars)})")
        st.write("High Recognition + High Uniqueness")
        for asset in stars:
            st.write(f"• {asset}")

    with col2:
        workhorses = master_df[master_df['Quadrant'] == 'Workhorses 🐴']['Element'].tolist()
        st.info(f"**Workhorses 🐴** ({len(workhorses)})")
        st.write("High Recognition + Lower Uniqueness")
        for asset in workhorses:
            st.write(f"• {asset}")

    with col3:
        gems = master_df[master_df['Quadrant'] == 'Hidden Gems 💎']['Element'].tolist()
        st.warning(f"**Hidden Gems 💎** ({len(gems)})")
        st.write("Lower Recognition + High Uniqueness")
        for asset in gems:
            st.write(f"• {asset}")

    with col4:
        questions = master_df[master_df['Quadrant'] == 'Question Marks ❓']['Element'].tolist()
        st.error(f"**Question Marks ❓** ({len(questions)})")
        st.write("Lower Recognition + Lower Uniqueness")
        for asset in questions:
            st.write(f"• {asset}")

    st.markdown("---")

    # Market Consistency Analysis
    st.markdown("### 🌍 Market Consistency Analysis")
    st.info("**Insight:** Are assets used consistently across markets?")

    markets = sorted(audit_df['Market'].unique())
    selected_markets = st.multiselect("Select markets to compare:", markets, default=markets)

    if selected_markets:
        market_data = []
        for market in selected_markets:
            market_df = audit_df[audit_df['Market'] == market]
            total_ads = len(market_df)
            for element in brand_elements:
                usage = market_df[element].sum() / total_ads if total_ads > 0 else 0
                market_data.append({'Market': market, 'Element': element, 'Usage': usage})

        market_comparison = pd.DataFrame(market_data)

        fig_market = px.bar(
            market_comparison,
            x='Element',
            y='Usage',
            color='Market',
            barmode='group',
            title='Brand Element Usage by Market',
            text=market_comparison['Usage'].apply(lambda x: f'{x:.0%}')
        )
        st.plotly_chart(fig_market, use_container_width=True)

        # Consistency score
        consistency_scores = market_comparison.groupby('Element')['Usage'].std()
        most_consistent = consistency_scores.idxmin()
        least_consistent = consistency_scores.idxmax()

        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Most Consistent:** {most_consistent} (σ={consistency_scores[most_consistent]:.3f})")
        with col2:
            st.warning(f"**Least Consistent:** {least_consistent} (σ={consistency_scores[least_consistent]:.3f})")

# ==================== TAB 3: NON-NEGOTIABLES ====================
with tab3:
    st.header("🎯 Non-Negotiables: Asset Usage Guidelines")
    st.caption("Data-driven recommendations for mandatory and optional asset usage")

    st.markdown("""
    <div style='background-color: #e3f2fd; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h4>Objective: Create actionable guidelines for market teams</h4>
    <p>Based on combined analysis of media usage, spend data, and consumer research,
    we recommend the following asset usage framework:</p>
    </div>
    """, unsafe_allow_html=True)

    # Auto-generate recommendations based on data
    must_use = master_df[
        (master_df['Recognition'] >= 0.40) &
        (master_df['Positive Associations'] >= 0.47) &
        (master_df['Overall Usage'] >= 0.50)
    ].sort_values('Recognition', ascending=False)

    recommended = master_df[
        ((master_df['Recognition'] >= 0.35) | (master_df['Uniqueness'] >= 0.60)) &
        (master_df['Positive Associations'] > master_df['Negative Associations'])
    ].sort_values(['Recognition', 'Uniqueness'], ascending=False)
    recommended = recommended[~recommended['Element'].isin(must_use['Element'])]

    requires_attention = master_df[
        (master_df['Recognition'] < 0.40) &
        (master_df['Total Investment'] > master_df['Total Investment'].median())
    ]

    # Display recommendations
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### ✅ MUST-USE Assets (Non-Negotiable)")
        st.success(f"**{len(must_use)} assets meet criteria:** High Recognition (≥40%) + Positive Sentiment + High Usage (≥50%)")

        for idx, row in must_use.iterrows():
            with st.expander(f"**{row['Element']}** - Recognition: {row['Recognition']:.0%} | Usage: {row['Overall Usage']:.0%}"):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Recognition", f"{row['Recognition']:.0%}")
                    st.metric("Uniqueness", f"{row['Uniqueness']:.0%}")
                with col_b:
                    st.metric("Positive", f"{row['Positive Associations']:.0%}")
                    st.metric("Negative", f"{row['Negative Associations']:.0%}")
                with col_c:
                    st.metric("Usage", f"{row['Overall Usage']:.0%}")
                    st.metric("Investment", f"€{row['Total Investment']:,.0f}")

                st.markdown("**Rationale:**")
                st.write(f"• Strong consumer recognition ({row['Recognition']:.0%})")
                st.write(f"• Positive associations outweigh negative ({row['Positive Associations']:.0%} vs {row['Negative Associations']:.0%})")
                st.write(f"• Already widely used across campaigns ({row['Overall Usage']:.0%})")

        st.markdown("---")

        st.markdown("### ⭐ RECOMMENDED Assets (Strongly Encouraged)")
        st.info(f"**{len(recommended)} assets show strong potential:** Good recognition or uniqueness + positive sentiment")

        for idx, row in recommended.iterrows():
            st.write(f"**{row['Element']}** - Recognition: {row['Recognition']:.0%} | Uniqueness: {row['Uniqueness']:.0%} | {row['POS/NEG']}")

        st.markdown("---")

        st.markdown("### ⚠️ REQUIRES ATTENTION")
        st.warning(f"**{len(requires_attention)} assets** have low recognition despite significant investment")

        for idx, row in requires_attention.iterrows():
            st.write(f"**{row['Element']}** - Recognition: {row['Recognition']:.0%} | Investment: €{row['Total Investment']:,.0f}")

    with col2:
        st.markdown("### 📋 Quick Reference")

        st.markdown("#### Must-Use (Non-Negotiable)")
        for idx, row in must_use.iterrows():
            st.success(f"✓ {row['Element']}")

        st.markdown("#### Recommended")
        for idx, row in recommended.head(5).iterrows():
            st.info(f"⭐ {row['Element']}")

        st.markdown("#### Review Needed")
        for idx, row in requires_attention.iterrows():
            st.warning(f"⚠️ {row['Element']}")

        st.markdown("---")

        # Download guidelines
        guidelines_text = "# Škoda Brand Asset Usage Guidelines\n\n"
        guidelines_text += "## MUST-USE Assets (Non-Negotiable)\n"
        for idx, row in must_use.iterrows():
            guidelines_text += f"- {row['Element']}: {row['Recognition']:.0%} recognition\n"
        guidelines_text += "\n## RECOMMENDED Assets\n"
        for idx, row in recommended.iterrows():
            guidelines_text += f"- {row['Element']}: {row['Recognition']:.0%} recognition, {row['Uniqueness']:.0%} uniqueness\n"

        st.download_button(
            label="📥 Download Guidelines (TXT)",
            data=guidelines_text,
            file_name="skoda_brand_guidelines.txt",
            mime="text/plain"
        )

# ==================== TAB 4: FUTURE-PROOFING ====================
with tab4:
    st.header("🔮 Future-Proofing Opportunities")
    st.caption("Actionable steps to improve long-term memorability and brand equity")

    st.markdown("""
    <div style='background-color: #fff3e0; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h4>Objective: Identify opportunities to strengthen brand assets over time</h4>
    <p>Analysis of underutilized assets, investment optimization, and consistency improvements</p>
    </div>
    """, unsafe_allow_html=True)

    # High Potential Assets (underutilized)
    st.markdown("### 💎 High Potential Assets: Underutilized Opportunities")

    high_potential = master_df[
        (master_df['Uniqueness'] >= 0.60) &
        (master_df['Overall Usage'] < 0.40) &
        (master_df['Positive Associations'] > master_df['Negative Associations'])
    ].sort_values('Uniqueness', ascending=False)

    if len(high_potential) > 0:
        st.success(f"**{len(high_potential)} assets identified** with high uniqueness but low current usage")

        for idx, row in high_potential.iterrows():
            with st.expander(f"**{row['Element']}** - Uniqueness: {row['Uniqueness']:.0%} | Current Usage: {row['Overall Usage']:.0%}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Uniqueness Score", f"{row['Uniqueness']:.0%}", "High differentiator")
                    st.metric("Current Usage", f"{row['Overall Usage']:.0%}", "Underutilized")
                    st.metric("Recognition", f"{row['Recognition']:.0%}")

                with col2:
                    st.metric("Positive Associations", f"{row['Positive Associations']:.0%}")
                    st.metric("Current Investment", f"€{row['Total Investment']:,.0f}")
                    st.metric("Recognition ROI", f"{row['Recognition ROI']:.2f}")

                st.markdown("**💡 Opportunity:**")
                st.write(f"• High uniqueness ({row['Uniqueness']:.0%}) suggests strong differentiation potential")
                st.write(f"• Currently used in only {row['Overall Usage']:.0%} of campaigns")
                st.write(f"• Positive sentiment ({row['Positive Associations']:.0%}) indicates consumer receptivity")

                st.markdown("**📈 Recommendations:**")
                st.write(f"• Increase usage from {row['Overall Usage']:.0%} to 50%+ of campaigns")
                st.write(f"• Integrate into high-visibility placements")
                st.write("• Create consistency guidelines for market teams")
    else:
        st.info("No significantly underutilized high-potential assets identified")

    st.markdown("---")

    # Investment Reallocation Opportunities
    st.markdown("### 💰 Investment Optimization")

    # Calculate efficiency scores
    master_df['Efficiency Score'] = (master_df['Recognition'] * master_df['Uniqueness']) / (master_df['Total Investment'] / 1000000)
    master_df['Efficiency Score'] = master_df['Efficiency Score'].fillna(0)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📈 High Efficiency (Underfunded)")
        high_efficiency = master_df.nlargest(3, 'Efficiency Score')

        for idx, row in high_efficiency.iterrows():
            if row['Total Investment'] < master_df['Total Investment'].median():
                st.success(f"**{row['Element']}**")
                st.write(f"• Efficiency Score: {row['Efficiency Score']:.2f}")
                st.write(f"• Current Investment: €{row['Total Investment']:,.0f}")
                st.write(f"• **Opportunity:** Increase investment to amplify impact")
                st.write("")

    with col2:
        st.markdown("#### 📉 Low Efficiency (Overfunded)")
        low_efficiency = master_df.nsmallest(3, 'Efficiency Score')

        for idx, row in low_efficiency.iterrows():
            if row['Total Investment'] > master_df['Total Investment'].median():
                st.warning(f"**{row['Element']}**")
                st.write(f"• Efficiency Score: {row['Efficiency Score']:.2f}")
                st.write(f"• Current Investment: €{row['Total Investment']:,.0f}")
                st.write(f"• **Opportunity:** Re-evaluate investment level")
                st.write("")

    st.markdown("---")

    # Consistency Improvements
    st.markdown("### 🎯 Consistency Improvement Opportunities")

    # Calculate consistency across markets
    consistency_data = []
    for element in brand_elements:
        usage_by_market = []
        for market in audit_df['Market'].unique():
            market_df = audit_df[audit_df['Market'] == market]
            usage = market_df[element].sum() / len(market_df) if len(market_df) > 0 else 0
            usage_by_market.append(usage)

        std_dev = pd.Series(usage_by_market).std()
        avg_usage = pd.Series(usage_by_market).mean()

        consistency_data.append({
            'Element': element,
            'Std Dev': std_dev,
            'Avg Usage': avg_usage,
            'Consistency Score': 1 - std_dev  # Higher is more consistent
        })

    consistency_df = pd.DataFrame(consistency_data).sort_values('Consistency Score')

    st.info("**Assets requiring consistency guidelines:** High usage variation across markets")

    for idx, row in consistency_df.head(5).iterrows():
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.write(f"**{row['Element']}**")
        with col2:
            st.metric("Variation", f"{row['Std Dev']:.2f}")
        with col3:
            st.write(f"Avg usage: {row['Avg Usage']:.0%} - Create standardized usage guidelines")

    st.markdown("---")

    # Action Plan Summary
    st.markdown("### 📋 Future-Proofing Action Plan")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Short-term (0-6 months)")
        st.write("1. **Increase must-use asset deployment**")
        for idx, row in must_use.head(3).iterrows():
            st.write(f"   • Ensure {row['Element']} in 80%+ of campaigns")

        st.write("2. **Test high-potential assets**")
        for idx, row in high_potential.head(2).iterrows():
            st.write(f"   • Pilot {row['Element']} in 50% more campaigns")

        st.write("3. **Standardize market guidelines**")
        st.write(f"   • Create usage frameworks for inconsistent assets")

    with col2:
        st.markdown("#### Long-term (6-18 months)")
        st.write("1. **Investment reallocation**")
        st.write("   • Shift budget from low-ROI to high-ROI assets")

        st.write("2. **Build uniqueness equity**")
        for idx, row in high_potential.head(2).iterrows():
            st.write(f"   • Amplify {row['Element']} for differentiation")

        st.write("3. **Continuous monitoring**")
        st.write("   • Track recognition metrics quarterly")
        st.write("   • Adjust based on performance data")

# ==================== TAB 5: DEEP DIVE ANALYSIS ====================
with tab5:
    st.header("🔍 Deep Dive Analysis")
    st.caption("Detailed breakdowns and custom filtering")

    # Filters
    st.markdown("### Filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_market = st.selectbox("Market", ['All'] + sorted(audit_df['Market'].unique().tolist()))
    with col2:
        selected_medium = st.selectbox("Medium", ['All'] + sorted(audit_df['Medium'].unique().tolist()))
    with col3:
        selected_placement = st.selectbox("Placement", ['All'] + sorted(audit_df['Placement'].unique().tolist()))

    # Apply filters
    filtered_df = audit_df.copy()
    if selected_market != 'All':
        filtered_df = filtered_df[filtered_df['Market'] == selected_market]
    if selected_medium != 'All':
        filtered_df = filtered_df[filtered_df['Medium'] == selected_medium]
    if selected_placement != 'All':
        filtered_df = filtered_df[filtered_df['Placement'] == selected_placement]

    st.info(f"Showing {len(filtered_df)} of {len(audit_df)} ads")

    st.markdown("---")

    # Investment breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Investment by Element")
        invest_data = []
        for element in brand_elements:
            element_df = filtered_df[filtered_df[element] == True]
            invest_data.append({
                'Element': element,
                'Investment': element_df['Spend'].sum()
            })
        invest_df = pd.DataFrame(invest_data).sort_values('Investment', ascending=True)

        fig_invest = px.bar(
            invest_df,
            y='Element',
            x='Investment',
            orientation='h',
            text=invest_df['Investment'].apply(lambda x: f'€{x:,.0f}'),
            title='Total Investment by Element'
        )
        st.plotly_chart(fig_invest, use_container_width=True)

    with col2:
        st.markdown("#### Usage Frequency")
        usage_data = []
        total = len(filtered_df)
        for element in brand_elements:
            count = filtered_df[element].sum()
            usage_data.append({
                'Element': element,
                'Usage': count / total if total > 0 else 0
            })
        usage_df = pd.DataFrame(usage_data).sort_values('Usage', ascending=True)

        fig_usage = px.bar(
            usage_df,
            y='Element',
            x='Usage',
            orientation='h',
            text=usage_df['Usage'].apply(lambda x: f'{x:.0%}'),
            title='Element Usage Frequency'
        )
        st.plotly_chart(fig_usage, use_container_width=True)

    st.markdown("---")

    # Personality attributes
    st.markdown("### Brand Personality Comparison")

    selected_elements = st.multiselect(
        "Select elements to compare:",
        brand_elements,
        default=brand_elements[:3]
    )

    if selected_elements:
        personality_data = []
        for element in selected_elements:
            research = research_data[element]
            personality_data.append({
                'Element': element,
                'Bold': research['bold'],
                'Stylish': research['stylish'],
                'Modern': research['modern']
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
            title='Brand Personality Attributes'
        )
        st.plotly_chart(fig_personality, use_container_width=True)

# ==================== TAB 6: DATA EXPLORER ====================
with tab6:
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
        research_display = []
        for element, data in research_data.items():
            research_display.append({
                'Element': element,
                'Recognition': data['recognition'],
                'Uniqueness': data['uniqueness'],
                'Positive Associations': data['positive'],
                'Negative Associations': data['negative'],
                'Bold': data['bold'],
                'Stylish': data['stylish'],
                'Modern': data['modern']
            })
        research_display_df = pd.DataFrame(research_display)

        st.dataframe(research_display_df.style.format({
            'Recognition': '{:.1%}',
            'Uniqueness': '{:.1%}',
            'Positive Associations': '{:.1%}',
            'Negative Associations': '{:.1%}',
            'Bold': '{:.1%}',
            'Stylish': '{:.1%}',
            'Modern': '{:.1%}'
        }), use_container_width=True)

    with tab_c:
        st.markdown("### Combined Metrics")
        st.dataframe(master_df, use_container_width=True)

        csv = master_df.to_csv(index=False)
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
<p><b>Škoda Brand Intelligence Dashboard</b> | Powered by Streamlit</p>
<p>Data Sources: 250915_SKO_Ads Overview.xlsx (Comms Audit) | P045556_ALL_Tables_20251016_Private.xlsx (Quantitative Research)</p>
</div>
""", unsafe_allow_html=True)
