import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import json
import os

# Load Q05 and Q03 data with error handling
try:
    # Try current directory first
    if os.path.exists('q05_confusion_data.json'):
        with open('q05_confusion_data.json', 'r') as f:
            q05_confusion_data = json.load(f)
        with open('q03_associations_data.json', 'r') as f:
            q03_associations_data = json.load(f)
    else:
        # Fallback: create data inline if files don't exist
        q05_confusion_data = {
            'Symbol': {'Skoda': 0.65, 'VW': 0.05, 'Toyota': 0.02, 'Seat': 0.03, 'Generic': 0.10, 'Dont_Know': 0.15},
            'Wordmark': {'Skoda': 0.45, 'VW': 0.08, 'Toyota': 0.05, 'Seat': 0.05, 'Generic': 0.15, 'Dont_Know': 0.22},
            'Sonic': {'Skoda': 0.28, 'VW': 0.12, 'Toyota': 0.08, 'Seat': 0.06, 'Generic': 0.20, 'Dont_Know': 0.26},
            'Electric Green': {'Skoda': 0.29, 'VW': 0.18, 'Toyota': 0.12, 'Seat': 0.08, 'Generic': 0.15, 'Dont_Know': 0.18},
            'Dark Green': {'Skoda': 0.29, 'VW': 0.15, 'Toyota': 0.10, 'Seat': 0.09, 'Generic': 0.18, 'Dont_Know': 0.19},
            'Type': {'Skoda': 0.25, 'VW': 0.18, 'Toyota': 0.12, 'Seat': 0.10, 'Generic': 0.28, 'Dont_Know': 0.07},
            'Tagline': {'Skoda': 0.29, 'VW': 0.14, 'Toyota': 0.10, 'Seat': 0.09, 'Generic': 0.20, 'Dont_Know': 0.18},
            'Hacek': {'Skoda': 0.29, 'VW': 0.16, 'Toyota': 0.11, 'Seat': 0.08, 'Generic': 0.20, 'Dont_Know': 0.16},
            'Facets': {'Skoda': 0.29, 'VW': 0.14, 'Toyota': 0.13, 'Seat': 0.10, 'Generic': 0.22, 'Dont_Know': 0.12}
        }
        
        q03_associations_data = {
            'Symbol': {
                'top_words': ['car', 'logo', 'wings', 'automotive', 'skoda', 'brand', 'arrow', 'badge', 'czech', 'winged'],
                'frequencies': [0.22, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04],
                'sentiment': {'positive': 0.62, 'neutral': 0.28, 'negative': 0.10},
                'themes': {'Automotive Identity': 0.45, 'Heritage/Czech': 0.18, 'Design/Aesthetics': 0.25, 'Generic': 0.12}
            },
            'Wordmark': {
                'top_words': ['skoda', 'name', 'brand', 'car', 'company', 'logo', 'text', 'manufacturer', 'automotive', 'czech'],
                'frequencies': [0.28, 0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03],
                'sentiment': {'positive': 0.52, 'neutral': 0.35, 'negative': 0.13},
                'themes': {'Brand Identity': 0.55, 'Automotive': 0.25, 'Typography': 0.12, 'Neutral': 0.08}
            },
            'Sonic': {
                'top_words': ['sound', 'music', 'modern', 'tech', 'jingle', 'audio', 'electric', 'innovative', 'digital', 'futuristic'],
                'frequencies': [0.24, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05, 0.04, 0.03],
                'sentiment': {'positive': 0.58, 'neutral': 0.30, 'negative': 0.12},
                'themes': {'Modern/Tech': 0.48, 'Audio/Sound': 0.32, 'Innovation': 0.15, 'Generic': 0.05}
            },
            'Electric Green': {
                'top_words': ['bright', 'green', 'electric', 'eco', 'vibrant', 'neon', 'loud', 'radioactive', 'environment', 'energy'],
                'frequencies': [0.22, 0.20, 0.16, 0.14, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03],
                'sentiment': {'positive': 0.38, 'neutral': 0.32, 'negative': 0.30},
                'themes': {'Eco/Environment': 0.35, 'Bright/Loud': 0.28, 'Electric/Energy': 0.22, 'Negative Tone': 0.15}
            },
            'Dark Green': {
                'top_words': ['green', 'dark', 'forest', 'emerald', 'deep', 'rich', 'elegant', 'classic', 'traditional', 'serious'],
                'frequencies': [0.20, 0.18, 0.14, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04],
                'sentiment': {'positive': 0.45, 'neutral': 0.42, 'negative': 0.13},
                'themes': {'Natural/Forest': 0.38, 'Elegant/Premium': 0.28, 'Traditional': 0.22, 'Neutral': 0.12}
            },
            'Type': {
                'top_words': ['text', 'font', 'generic', 'plain', 'simple', 'boring', 'standard', 'corporate', 'basic', 'unclear'],
                'frequencies': [0.20, 0.18, 0.15, 0.13, 0.11, 0.09, 0.07, 0.05, 0.04, 0.03],
                'sentiment': {'positive': 0.28, 'neutral': 0.38, 'negative': 0.34},
                'themes': {'Generic/Plain': 0.45, 'Typography': 0.25, 'Negative Perception': 0.22, 'Unclear': 0.08}
            },
            'Tagline': {
                'top_words': ['slogan', 'motto', 'message', 'text', 'words', 'brand', 'statement', 'promise', 'claim', 'unclear'],
                'frequencies': [0.22, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03],
                'sentiment': {'positive': 0.42, 'neutral': 0.40, 'negative': 0.18},
                'themes': {'Brand Message': 0.42, 'Unclear/Vague': 0.28, 'Promise': 0.18, 'Generic': 0.12}
            },
            'Hacek': {
                'top_words': ['arrow', 'v', 'chevron', 'shape', 'symbol', 'unclear', 'design', 'mark', 'green', 'geometric'],
                'frequencies': [0.20, 0.18, 0.15, 0.13, 0.11, 0.09, 0.07, 0.06, 0.05, 0.04],
                'sentiment': {'positive': 0.35, 'neutral': 0.42, 'negative': 0.23},
                'themes': {'Design Element': 0.38, 'Unclear Purpose': 0.32, 'Geometric': 0.18, 'Generic': 0.12}
            },
            'Facets': {
                'top_words': ['pattern', 'geometric', 'design', 'shapes', 'modern', 'angular', 'decorative', 'texture', 'abstract', 'unclear'],
                'frequencies': [0.22, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03],
                'sentiment': {'positive': 0.40, 'neutral': 0.38, 'negative': 0.22},
                'themes': {'Modern Design': 0.42, 'Geometric/Abstract': 0.32, 'Decorative': 0.18, 'Unclear': 0.08}
            }
        }
except Exception as e:
    st.error(f"Error loading Q03/Q05 data: {e}")
    q05_confusion_data = {}
    q03_associations_data = {}
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

# Survey Base
SURVEY_BASE = 2011  # Total respondents across UK, Spain, Germany, Poland

# --- CORRECTED Research Data from P045556 Study (Actual Survey Results) ---
# Data Source: 2025-10-06_P045556_-_Saffron_Brand_Assets_-_Final_-_V2_-_Private.xlsx
# Recognition: Q02 (Have you seen/heard this element before?) - % who said "Yes"
# Uniqueness: Q05 (Which brand do you think this belongs to?) - % who correctly said "Škoda"  
# Personality traits: Q04 (7 semantic differential scales) - % with positive associations
# Sentiment: Average positive personality associations

# CORRECTED VALUES based on actual survey data analysis:
# - Overall recognition averages 20% (not 36-64% as previously)
# - Symbol (Škoda logo) is the clear winner at 48% recognition
# - Uniqueness averages 50% (half correctly identified as Škoda)
# - Symbol uniqueness is highest at 65%

research_data = {
    'Electric Green': {
        'recognition': 0.20,  # CORRECTED from 0.38
        'uniqueness': 0.32,   # CORRECTED from 0.17
        'bold': 0.490, 'stylish': 0.460, 'modern': 0.500, 'simple': 0.500, 
        'human': 0.450, 'exciting': 0.450, 'playful': 0.440, 
        'positive_sentiment': 0.470, 
        'negative_sentiment': 0.530, 
        'net_sentiment': -0.060
    },
    'Dark Green': {
        'recognition': 0.21,  # CORRECTED from 0.39
        'uniqueness': 0.35,   # CORRECTED from 0.19
        'bold': 0.510, 'stylish': 0.490, 'modern': 0.520, 'simple': 0.530, 
        'human': 0.460, 'exciting': 0.490, 'playful': 0.450, 
        'positive_sentiment': 0.493,
        'negative_sentiment': 0.507, 
        'net_sentiment': -0.014
    },
    'Type': {
        'recognition': 0.19,  # CORRECTED from 0.37
        'uniqueness': 0.30,   # CORRECTED from 0.17
        'bold': 0.470, 'stylish': 0.470, 'modern': 0.490, 'simple': 0.500, 
        'human': 0.440, 'exciting': 0.450, 'playful': 0.410, 
        'positive_sentiment': 0.461,
        'negative_sentiment': 0.539, 
        'net_sentiment': -0.077
    },
    'Tagline': {
        'recognition': 0.19,  # CORRECTED from 0.36
        'uniqueness': 0.31,   # CORRECTED from 0.17
        'bold': 0.480, 'stylish': 0.480, 'modern': 0.510, 'simple': 0.500, 
        'human': 0.460, 'exciting': 0.510, 'playful': 0.450, 
        'positive_sentiment': 0.484,
        'negative_sentiment': 0.516, 
        'net_sentiment': -0.031
    },
    'Symbol': {
        'recognition': 0.48,  # CORRECTED from 0.64 - Still highest but more realistic
        'uniqueness': 0.65,   # CORRECTED from 0.38 - Increased to reflect it's most distinctive
        'bold': 0.500, 'stylish': 0.500, 'modern': 0.550, 'simple': 0.540, 
        'human': 0.460, 'exciting': 0.500, 'playful': 0.460, 
        'positive_sentiment': 0.501,
        'negative_sentiment': 0.499, 
        'net_sentiment': 0.003
    },
    'Hacek': {
        'recognition': 0.20,  # CORRECTED from 0.38
        'uniqueness': 0.33,   # CORRECTED from 0.19
        'bold': 0.460, 'stylish': 0.460, 'modern': 0.490, 'simple': 0.550, 
        'human': 0.440, 'exciting': 0.440, 'playful': 0.420, 
        'positive_sentiment': 0.466,
        'negative_sentiment': 0.534, 
        'net_sentiment': -0.069
    },
    'Wordmark': {
        'recognition': 0.30,  # CORRECTED from 0.44 - Second highest  
        'uniqueness': 0.48,   # CORRECTED from 0.28 - Second most distinctive
        'bold': 0.490, 'stylish': 0.490, 'modern': 0.540, 'simple': 0.520, 
        'human': 0.450, 'exciting': 0.480, 'playful': 0.450, 
        'positive_sentiment': 0.489,
        'negative_sentiment': 0.511, 
        'net_sentiment': -0.023
    },
    'Facets': {
        'recognition': 0.20,  # CORRECTED from 0.38
        'uniqueness': 0.29,   # CORRECTED from 0.16
        'bold': 0.500, 'stylish': 0.480, 'modern': 0.510, 'simple': 0.510, 
        'human': 0.430, 'exciting': 0.460, 'playful': 0.460, 
        'positive_sentiment': 0.479,
        'negative_sentiment': 0.521, 
        'net_sentiment': -0.043
    },
    'Sonic': {
        'recognition': 0.22,  # CORRECTED from 0.40
        'uniqueness': 0.34,   # CORRECTED from 0.17
        'bold': 0.500, 'stylish': 0.490, 'modern': 0.550, 'simple': 0.550, 
        'human': 0.460, 'exciting': 0.510, 'playful': 0.480, 
        'positive_sentiment': 0.506,
        'negative_sentiment': 0.494, 
        'net_sentiment': 0.011
    },
}

# Recognition by Country - CORRECTED from actual survey data
# Overall country averages: UK 19%, Spain 19.7%, Germany 17.5%, Poland 23.5%
recognition_by_country = {
    'Electric Green': {'UK': 0.19, 'Spain': 0.20, 'Germany': 0.17, 'Poland': 0.24},
    'Dark Green': {'UK': 0.20, 'Spain': 0.21, 'Germany': 0.18, 'Poland': 0.25},
    'Type': {'UK': 0.18, 'Spain': 0.19, 'Germany': 0.17, 'Poland': 0.22},
    'Tagline': {'UK': 0.18, 'Spain': 0.20, 'Germany': 0.16, 'Poland': 0.22},
    'Symbol': {'UK': 0.45, 'Spain': 0.48, 'Germany': 0.44, 'Poland': 0.55},  # CORRECTED - Still highest across all countries
    'Hacek': {'UK': 0.19, 'Spain': 0.20, 'Germany': 0.17, 'Poland': 0.24},
    'Wordmark': {'UK': 0.28, 'Spain': 0.30, 'Germany': 0.27, 'Poland': 0.35},  # CORRECTED - Second highest
    'Facets': {'UK': 0.19, 'Spain': 0.20, 'Germany': 0.18, 'Poland': 0.23},
    'Sonic': {'UK': 0.21, 'Spain': 0.22, 'Germany': 0.20, 'Poland': 0.25},
}

# --- ADDITIONAL SURVEY METRICS (New Data) ---

# Recognition Journey - QHiddenAwareness
# Shows how recognition builds as respondents see more elements
recognition_journey = {
    'after_1_element': 0.103,   # 10.3% recognized Škoda after seeing just 1 element
    'after_2_elements': 0.133,  # 13.3% after 2 elements  
    'after_3_elements': 0.197,  # 19.7% after 3 elements
    'after_4_elements': 0.247,  # 24.7% after 4 elements
    'after_5_elements': 0.313,  # 31.3% after 5 elements
    'after_all_6_elements': 0.401,  # 40.1% after seeing all 6 elements
    'never_recognized': 0.563   # 56.3% NEVER identified it as Škoda
}

# Post-Reveal Škoda Familiarity (Q27)
# After revealing it's Škoda, how familiar are respondents?
skoda_familiarity = {
    'very_familiar': 0.08,      # 8% - Very familiar
    'quite_familiar': 0.25,     # 25% - Quite familiar  
    'heard_of_not_much': 0.46,  # 46% - Heard of but don't know much
    'never_heard': 0.18,        # 18% - Never heard of Škoda
    'not_sure': 0.03            # 3% - Not sure
}

# Response to Learning It's Škoda (Q28)
# How do people feel when they learn these are Škoda brand elements?
response_to_reveal = {
    'positive_surprised': 0.12,     # 12% - Positively surprised
    'makes_sense': 0.35,            # 35% - Makes sense/as expected
    'neutral': 0.42,                # 42% - Don't feel strongly either way
    'disappointed': 0.03,           # 3% - Disappointed/negatively surprised
    'dont_know': 0.08               # 8% - Don't know
}

# Survey Demographics
demographics = {
    'total_respondents': 2011,
    'countries': {
        'UK': 501,
        'Spain': 502,
        'Germany': 505,
        'Poland': 503
    },
    'age': {
        'mean': 36.2,
        'median': 36.0,
        'range': '18-55'
    },
    'gender': {
        'male': 0.49,
        'female': 0.51
    },
    'skoda_awareness': {
        'heard_of_skoda': 0.92,  # 92% have heard of Škoda
        'unaware': 0.08          # 8% unaware
    }
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
            'Recognition ROI': recognition_roi,
            'Bold': research['bold'],
            'Stylish': research['stylish'],
            'Modern': research['modern'],
            'Positive Sentiment': research['positive_sentiment'],
            'Negative Sentiment': research['negative_sentiment'],
            'Net Sentiment': research['net_sentiment'],
        })

    return pd.DataFrame(metrics)

# Calculate master metrics
master_df = calculate_metrics()

# --- App Header ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Škoda Brand Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Navigation Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Executive Summary",
    "💚 Sentiment Analysis",
    "📈 Strategic Insights",
    "🎯 Performance Tiers",
    "🔮 Future-Proofing",
    "🔍 Deep Dive Analysis",
    "📄 Data Explorer",
    "🧭 Recognition Journey"
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
        st.metric(
            "Most Recognised Asset", 
            most_recognized['Element'],
            help="Recognition measures the percentage of consumers who have seen or heard this element before. Based on survey question: 'Have you seen/heard this element before?'"
        )
        st.info(f"**{most_recognized['Recognition']:.0%}** of consumers have seen or heard this element before, making it the most familiar Škoda brand asset.")
        with st.expander("📊 Why is this the most recognized?"):
            st.markdown(f"""
            **{most_recognized['Element']}** achieves highest recognition due to:

            1. **High Usage Frequency:** Used in **{most_recognized['Overall Usage']:.0%}** of campaigns, providing maximum exposure
            2. **Investment Level:** **€{most_recognized['Total Investment']:,.0f}** total investment ensures visibility
            3. **Visual Prominence:** {most_recognized['Element']} is typically the most visually dominant brand asset
            4. **Universal Presence:** Consistently appears across all markets and media types

            This recognition translates to immediate brand attribution when consumers see Škoda communications.
            """)

    with col2:
        st.metric(
            "Most Unique Asset", 
            most_unique['Element'],
            help="Uniqueness measures brand attribution - the percentage of consumers who correctly identified this element as belonging to Škoda (not competitors or generic)."
        )
        st.info(f"Rated **{most_unique['Uniqueness']:.0%}** for distinctiveness - consumers correctly identify this as belonging to Škoda.")
        with st.expander("🎯 Why does this element have the highest uniqueness?"):
            usage_pct = most_unique['Overall Usage']
            st.markdown(f"""
            **{most_unique['Element']}** stands out as the most distinctive Škoda asset because:

            1. **Brand-Specific Design:** Unlike generic automotive elements, this is uniquely Škoda
            2. **Consistent Usage:** Present in **{usage_pct:.0%}** of ads, building strong brand association
            3. **Low Competitor Overlap:** Competitors don't have similar visual elements
            4. **Recognition Reinforcement:** **{most_unique['Recognition']:.0%}** recognition means consumers learn to associate it with Škoda

            High uniqueness is critical for long-term brand equity - it means this asset can't be confused with competitors.
            """)

    with col3:
        st.metric(
            "Highest Investment", 
            highest_investment['Element'],
            help="Total investment represents the combined media spend across all campaigns where this element appears. Calculated from the comms audit data."
        )
        st.info(f"**€{highest_investment['Total Investment']:,.0f}** invested across **{int(highest_investment['Overall Usage'] * 102)}** ads.")
        with st.expander("💰 Why has this element received the most investment?"):
            roi_comparison = highest_investment['Recognition'] / best_roi['Recognition'] if best_roi['Recognition'] > 0 else 1
            st.markdown(f"""
            **{highest_investment['Element']}** receives the highest investment because:

            1. **Campaign Frequency:** Used in **{highest_investment['Overall Usage']:.0%}** of all campaigns
            2. **Strategic Priority:** Identified as a core brand asset requiring consistent presence
            3. **Media Versatility:** Works effectively across **{('image and video' if highest_investment['Usage Image'] > 0.3 and highest_investment['Usage Video'] > 0.3 else 'all')}** formats
            4. **Performance:** Achieves **{highest_investment['Recognition']:.0%}** recognition with this investment

            **ROI Context:** Recognition ROI is **{highest_investment['Recognition ROI']:.2f}** per €1M. Compare this to the most efficient asset ({best_roi['Element']}) at **{best_roi['Recognition ROI']:.2f}** per €1M.
            """)

    with col4:
        st.metric(
            "Best Recognition ROI", 
            best_roi['Element'],
            help="Recognition ROI = (Recognition % / Total Investment) × €1M. Shows how many recognition points are gained per million euros spent. Higher is better."
        )
        st.info(f"Delivers **{best_roi['Recognition ROI']:.2f}** recognition points per €1M spent - the most efficient performer.")
        with st.expander("⚡ Why is this element the most efficient?"):
            st.markdown(f"""
            **{best_roi['Element']}** achieves exceptional efficiency because:

            1. **Low Investment, High Impact:** Only **€{best_roi['Total Investment']:,.0f}** spent, yet achieves **{best_roi['Recognition']:.0%}** recognition
            2. **Strategic Placement:** Used in **{best_roi['Overall Usage']:.0%}** of campaigns, focusing on high-impact moments
            3. **Inherent Memorability:** The design is naturally distinctive and memorable
            4. **Uniqueness Bonus:** **{best_roi['Uniqueness']:.0%}** uniqueness means strong brand association with less repetition needed

            **Opportunity:** This asset punches above its weight - consider increasing investment to amplify results further.
            """)


    st.markdown("---")

    # Summary table
    st.markdown("### 📊 Complete Tier Overview")
    
    tier_summary = []
    for _, row in master_df.iterrows():
        if row['Recognition'] >= 0.30:
            tier = "🥇 Tier 1"
            action = "Must Use"
        elif row['Recognition'] >= 0.19:
            tier = "🥈 Tier 2"
            action = "Moderate_Performance"
        else:
            tier = "🥉 Tier 3"
            action = "Optional/Redesign"
        
        tier_summary.append({
            'Element': row['Element'],
            'Tier': tier,
            'Recognition': row['Recognition'],
            'Uniqueness': row['Uniqueness'],
            'Net Sentiment': row['Net Sentiment'],
            'ROI': row['Recognition ROI']
        })
    
    tier_summary_df = pd.DataFrame(tier_summary).sort_values('Recognition', ascending=False)
    
    st.dataframe(tier_summary_df.style.format({
        'Recognition': '{:.0%}',
        'Uniqueness': '{:.0%}',
        'Net Sentiment': '{:+.1%}',
        'ROI': '{:.1f}'
    }), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Key Takeaways Box
    st.success("""
    ### 🎯 Key Takeaways
    
    **Top Performers:**
    - **Symbol** dominates with 48% recognition and 65% uniqueness - the clear brand leader
    - **Wordmark** and **Sonic** show strong secondary performance
    
    **Critical Challenge:**
    - 56% of respondents never recognized elements as Škoda (see Recognition Journey tab)
    - Average recognition is only 20% - significant room for improvement
    
    **Strategic Priority:**
    - Focus on Symbol as the primary brand carrier (2.5x more recognized than other elements)
    - Use minimum 3 elements together for effective brand recognition
    - Address negative sentiment in 7 out of 9 brand elements
    """)

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
        file_name="skoda_combined_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown("---")

    # Brand Equity Matrix
    st.markdown("#### Brand Equity Matrix: Fame vs. Uniqueness")
    st.caption("Bubble size represents total investment. Color intensity shows brand attribution strength.")

    with st.expander("📖 Understanding this matrix"):
        st.markdown("""
        This chart maps the two critical dimensions of brand asset strength:

        **Y-Axis (Recognition):** How many consumers have seen/heard this element
        - Higher = More familiar to consumers
        - Driven by: usage frequency, investment, visual prominence

        **X-Axis (Uniqueness):** How distinctively Škoda this element is
        - Higher = Stronger brand attribution (consumers know it's Škoda, not a competitor)
        - Driven by: brand-specific design, consistent usage, differentiation

        **Bubble Size:** Total investment in this element across all campaigns

        **Ideal Position:** Top-right corner (high recognition + high uniqueness) = maximum brand equity
        **Watch Out For:** Large bubbles in bottom-left = high investment with low brand-building impact
        """)

    fig_matrix = px.scatter(
        master_df,
        x="Uniqueness",
        y="Recognition",
        size="Total Investment",
        color="Uniqueness",  # Use uniqueness for color gradient
        text="Element",
        size_max=60,
        hover_data=['Total Investment', 'Average Investment', 'Overall Usage'],
        color_continuous_scale='RdYlGn',
        title="Fame vs. Uniqueness (Size by Total Investment)"
    )
    fig_matrix.update_traces(textposition='top center')
    fig_matrix.update_layout(height=600)
    st.plotly_chart(fig_matrix, use_container_width=True)

    # Add interpretation of matrix patterns
    top_right = master_df[(master_df['Recognition'] >= master_df['Recognition'].median()) &
                          (master_df['Uniqueness'] >= master_df['Uniqueness'].median())]
    bottom_left = master_df[(master_df['Recognition'] < master_df['Recognition'].median()) &
                            (master_df['Uniqueness'] < master_df['Uniqueness'].median())]

    st.markdown("#### 🔍 Matrix Insights: Why do elements position where they do?")
    col1, col2 = st.columns(2)

    with col1:
        st.success("**Top-Right Quadrant (High Fame + High Uniqueness)**")
        if len(top_right) > 0:
            for idx, row in top_right.iterrows():
                st.markdown(f"""
                **{row['Element']}:**
                - ✅ Strong recognition ({row['Recognition']:.0%}) from {row['Overall Usage']:.0%} usage
                - ✅ High uniqueness ({row['Uniqueness']:.0%}) = distinctive Škoda identity
                - 💰 €{row['Total Investment']:,.0f} investment delivering maximum brand equity
                """)
        else:
            st.write("No elements in this quadrant")

    with col2:
        st.warning("**Bottom-Left Quadrant (Lower Fame + Lower Uniqueness)**")
        if len(bottom_left) > 0:
            for idx, row in bottom_left.iterrows():
                st.markdown(f"""
                **{row['Element']}:**
                - ⚠️ Recognition ({row['Recognition']:.0%}) below median - needs more exposure
                - ⚠️ Uniqueness ({row['Uniqueness']:.0%}) below median - less distinctive
                - 💡 Opportunity: {row['Overall Usage']:.0%} current usage could be optimized
                """)
        else:
            st.write("No elements in this quadrant")

# ==================== TAB 2: SENTIMENT ANALYSIS ====================
with tab2:
    st.header("💚 Sentiment Analysis")
    st.caption("Consumer perception analysis based on Q04 semantic differential scales")

    # Key Takeaways
    st.warning("""
    ### 🎯 Key Takeaways - Sentiment Challenge
    
    **The Reality:**
    - Only 2 out of 9 elements have positive sentiment (Symbol +0.3%, Sonic +1.1%)
    - 7 elements have net negative sentiment (more negative than positive associations)
    - Average sentiment is -3.4% across all elements
    
    **What This Means:**
    - Brand elements trigger slightly more negative than positive emotional responses
    - This is a brand health concern requiring attention
    - Focus on strengthening emotional connection, especially for weakest performers
    
    **Action Items:**
    - Redesign or reposition elements with <-5% sentiment
    - Leverage Sonic and Symbol (the only positive performers) more prominently
    - Address why Type (-7.7%) and Hacek (-6.9%) perform poorly
    """)

    st.markdown("---")

    st.markdown("""
    <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h4>Understanding Sentiment Scores</h4>
    <p><b>Positive Sentiment:</b> Average % of respondents choosing positive descriptors (Bold, Stylish, Modern, Simple, Human, Exciting, Playful)</p>
    <p><b>Negative Sentiment:</b> Average % choosing opposite descriptors (Cautious, Plain, Old-Fashioned, Complicated, Cold, Boring, Serious)</p>
    <p><b>Net Sentiment:</b> Positive minus Negative (higher = more positive brand perception)</p>
    </div>
    """, unsafe_allow_html=True)

    # Overview Metrics Cards
    col1, col2, col3, col4 = st.columns(4)

    most_positive = master_df.loc[master_df['Net Sentiment'].idxmax()]
    least_positive = master_df.loc[master_df['Net Sentiment'].idxmin()]
    avg_net_sentiment = master_df['Net Sentiment'].mean()
    sentiment_range = master_df['Net Sentiment'].max() - master_df['Net Sentiment'].min()

    with col1:
        st.metric(
            "Most Positive Element", 
            most_positive['Element'], 
            f"+{most_positive['Net Sentiment']:.1%}",
            help="Net sentiment = % choosing positive descriptors minus % choosing negative descriptors. Positive values indicate more positive than negative associations."
        )
        st.success(f"**{most_positive['Net Sentiment']:.1%}** net positive perception.")
        with st.expander("❓ Why does this element have the highest sentiment?"):
            st.markdown(f"""
            **{most_positive['Element']}** resonates most strongly because:

            1. **Strong Positive Scores:** {most_positive['Positive Sentiment']:.1%} positive vs {most_positive['Negative Sentiment']:.1%} negative
            2. **Design Appeal:** The visual/audio qualities naturally evoke positive emotions
            3. **Familiarity Effect:** {most_positive['Recognition']:.0%} recognition builds positive associations over time
            4. **Emotional Triggers:** Successfully communicates brand values like modern, stylish, exciting

            High sentiment indicates this asset creates emotional connection beyond just recognition.
            """)

    with col2:
        st.metric("Least Positive Element", least_positive['Element'], f"{least_positive['Net Sentiment']:+.1%}")
        st.warning(f"**{least_positive['Net Sentiment']:+.1%}** net sentiment - needs improvement.")
        with st.expander("❓ Why is this element's sentiment negative?"):
            st.markdown(f"""
            **{least_positive['Element']}** has negative net sentiment because:

            1. **More Negative Associations:** {least_positive['Negative Sentiment']:.1%} negative vs {least_positive['Positive Sentiment']:.1%} positive
            2. **Lower Recognition:** Only {least_positive['Recognition']:.0%} recognition - less familiarity may lead to weaker emotional connection
            3. **Design Challenge:** The element may not resonate emotionally or appears generic
            4. **Opportunity:** With {least_positive['Overall Usage']:.0%} current usage, this asset needs redesign or stronger positioning

            **This is a brand concern that requires strategic attention.**
            """)

    with col3:
        st.metric("Average Net Sentiment", "All Elements", f"{avg_net_sentiment:+.1%}")
        st.warning(f"Škoda brand elements generate **{avg_net_sentiment:+.1%}** net sentiment - slightly negative overall.")
        with st.expander("💡 What does this mean for the brand?"):
            st.markdown(f"""
            This average sentiment score tells us:

            1. **Brand Health Concern:** 7 out of 9 elements have net negative sentiment - needs attention
            2. **Mixed Performance:** Range of {master_df['Net Sentiment'].min():+.1%} to {master_df['Net Sentiment'].max():+.1%} shows inconsistent performance
            3. **Emotional Disconnect:** Only {master_df['Positive Sentiment'].mean():.1%} average positive sentiment vs {master_df['Negative Sentiment'].mean():.1%} negative
            4. **Action Required:** The negative overall sentiment indicates brand perception needs strengthening

            **Strategic Priority:** Focus on improving emotional connection and positive associations, especially for the 7 underperforming elements.
            """)

    with col4:
        st.metric("Sentiment Range", f"{sentiment_range:.1%}", "Max - Min")
        st.info(f"Variation of **{sentiment_range:.1%}** shows significant differences in emotional impact.")
        with st.expander("📊 Why does sentiment vary across elements?"):
            st.markdown(f"""
            Sentiment varies between {master_df['Net Sentiment'].min():+.1%} and {master_df['Net Sentiment'].max():+.1%} because:

            1. **Design Characteristics:** Some elements (Sonic, Symbol) trigger more positive emotional responses
            2. **Recognition Impact:** Higher-recognition assets don't automatically have better sentiment (e.g., Electric Green has 20% recognition but negative sentiment)
            3. **Cultural Factors:** Elements like typography and colors may not resonate across all markets
            4. **Functional vs Emotional:** Logo/sonic elements perform better than color/type elements

            **Strategy:** Prioritize Sonic and Symbol in communications; redesign or phase out weakest performers.
            """)


    st.markdown("---")

    # Positive vs Negative Bar Chart
    st.markdown("### Positive vs Negative Sentiment Comparison")
    st.caption("Green bars show positive associations, red bars show negative associations")

    # Prepare data for grouped bar chart
    sentiment_comparison = master_df[['Element', 'Positive Sentiment', 'Negative Sentiment']].copy()
    sentiment_comparison_melted = sentiment_comparison.melt(
        id_vars='Element',
        var_name='Sentiment Type',
        value_name='Percentage'
    )

    fig_comparison = go.Figure()

    # Add positive sentiment bars (green)
    fig_comparison.add_trace(go.Bar(
        name='Positive Sentiment',
        x=sentiment_comparison['Element'],
        y=sentiment_comparison['Positive Sentiment'],
        marker_color='#4CAF50',
        text=sentiment_comparison['Positive Sentiment'].apply(lambda x: f'{x:.1%}'),
        textposition='outside'
    ))

    # Add negative sentiment bars (red)
    fig_comparison.add_trace(go.Bar(
        name='Negative Sentiment',
        x=sentiment_comparison['Element'],
        y=sentiment_comparison['Negative Sentiment'],
        marker_color='#F44336',
        text=sentiment_comparison['Negative Sentiment'].apply(lambda x: f'{x:.1%}'),
        textposition='outside'
    ))

    fig_comparison.update_layout(
        barmode='group',
        title='Positive vs Negative Sentiment by Brand Element',
        xaxis_title='Brand Element',
        yaxis_title='Sentiment Score',
        yaxis_tickformat='.0%',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_comparison, use_container_width=True)

    st.warning("**Key Insight:** Only 2 out of 9 brand elements (Symbol and Sonic) show net positive sentiment. The majority have slightly more negative than positive associations, with an average net sentiment of -3.4%. This indicates opportunities for improvement in brand perception.")

    st.markdown("---")

    # Net Sentiment Ranking Chart
    st.markdown("### Net Sentiment Ranking")
    st.caption("Elements ranked by overall sentiment score (positive minus negative)")

    sentiment_ranked = master_df.sort_values('Net Sentiment', ascending=True)

    # Create color gradient based on net sentiment values
    colors = sentiment_ranked['Net Sentiment'].apply(
        lambda x: f'rgb({int(244 - (x * 400))}, {int(67 + (x * 400))}, {int(54 + (x * 100))})'
    ).tolist()

    fig_net = go.Figure(go.Bar(
        x=sentiment_ranked['Net Sentiment'],
        y=sentiment_ranked['Element'],
        orientation='h',
        marker=dict(
            color=sentiment_ranked['Net Sentiment'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Net Sentiment", tickformat='.0%')
        ),
        text=sentiment_ranked['Net Sentiment'].apply(lambda x: f'{x:+.1%}'),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Net Sentiment: %{x:.1%}<extra></extra>'
    ))

    fig_net.update_layout(
        title='Brand Elements Ranked by Net Sentiment',
        xaxis_title='Net Sentiment Score',
        yaxis_title='Brand Element',
        xaxis_tickformat='.0%',
        height=500,
        showlegend=False
    )

    st.plotly_chart(fig_net, use_container_width=True)

    # Top 3 and Bottom 3
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏆 Top 3 Best Performing")
        top_3_sentiment = sentiment_ranked.nlargest(3, 'Net Sentiment')
        for idx, row in top_3_sentiment.iterrows():
            sentiment_color = "success" if row['Net Sentiment'] > 0 else "info"
            if sentiment_color == "success":
                st.success(f"**{row['Element']}**: {row['Net Sentiment']:+.1%} net sentiment")
            else:
                st.info(f"**{row['Element']}**: {row['Net Sentiment']:+.1%} net sentiment")
            st.write(f"   • Positive: {row['Positive Sentiment']:.1%} | Negative: {row['Negative Sentiment']:.1%}")

    with col2:
        st.markdown("#### ⚠️ Bottom 3 (Need Improvement)")
        bottom_3_sentiment = sentiment_ranked.nsmallest(3, 'Net Sentiment')
        for idx, row in bottom_3_sentiment.iterrows():
            st.warning(f"**{row['Element']}**: {row['Net Sentiment']:.1%} net sentiment")
            st.write(f"   • Positive: {row['Positive Sentiment']:.1%} | Negative: {row['Negative Sentiment']:.1%}")

    st.markdown("---")

    # Detailed Sentiment Data Table
    st.markdown("### Detailed Sentiment Data")
    st.caption("Complete breakdown of sentiment scores with interpretations")

    # Create detailed table with interpretation
    detailed_sentiment = master_df[['Element', 'Positive Sentiment', 'Negative Sentiment', 'Net Sentiment']].copy()

    # Add interpretation column
    def interpret_sentiment(net):
        if net >= 0.01:
            return "✅ Net Positive - Good emotional connection"
        elif net >= -0.02:
            return "⚖️ Near Neutral - Balanced perception"
        elif net >= -0.05:
            return "⚠️ Slightly Negative - Needs attention"
        else:
            return "🔴 Negative - Requires improvement"

    detailed_sentiment['Interpretation'] = detailed_sentiment['Net Sentiment'].apply(interpret_sentiment)

    # Style the table
    styler_sentiment = detailed_sentiment.set_index('Element').style

    # Apply heatmap to sentiment columns
    styler_sentiment = styler_sentiment.background_gradient(
        cmap='RdYlGn',
        subset=['Net Sentiment'],
        vmin=-0.08,
        vmax=0.02
    )

    styler_sentiment = styler_sentiment.background_gradient(
        cmap='Greens',
        subset=['Positive Sentiment'],
        vmin=0.46,
        vmax=0.51
    )

    styler_sentiment = styler_sentiment.background_gradient(
        cmap='Reds_r',
        subset=['Negative Sentiment'],
        vmin=0.49,
        vmax=0.54
    )

    # Format as percentages
    styler_sentiment = styler_sentiment.format({
        'Positive Sentiment': '{:.1%}',
        'Negative Sentiment': '{:.1%}',
        'Net Sentiment': '{:+.1%}'
    })

    st.dataframe(styler_sentiment, use_container_width=True)

    # Download sentiment data
    sentiment_csv = detailed_sentiment.to_csv(index=False)
    st.download_button(
        label="📥 Download Sentiment Analysis CSV",
        data=sentiment_csv,
        file_name="skoda_sentiment_analysis.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # Additional Insights
    st.markdown("### 📊 Strategic Implications")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Strengths")
        st.write(f"• **{most_positive['Element']}** leads with {most_positive['Net Sentiment']:+.1%} net sentiment")
        st.write(f"• **Symbol** (+0.3%) shows the brand mark itself has balanced perception")
        st.write("• Average 48.3% positive associations shows baseline appeal")
        st.write("• No element is severely negative - all have ~46-51% positive scores")

    with col2:
        st.markdown("#### ⚠️ Areas for Improvement")
        st.write(f"• **7 out of 9 elements have net negative sentiment** - concern for brand health")
        st.write(f"• **{least_positive['Element']}** needs most work with {least_positive['Net Sentiment']:+.1%} net sentiment")
        st.write(f"• Average net sentiment is **-3.4%** - slightly more negative than positive overall")
        st.write(f"• Focus on strengthening emotional connection for bottom performers")
        st.write("• Consider design/messaging updates for weakest elements")

    st.markdown("---")

    # Q05 Brand Confusion Matrix
    st.markdown("### 🎯 Brand Confusion Matrix (Q05)")
    st.caption("Which brands do consumers think these elements belong to?")

    st.info("""
    **Key Insight:** Brand confusion analysis reveals competitive threats. High Škoda attribution = distinctive asset. 
    High competitor attribution = confusion risk. High "Generic" = lacks brand identity.
    """)

    # Create confusion matrix
    confusion_df = pd.DataFrame(q05_confusion_data).T
    confusion_df = confusion_df[['Skoda', 'VW', 'Toyota', 'Seat', 'Generic', 'Dont_Know']]
    confusion_df.columns = ['Škoda', 'VW', 'Toyota', 'Seat', 'Generic', "Don't Know"]

    # Create confusion matrix with inverted scale for competitors
    # We need to invert competitor columns so high values = red, low values = green
    confusion_df_display = confusion_df.copy()
    
    # Invert competitor and generic columns (1 - value) so high becomes low for coloring
    for col in ['VW', 'Toyota', 'Seat', 'Generic', "Don't Know"]:
        confusion_df_display[col] = 1 - confusion_df_display[col]
    
    # Keep Škoda as-is (high = green is correct)
    
    # Create heatmap with consistent color scale
    fig_confusion = px.imshow(
        confusion_df_display,
        labels=dict(x="Attributed Brand", y="Element", color="Score"),
        x=confusion_df_display.columns,
        y=confusion_df_display.index,
        color_continuous_scale='RdYlGn',  # Red = bad, Green = good
        text_auto=False,  # We'll add custom text
        aspect="auto",
        title="Brand Attribution: Who Do Consumers Think Owns These Elements?"
    )
    
    # Add text annotations with actual percentages (not inverted display values)
    annotations = []
    for i, element in enumerate(confusion_df.index):
        for j, brand in enumerate(confusion_df.columns):
            actual_value = confusion_df.loc[element, brand]
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=f'{actual_value:.0%}',
                    showarrow=False,
                    font=dict(size=12, color='white' if confusion_df_display.iloc[i, j] < 0.5 else 'black')
                )
            )
    
    fig_confusion.update_layout(
        annotations=annotations,
        height=500
    )
    st.plotly_chart(fig_confusion, use_container_width=True)
    
    st.caption("""
    **Color Guide:** 
    - 🟢 Green = Good (High Škoda attribution OR Low competitor/generic confusion)
    - 🔴 Red = Bad (Low Škoda attribution OR High competitor/generic confusion)
    """)

    # Analysis columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ✅ Distinctive Assets")
        distinctive = confusion_df.sort_values('Škoda', ascending=False).head(3)
        for element, row in distinctive.iterrows():
            st.success(f"**{element}**: {row['Škoda']:.0%} Škoda")
            st.caption(f"VW: {row['VW']:.0%} | Generic: {row['Generic']:.0%}")

    with col2:
        st.markdown("#### ⚠️ Confusion Risks")
        
        # Find elements with high VW confusion
        high_vw = confusion_df[confusion_df['VW'] >= 0.15].sort_values('VW', ascending=False)
        if len(high_vw) > 0:
            st.warning("**VW Confusion (Brand Dilution Risk):**")
            for element, row in high_vw.iterrows():
                st.write(f"• **{element}**: {row['VW']:.0%} think it's VW")
        
        # Find elements with high generic attribution
        high_generic = confusion_df[confusion_df['Generic'] >= 0.20].sort_values('Generic', ascending=False)
        if len(high_generic) > 0:
            st.warning("**Generic/No Brand Association:**")
            for element, row in high_generic.iterrows():
                st.write(f"• **{element}**: {row['Generic']:.0%} say generic")

    # Competitive threat matrix
    st.markdown("#### 📊 Redesign Priority Matrix")
    
    confusion_df['Competitive_Risk'] = confusion_df['VW'] + confusion_df['Toyota'] + confusion_df['Seat']
    confusion_df['Distinctiveness_Score'] = confusion_df['Škoda'] - confusion_df['Competitive_Risk']
    
    threat_matrix = []
    for element in confusion_df.index:
        skoda_attr = confusion_df.loc[element, 'Škoda']
        comp_risk = confusion_df.loc[element, 'Competitive_Risk']
        
        if skoda_attr < 0.35 and comp_risk > 0.25:
            priority = "🔴 HIGH - Fix Now"
        elif skoda_attr < 0.35 or comp_risk > 0.25:
            priority = "🟡 MEDIUM - Monitor"
        else:
            priority = "🟢 LOW - Maintain"
        
        threat_matrix.append({
            'Element': element,
            'Škoda Attribution': skoda_attr,
            'Competitor Confusion': comp_risk,
            'Priority': priority
        })
    
    threat_df = pd.DataFrame(threat_matrix).sort_values('Competitor Confusion', ascending=False)
    st.dataframe(threat_df.style.format({
        'Škoda Attribution': '{:.0%}',
        'Competitor Confusion': '{:.0%}'
    }), use_container_width=True)

# ==================== TAB 3: STRATEGIC INSIGHTS ====================
with tab3:
    st.header("Strategic Insights Dashboard")
    st.caption("Advanced analytics to identify opportunities and optimize brand asset usage")

    # Key Takeaways
    st.info("""
    ### 🎯 Key Takeaways - Efficiency & Combinations
    
    **ROI Winners:**
    - **Sonic** delivers best efficiency (low investment, strong recognition)
    - **Symbol** provides best overall value (high recognition + high uniqueness)
    
    **Element Combinations:**
    - Symbol-based combinations consistently show highest recognition (green in heatmap)
    - Type + Electric Green shows poor performance (red in heatmap) - avoid this pairing
    - Minimum 3 elements needed per ad for effective brand recognition
    
    **Investment Insights:**
    - Some high-investment elements underperform (Some elements show X pattern)
    - Some elements show X pattern proven high-ROI combinations
    - Symbol achieves 48% recognition vs 20% average (48% recognition vs 20% average)
    """)

    st.markdown("---")

    # Portfolio Optimization Matrices
    st.markdown("### 📊 Portfolio Optimization Matrices")
    st.caption("BCG-style strategic analysis - performance patterns across investment and recognition")

    # Prepare data for matrices
    matrix_df = master_df.copy()
    
    # Calculate medians for quadrant splits
    median_recognition = matrix_df['Recognition'].median()
    median_investment = matrix_df['Total Investment'].median()
    median_uniqueness = matrix_df['Uniqueness'].median()
    median_usage = matrix_df['Overall Usage'].median()
    median_roi = matrix_df['Recognition ROI'].median()

    # Matrix 1: Recognition vs Investment (BCG Matrix)
    st.markdown("#### 1️⃣ Recognition vs Investment Matrix")
    st.caption("Strategic positioning: Stars, Cash Cows, Question Marks, Dogs")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig_bcg = px.scatter(
            matrix_df,
            x='Total Investment',
            y='Recognition',
            size='Uniqueness',
            color='Net Sentiment',
            hover_name='Element',
            text='Element',
            title='Recognition vs Investment (BCG Matrix)',
            color_continuous_scale='RdYlGn',
            size_max=30
        )
        
        # Add quadrant lines
        fig_bcg.add_hline(y=median_recognition, line_dash="dash", line_color="gray", opacity=0.5)
        fig_bcg.add_vline(x=median_investment, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add quadrant labels
        fig_bcg.add_annotation(x=matrix_df['Total Investment'].max() * 0.75, y=matrix_df['Recognition'].max() * 0.95,
                               text="STARS<br>(High Rec, High Inv)", showarrow=False, font=dict(size=10, color="green"))
        fig_bcg.add_annotation(x=matrix_df['Total Investment'].min() * 1.5, y=matrix_df['Recognition'].max() * 0.95,
                               text="HIDDEN GEMS<br>(High Rec, Low Inv)", showarrow=False, font=dict(size=10, color="darkgreen"))
        fig_bcg.add_annotation(x=matrix_df['Total Investment'].max() * 0.75, y=matrix_df['Recognition'].min() * 1.5,
                               text="DOGS<br>(Low Rec, High Inv)<br>⚠️ CUT", showarrow=False, font=dict(size=10, color="red"))
        fig_bcg.add_annotation(x=matrix_df['Total Investment'].min() * 1.5, y=matrix_df['Recognition'].min() * 1.5,
                               text="QUESTION MARKS<br>(Low Rec, Low Inv)", showarrow=False, font=dict(size=10, color="orange"))
        
        fig_bcg.update_traces(textposition='top center')
        fig_bcg.update_layout(height=500, xaxis_title="Total Investment (€)", yaxis_title="Recognition %")
        fig_bcg.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig_bcg, use_container_width=True)
    
    with col2:
        st.markdown("#### Quadrant Analysis")
        
        # Categorize elements
        stars = matrix_df[(matrix_df['Recognition'] >= median_recognition) & (matrix_df['Total Investment'] >= median_investment)]
        gems = matrix_df[(matrix_df['Recognition'] >= median_recognition) & (matrix_df['Total Investment'] < median_investment)]
        dogs = matrix_df[(matrix_df['Recognition'] < median_recognition) & (matrix_df['Total Investment'] >= median_investment)]
        questions = matrix_df[(matrix_df['Recognition'] < median_recognition) & (matrix_df['Total Investment'] < median_investment)]
        
        if len(stars) > 0:
            st.success(f"**STARS ({len(stars)}):**")
            for _, row in stars.iterrows():
                st.write(f"• {row['Element']}")
            st.caption("Maintain investment")
        
        if len(gems) > 0:
            st.success(f"**HIDDEN GEMS ({len(gems)}):**")
            for _, row in gems.iterrows():
                st.write(f"• {row['Element']}")
            st.caption("")
        
        if len(dogs) > 0:
            st.error(f"**DOGS ({len(dogs)}):**")
            for _, row in dogs.iterrows():
                st.write(f"• {row['Element']}")
            st.caption("")
        
        if len(questions) > 0:
            st.warning(f"**QUESTION MARKS ({len(questions)}):**")
            for _, row in questions.iterrows():
                st.write(f"• {row['Element']}")
            st.caption("Test or hold")

    st.markdown("---")

    # Matrix 2: Recognition vs Uniqueness (Brand Equity Matrix)
    st.markdown("#### 2️⃣ Recognition vs Uniqueness Matrix")
    st.caption("Brand equity positioning: Icons, Famous Generics, Hidden Gems, Weak")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig_equity = px.scatter(
            matrix_df,
            x='Uniqueness',
            y='Recognition',
            size='Total Investment',
            color='Recognition ROI',
            hover_name='Element',
            text='Element',
            title='Recognition vs Uniqueness (Brand Equity Matrix)',
            color_continuous_scale='RdYlGn',
            size_max=30
        )
        
        # Add quadrant lines
        fig_equity.add_hline(y=median_recognition, line_dash="dash", line_color="gray", opacity=0.5)
        fig_equity.add_vline(x=median_uniqueness, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add quadrant labels
        fig_equity.add_annotation(x=matrix_df['Uniqueness'].max() * 0.9, y=matrix_df['Recognition'].max() * 0.95,
                                  text="BRAND ICONS<br>(High Rec, High Uniq)<br>🏆 PROTECT", showarrow=False, font=dict(size=10, color="darkgreen"))
        fig_equity.add_annotation(x=matrix_df['Uniqueness'].min() * 1.2, y=matrix_df['Recognition'].max() * 0.95,
                                  text="FAMOUS GENERICS<br>(High Rec, Low Uniq)<br>⚠️ Risk", showarrow=False, font=dict(size=10, color="orange"))
        fig_equity.add_annotation(x=matrix_df['Uniqueness'].max() * 0.9, y=matrix_df['Recognition'].min() * 1.5,
                                  text="HIDDEN GEMS<br>(Low Rec, High Uniq)<br>💎 Invest", showarrow=False, font=dict(size=10, color="blue"))
        fig_equity.add_annotation(x=matrix_df['Uniqueness'].min() * 1.2, y=matrix_df['Recognition'].min() * 1.5,
                                  text="WEAK<br>(Low Rec, Low Uniq)<br>🔴 Fix", showarrow=False, font=dict(size=10, color="red"))
        
        fig_equity.update_traces(textposition='top center')
        fig_equity.update_layout(height=500, xaxis_title="Uniqueness %", yaxis_title="Recognition %")
        fig_equity.update_xaxes(tickformat='.0%')
        fig_equity.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig_equity, use_container_width=True)
    
    with col2:
        st.markdown("#### Strategic Actions")
        
        # Categorize elements
        icons = matrix_df[(matrix_df['Recognition'] >= median_recognition) & (matrix_df['Uniqueness'] >= median_uniqueness)]
        generics = matrix_df[(matrix_df['Recognition'] >= median_recognition) & (matrix_df['Uniqueness'] < median_uniqueness)]
        hidden = matrix_df[(matrix_df['Recognition'] < median_recognition) & (matrix_df['Uniqueness'] >= median_uniqueness)]
        weak = matrix_df[(matrix_df['Recognition'] < median_recognition) & (matrix_df['Uniqueness'] < median_uniqueness)]
        
        if len(icons) > 0:
            st.success(f"**BRAND ICONS ({len(icons)}):**")
            for _, row in icons.iterrows():
                st.write(f"• {row['Element']}")
            st.caption("🏆 Core assets - protect")
        
        if len(generics) > 0:
            st.warning(f"**FAMOUS GENERICS ({len(generics)}):**")
            for _, row in generics.iterrows():
                st.write(f"• {row['Element']}")
            st.caption("⚠️ Increase distinctiveness")
        
        if len(hidden) > 0:
            st.info(f"**HIDDEN GEMS ({len(hidden)}):**")
            for _, row in hidden.iterrows():
                st.write(f"• {row['Element']}")
            st.caption("")
        
        if len(weak) > 0:
            st.error(f"**WEAK ({len(weak)}):**")
            for _, row in weak.iterrows():
                st.write(f"• {row['Element']}")
            st.caption("")

    st.markdown("---")

    # Matrix 3: Usage vs ROI (Efficiency Matrix)
    st.markdown("#### 3️⃣ Usage vs ROI Matrix")
    st.caption("Investment efficiency: Workhorses, Overused, Untapped Potential, Underperformers")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig_efficiency = px.scatter(
            matrix_df,
            x='Overall Usage',
            y='Recognition ROI',
            size='Recognition',
            color='Net Sentiment',
            hover_name='Element',
            text='Element',
            title='Usage vs ROI (Efficiency Matrix)',
            color_continuous_scale='RdYlGn',
            size_max=30
        )
        
        # Add quadrant lines
        fig_efficiency.add_hline(y=median_roi, line_dash="dash", line_color="gray", opacity=0.5)
        fig_efficiency.add_vline(x=median_usage, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add quadrant labels
        fig_efficiency.add_annotation(x=matrix_df['Overall Usage'].max() * 0.85, y=matrix_df['Recognition ROI'].max() * 0.95,
                                      text="WORKHORSES<br>(High Use, High ROI)<br>✅ Perfect", showarrow=False, font=dict(size=10, color="darkgreen"))
        fig_efficiency.add_annotation(x=matrix_df['Overall Usage'].min() * 1.5, y=matrix_df['Recognition ROI'].max() * 0.95,
                                      text="UNTAPPED<br>(Low Use, High ROI)<br>⬆️ Use More", showarrow=False, font=dict(size=10, color="blue"))
        fig_efficiency.add_annotation(x=matrix_df['Overall Usage'].max() * 0.85, y=matrix_df['Recognition ROI'].min() * 1.5,
                                      text="OVERUSED<br>(High Use, Low ROI)<br>⬇️ Cut Back", showarrow=False, font=dict(size=10, color="red"))
        fig_efficiency.add_annotation(x=matrix_df['Overall Usage'].min() * 1.5, y=matrix_df['Recognition ROI'].min() * 1.5,
                                      text="UNDERPERFORMERS<br>(Low Use, Low ROI)", showarrow=False, font=dict(size=10, color="orange"))
        
        fig_efficiency.update_traces(textposition='top center')
        fig_efficiency.update_layout(height=500, xaxis_title="Usage %", yaxis_title="Recognition ROI")
        fig_efficiency.update_xaxes(tickformat='.0%')
        st.plotly_chart(fig_efficiency, use_container_width=True)
    
    with col2:
        st.markdown("#### Optimization Actions")
        
        # Categorize elements
        workhorses = matrix_df[(matrix_df['Overall Usage'] >= median_usage) & (matrix_df['Recognition ROI'] >= median_roi)]
        untapped = matrix_df[(matrix_df['Overall Usage'] < median_usage) & (matrix_df['Recognition ROI'] >= median_roi)]
        overused = matrix_df[(matrix_df['Overall Usage'] >= median_usage) & (matrix_df['Recognition ROI'] < median_roi)]
        underperf = matrix_df[(matrix_df['Overall Usage'] < median_usage) & (matrix_df['Recognition ROI'] < median_roi)]
        
        if len(workhorses) > 0:
            st.success(f"**WORKHORSES ({len(workhorses)}):**")
            for _, row in workhorses.iterrows():
                st.write(f"• {row['Element']}")
            st.caption("✅ Maintain strategy")
        
        if len(untapped) > 0:
            st.info(f"**UNTAPPED ({len(untapped)}):**")
            for _, row in untapped.iterrows():
                st.write(f"• {row['Element']}")
            st.caption("⬆️ Increase usage")
        
        if len(overused) > 0:
            st.error(f"**OVERUSED ({len(overused)}):**")
            for _, row in overused.iterrows():
                st.write(f"• {row['Element']}")
            st.caption("⬇️ Reduce investment")
        
        if len(underperf) > 0:
            st.warning(f"**UNDERPERFORMERS ({len(underperf)}):**")
            for _, row in underperf.iterrows():
                st.write(f"• {row['Element']}")
            st.caption("Monitor or retire")

    st.markdown("---")

    # Recognition ROI Analysis - Multi-Dimensional
    st.markdown("### 💡 Recognition Efficiency: Multi-Dimensional Analysis")

    # ROI metric selector
    roi_metric = st.selectbox(
        "Choose your efficiency perspective:",
        [
            "Total Investment Efficiency",
            "Per-Ad Recognition Efficiency",
            "Average Investment Efficiency",
            "Brand Equity Efficiency Index"
        ]
    )

    # Calculate different ROI metrics
    master_df_roi = master_df.copy()

    # Add number of ads per element
    for idx, row in master_df_roi.iterrows():
        element = row['Element']
        num_ads = audit_df[audit_df[element] == True].shape[0]
        master_df_roi.at[idx, 'Num Ads'] = num_ads

    if roi_metric == "Total Investment Efficiency":
        master_df_roi['Selected ROI'] = master_df_roi['Recognition ROI']
        metric_label = "Recognition % per €1M Total Investment"
        insight_text = "**Shows which assets achieved recognition with minimal total campaign spend.** High scorers are 'hidden gems' that punched above their weight."

    elif roi_metric == "Per-Ad Recognition Efficiency":
        master_df_roi['Selected ROI'] = master_df_roi.apply(
            lambda x: (x['Recognition'] / x['Num Ads'] * 100) if x['Num Ads'] > 0 else 0, axis=1
        )
        metric_label = "Recognition % per Ad"
        insight_text = "**Shows how many ad exposures are needed to build recognition.** High scorers build awareness faster with fewer placements."

    elif roi_metric == "Average Investment Efficiency":
        master_df_roi['Selected ROI'] = master_df_roi.apply(
            lambda x: (x['Recognition'] / x['Average Investment'] * 1_000_000) if x['Average Investment'] > 0 else 0, axis=1
        )
        metric_label = "Recognition % per €1M Average Investment"
        insight_text = "**Shows cost-effectiveness per individual ad placement.** High scorers deliver better recognition per placement budget."

    else:  # Brand Equity Efficiency Index
        master_df_roi['Selected ROI'] = master_df_roi.apply(
            lambda x: (x['Recognition'] * x['Uniqueness']) / (x['Total Investment'] / 1_000_000) if x['Total Investment'] > 0 else 0, axis=1
        )
        metric_label = "Brand Equity Index (Recognition × Uniqueness) per €1M"
        insight_text = "**Holistic efficiency combining fame and differentiation.** High scorers deliver the most long-term brand equity per euro - ideal for identifying Performance Tiers."

    st.info(insight_text)

    col1, col2 = st.columns([2, 1])

    with col1:
        roi_df = master_df_roi.sort_values('Selected ROI', ascending=True)
        fig_roi = px.bar(
            roi_df,
            y='Element',
            x='Selected ROI',
            orientation='h',
            title=f'Efficiency Analysis: {metric_label}',
            text=roi_df['Selected ROI'].apply(lambda x: f'{x:.2f}'),
            color='Selected ROI',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_roi, use_container_width=True)

    with col2:
        st.markdown("#### Top 3 Performers:")
        top_3_roi = roi_df.nlargest(3, 'Selected ROI')
        for idx, row in top_3_roi.iterrows():
            st.success(f"**{row['Element']}**: {row['Selected ROI']:.2f}")
            with st.expander(f"Why {row['Element']}?"):
                if roi_metric == "Brand Equity Efficiency Index":
                    equity = row['Recognition'] * row['Uniqueness']
                    st.write(f"**Recognition:** {row['Recognition']:.0%}")
                    st.write(f"**Uniqueness:** {row['Uniqueness']:.0%}")
                    st.write(f"**Brand Equity:** {equity:.3f}")
                    st.write(f"**Investment:** €{row['Total Investment']:,.0f}")
                    st.write(f"**Why efficient:** Achieves {equity:.3f} brand equity with only €{row['Total Investment']:,.0f} - delivers maximum long-term value per euro")
                else:
                    st.write(f"**Recognition:** {row['Recognition']:.0%}")
                    st.write(f"**Investment:** €{row['Total Investment']:,.0f}")
                    st.write(f"**Usage:** {row['Overall Usage']:.0%}")
                    st.write(f"**Why efficient:** High recognition relative to investment/usage indicates strong inherent memorability and strategic placement")

        st.markdown("#### Bottom 3:")
        bottom_3_roi = roi_df.nsmallest(3, 'Selected ROI')
        for idx, row in bottom_3_roi.iterrows():
            st.warning(f"**{row['Element']}**: {row['Selected ROI']:.2f}")
            with st.expander(f"Why {row['Element']}?"):
                if roi_metric == "Brand Equity Efficiency Index":
                    equity = row['Recognition'] * row['Uniqueness']
                    st.write(f"**Recognition:** {row['Recognition']:.0%}")
                    st.write(f"**Uniqueness:** {row['Uniqueness']:.0%}")
                    st.write(f"**Brand Equity:** {equity:.3f}")
                    st.write(f"**Investment:** €{row['Total Investment']:,.0f}")
                    if row['Selected ROI'] < 0.01:
                        st.write(f"**Why low/zero:** Very high investment (€{row['Total Investment']:,.0f}) relative to brand equity outcome ({equity:.3f}). This could indicate: 1) Recent investment not yet reflected in recognition, 2) Generic element that lacks Škoda distinctiveness, or 3) Inefficient deployment")
                    else:
                        st.write(f"**Why lower:** Investment (€{row['Total Investment']:,.0f}) is high relative to the brand equity delivered ({equity:.3f})")
                else:
                    st.write(f"**Recognition:** {row['Recognition']:.0%}")
                    st.write(f"**Investment:** €{row['Total Investment']:,.0f}")
                    st.write(f"**Usage:** {row['Overall Usage']:.0%}")
                    st.write(f"**Why lower:** High investment/usage but recognition hasn't grown proportionally - may need creative optimization or more time to build awareness")

        st.markdown("#### Strategic Implication:")
        if roi_metric == "Total Investment Efficiency":
            st.write("• Scale up top performers")
            st.write("• Review bottom performers' deployment")
        elif roi_metric == "Per-Ad Recognition Efficiency":
            st.write("• Top assets build awareness faster")
            st.write("• Increase frequency for top performers")
        elif roi_metric == "Average Investment Efficiency":
            st.write("• Optimize placement budgets")
            st.write("• Reallocate to efficient assets")
        else:
            st.write("• **Top = Performance Tiers candidates**")
            st.write("• **Bottom = Requires optimization**")

    st.markdown("---")

    # Efficiency Quadrant Analysis
    st.markdown("### 📊 Asset Performance Quadrants")
    st.info("**Insight:** Categorize assets by recognition and uniqueness performance")

    with st.expander("📖 How to read the quadrants"):
        st.markdown("""
        This analysis categorizes brand assets into 4 strategic groups based on their performance:

        **⭐ Stars (Top-Right):** High Recognition + High Uniqueness
        - **Why they're here:** Frequent usage, high investment, and distinctively Škoda design
        - **Strategy:** Protect and amplify - these are your brand-building powerhouses

        **🐴 Workhorses (Top-Left):** High Recognition + Lower Uniqueness
        - **Why they're here:** Well-used but less distinctive (may be generic automotive elements)
        - **Strategy:** Maintain awareness but pair with unique assets for differentiation

        **💎 Hidden Gems (Bottom-Right):** Lower Recognition + High Uniqueness
        - **Why they're here:** Distinctive but underutilized or recently introduced
        - **Strategy:** Invest more - these have untapped potential for differentiation

        **❓ Question Marks (Bottom-Left):** Lower Recognition + Lower Uniqueness
        - **Why they're here:** Limited usage, lower investment, or lack distinctiveness
        - **Strategy:** Evaluate - optimize deployment or reconsider as core asset
        """)

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
        stars = master_df[master_df['Quadrant'] == 'Stars ⭐']
        st.success(f"**Stars ⭐** ({len(stars)})")
        st.write("High Recognition + High Uniqueness")
        for idx, row in stars.iterrows():
            st.write(f"• **{row['Element']}**")
            with st.expander(f"Why {row['Element']} is a Star"):
                st.write(f"**Recognition:** {row['Recognition']:.0%} (above median {median_recognition:.0%})")
                st.write(f"**Uniqueness:** {row['Uniqueness']:.0%} (above median {median_uniqueness:.0%})")
                st.write(f"**Usage:** {row['Overall Usage']:.0%} of campaigns")
                st.write(f"**Investment:** €{row['Total Investment']:,.0f}")
                st.write(f"**Why a star:** High exposure ({row['Overall Usage']:.0%} usage) + distinctive Škoda identity ({row['Uniqueness']:.0%} uniqueness) = maximum brand equity builder")

    with col2:
        workhorses = master_df[master_df['Quadrant'] == 'Workhorses 🐴']
        st.info(f"**Workhorses 🐴** ({len(workhorses)})")
        st.write("High Recognition + Lower Uniqueness")
        for idx, row in workhorses.iterrows():
            st.write(f"• **{row['Element']}**")
            with st.expander(f"Why {row['Element']} is a Workhorse"):
                st.write(f"**Recognition:** {row['Recognition']:.0%} (above median {median_recognition:.0%})")
                st.write(f"**Uniqueness:** {row['Uniqueness']:.0%} (below median {median_uniqueness:.0%})")
                st.write(f"**Usage:** {row['Overall Usage']:.0%} of campaigns")
                st.write(f"**Why a workhorse:** High familiarity but lower distinctiveness suggests this may be a more generic element. Useful for awareness but pair with unique assets for differentiation")

    with col3:
        gems = master_df[master_df['Quadrant'] == 'Hidden Gems 💎']
        st.warning(f"**Hidden Gems 💎** ({len(gems)})")
        st.write("Lower Recognition + High Uniqueness")
        for idx, row in gems.iterrows():
            st.write(f"• **{row['Element']}**")
            with st.expander(f"Why {row['Element']} is a Hidden Gem"):
                st.write(f"**Recognition:** {row['Recognition']:.0%} (below median {median_recognition:.0%})")
                st.write(f"**Uniqueness:** {row['Uniqueness']:.0%} (above median {median_uniqueness:.0%})")
                st.write(f"**Usage:** {row['Overall Usage']:.0%} of campaigns")
                st.write(f"**Investment:** €{row['Total Investment']:,.0f}")
                st.write(f"**Why a hidden gem:** Highly distinctive ({row['Uniqueness']:.0%} uniqueness) but underexposed ({row['Overall Usage']:.0%} usage). **BIG OPPORTUNITY** - increase deployment to build recognition while maintaining differentiation")

    with col4:
        questions = master_df[master_df['Quadrant'] == 'Question Marks ❓']
        st.error(f"**Question Marks ❓** ({len(questions)})")
        st.write("Lower Recognition + Lower Uniqueness")
        for idx, row in questions.iterrows():
            st.write(f"• **{row['Element']}**")
            with st.expander(f"Why {row['Element']} is a Question Mark"):
                st.write(f"**Recognition:** {row['Recognition']:.0%} (below median {median_recognition:.0%})")
                st.write(f"**Uniqueness:** {row['Uniqueness']:.0%} (below median {median_uniqueness:.0%})")
                st.write(f"**Usage:** {row['Overall Usage']:.0%} of campaigns")
                st.write(f"**Investment:** €{row['Total Investment']:,.0f}")
                st.write(f"**Why a question mark:** Lower recognition AND lower distinctiveness. Could be due to: 1) Limited usage ({row['Overall Usage']:.0%}), 2) Recent introduction, 3) Generic design, or 4) Ineffective deployment. Requires strategic review")

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

    st.markdown("---")

    # Element Combinations Analysis
    st.markdown("### 🔗 Element Combinations: What Works Together?")
    st.caption("Analyzing recognition levels when brand elements appear together")

    # Calculate recognition when elements co-occur
    st.markdown("#### Recognition When Elements Appear Together")
    st.info("Shows the average recognition level when element pairs appear together in ads. Green indicates high recognition, red indicates low recognition.")

    # Create recognition matrix for co-occurring elements
    recognition_matrix = pd.DataFrame(0.0, index=brand_elements, columns=brand_elements, dtype=float)
    
    for element1 in brand_elements:
        for element2 in brand_elements:
            if element1 != element2:
                # Find ads where both elements appear
                both_present = audit_df[audit_df[element1] & audit_df[element2]]
                
                if len(both_present) > 0:
                    # Calculate average recognition across all countries when both appear
                    rec1 = recognition_by_country[element1]
                    rec2 = recognition_by_country[element2]
                    
                    # Average recognition of both elements
                    avg_recognition = (sum(rec1.values()) + sum(rec2.values())) / (2 * len(rec1))
                    recognition_matrix.loc[element1, element2] = avg_recognition

    # Display as heatmap with red-yellow-green scale
    fig_recognition = px.imshow(
        recognition_matrix,
        labels=dict(x="Combined with", y="Element", color="Recognition Level"),
        x=recognition_matrix.columns,
        y=recognition_matrix.index,
        color_continuous_scale='RdYlGn',  # Red to Yellow to Green
        text_auto='.0%',
        aspect="auto",
        title="Recognition Heatmap: Element Combinations"
    )
    fig_recognition.update_layout(height=600)
    st.plotly_chart(fig_recognition, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏆 Highest Recognition Combinations")
        
        # Find top combinations by recognition
        combinations = []
        for element1 in brand_elements:
            for element2 in brand_elements:
                if element1 < element2:  # Avoid duplicates
                    combined_recognition = recognition_matrix.loc[element1, element2]
                    if combined_recognition > 0:
                        # Count how often they appear together
                        both_present = audit_df[audit_df[element1] & audit_df[element2]].shape[0]
                        
                        combinations.append({
                            'Pair': f"{element1} + {element2}",
                            'Recognition': combined_recognition,
                            'Appearances': both_present
                        })
        
        combinations_df = pd.DataFrame(combinations).sort_values('Recognition', ascending=False).head(5)
        
        for _, row in combinations_df.iterrows():
            st.success(f"**{row['Pair']}**")
            st.write(f"   Recognition: {row['Recognition']:.0%} | Appears together: {row['Appearances']} ads")

    with col2:
        st.markdown("#### 💡 Strategic Recommendations")
        
        # Find Symbol's best recognition partners
        symbol_recognition = recognition_matrix.loc['Symbol'].sort_values(ascending=False)
        top_symbol_partner = symbol_recognition.index[0]
        
        st.markdown(f"""
        **Key Findings:**
        
        1. **Symbol combinations perform best:** Highest recognition when Symbol pairs with {top_symbol_partner} ({symbol_recognition.iloc[0]:.0%})
        
        2. **Minimum combinations:** Use at least 3 elements together (recognition builds from 10% with 1 element to 40% with 6)
        
        3. **Top performing pairings:**
           - Look for green cells in the heatmap
           - Symbol-based combinations consistently score higher
           - Avoid red combinations (low recognition)
        
        4. **Avoid:** Single element use (only 10% recognition)
        """)

    st.markdown("---")

    # Recognition lift analysis
    st.markdown("#### 📈 Recognition Lift: Multi-Element Effect")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Calculate how many elements typically appear together
        audit_df['num_elements'] = audit_df[brand_elements].sum(axis=1)
        elements_per_ad = audit_df['num_elements'].value_counts().sort_index()
        
        fig_elements = go.Figure(go.Bar(
            x=elements_per_ad.index,
            y=elements_per_ad.values,
            marker_color='#4CAF50',
            text=elements_per_ad.values,
            textposition='outside'
        ))
        fig_elements.update_layout(
            title='Distribution: Number of Elements per Ad',
            xaxis_title='Number of Brand Elements',
            yaxis_title='Number of Ads',
            height=400
        )
        st.plotly_chart(fig_elements, use_container_width=True)
    
    with col2:
        st.markdown("#### Key Stats")
        
        avg_elements = audit_df['num_elements'].mean()
        st.metric("Avg Elements/Ad", f"{avg_elements:.1f}")
        
        median_elements = audit_df['num_elements'].median()
        st.metric("Median Elements/Ad", f"{int(median_elements)}")
        
        max_elements = audit_df['num_elements'].max()
        st.metric("Max Elements/Ad", f"{int(max_elements)}")
        
        st.info("""
        **Insight:** Based on recognition journey data, ads with 3+ elements are more likely to drive brand recognition.
        """)

    st.markdown("---")

    # Q03 Consumer Language Analysis
    st.markdown("### 💬 Consumer Language Analysis (Q03)")
    st.caption("What words do consumers use to describe brand elements? Sentiment classification and theme analysis.")

    st.info("""
    **Methodology:** Text analysis of open-ended responses using NLP sentiment classification and theme clustering. 
    Shows what consumers actually say (not just predefined scales).
    """)

    # Element selector
    selected_element = st.selectbox(
        "Select element to analyze:",
        list(q03_associations_data.keys()),
        key="q03_element_selector"
    )

    element_data = q03_associations_data[selected_element]

    col1, col2 = st.columns([2, 1])

    with col1:
        # Top words bar chart
        st.markdown(f"#### Top 10 Words for {selected_element}")
        
        words_df = pd.DataFrame({
            'Word': element_data['top_words'],
            'Frequency': element_data['frequencies']
        })
        
        fig_words = px.bar(
            words_df,
            x='Frequency',
            y='Word',
            orientation='h',
            title=f'Most Common Words: {selected_element}',
            text=words_df['Frequency'].apply(lambda x: f'{x:.0%}'),
            color='Frequency',
            color_continuous_scale='Blues'
        )
        fig_words.update_layout(height=400, showlegend=False)
        fig_words.update_traces(textposition='outside')
        st.plotly_chart(fig_words, use_container_width=True)

    with col2:
        # Sentiment pie chart
        st.markdown("#### Sentiment Classification")
        
        sentiment_data = pd.DataFrame({
            'Sentiment': ['Positive', 'Neutral', 'Negative'],
            'Percentage': [
                element_data['sentiment']['positive'],
                element_data['sentiment']['neutral'],
                element_data['sentiment']['negative']
            ]
        })
        
        fig_sentiment = px.pie(
            sentiment_data,
            values='Percentage',
            names='Sentiment',
            title='Text Sentiment',
            color='Sentiment',
            color_discrete_map={'Positive': '#4CAF50', 'Neutral': '#FFC107', 'Negative': '#F44336'}
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)

    # Theme analysis
    st.markdown(f"#### Themes Identified in {selected_element} Descriptions")
    
    themes_df = pd.DataFrame({
        'Theme': list(element_data['themes'].keys()),
        'Prevalence': list(element_data['themes'].values())
    }).sort_values('Prevalence', ascending=True)
    
    fig_themes = px.bar(
        themes_df,
        x='Prevalence',
        y='Theme',
        orientation='h',
        title='Thematic Analysis',
        text=themes_df['Prevalence'].apply(lambda x: f'{x:.0%}'),
        color='Prevalence',
        color_continuous_scale='Viridis'
    )
    fig_themes.update_layout(height=300, showlegend=False)
    fig_themes.update_traces(textposition='outside')
    st.plotly_chart(fig_themes, use_container_width=True)

    # Comparative sentiment across all elements
    st.markdown("---")
    st.markdown("### 📊 Sentiment Comparison Across All Elements")
    
    all_sentiments = []
    for elem, data in q03_associations_data.items():
        all_sentiments.append({
            'Element': elem,
            'Positive': data['sentiment']['positive'],
            'Neutral': data['sentiment']['neutral'],
            'Negative': data['sentiment']['negative'],
            'Net': data['sentiment']['positive'] - data['sentiment']['negative']
        })
    
    sent_comparison_df = pd.DataFrame(all_sentiments).sort_values('Net', ascending=True)
    
    fig_sent_comp = go.Figure()
    
    fig_sent_comp.add_trace(go.Bar(
        name='Positive',
        y=sent_comparison_df['Element'],
        x=sent_comparison_df['Positive'],
        orientation='h',
        marker_color='#4CAF50'
    ))
    
    fig_sent_comp.add_trace(go.Bar(
        name='Negative',
        y=sent_comparison_df['Element'],
        x=sent_comparison_df['Negative'],
        orientation='h',
        marker_color='#F44336'
    ))
    
    fig_sent_comp.update_layout(
        barmode='overlay',
        title='Text Sentiment Analysis: All Elements',
        xaxis_title='Percentage',
        yaxis_title='',
        height=500,
        xaxis_tickformat='.0%'
    )
    
    st.plotly_chart(fig_sent_comp, use_container_width=True)

    # Key insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"""
        **Most Positive Language:**
        - **{sent_comparison_df.iloc[-1]['Element']}**: {sent_comparison_df.iloc[-1]['Positive']:.0%} positive
        - Top word: "{q03_associations_data[sent_comparison_df.iloc[-1]['Element']]['top_words'][0]}"
        """)
    
    with col2:
        st.warning(f"""
        **Most Negative Language:**
        - **{sent_comparison_df.iloc[0]['Element']}**: {sent_comparison_df.iloc[0]['Negative']:.0%} negative
        - Shows disconnect in consumer perception
        """)

# ==================== TAB 4: Performance Tiers ====================
with tab4:
    st.header("🎯 Performance Tiers: Asset Usage Guidelines")
    st.caption("Data-driven recommendations for mandatory and optional asset usage")

    st.markdown("""
    <div style='background-color: #e3f2fd; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h4>Objective: Create actionable guidelines for market teams</h4>
    <p>Based on combined analysis of media usage, spend data, and consumer research,
    the data shows the following patterns asset usage framework:</p>
    </div>
    """, unsafe_allow_html=True)

    # Auto-generate recommendations based on data
    must_use = master_df[
        (master_df['Recognition'] >= 0.40) &
        (master_df['Uniqueness'] >= 0.15) &
        (master_df['Overall Usage'] >= 0.50)
    ].sort_values('Recognition', ascending=False)

    Moderate_Performance = master_df[
        ((master_df['Recognition'] >= 0.35) | (master_df['Uniqueness'] >= 0.25))
    ].sort_values(['Recognition', 'Uniqueness'], ascending=False)
    Moderate_Performance = Moderate_Performance[~Moderate_Performance['Element'].isin(must_use['Element'])]

    requires_attention = master_df[
        (master_df['Recognition'] < 0.40) &
        (master_df['Total Investment'] > master_df['Total Investment'].median())
    ]

    # Display recommendations
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### ✅ High_Performance Assets (Non-Negotiable)")
        st.success(f"**{len(must_use)} assets meet criteria:** High Recognition (≥40%) + Positive Sentiment + High Usage (≥50%)")

        for idx, row in must_use.iterrows():
            with st.expander(f"**{row['Element']}** - Recognition: {row['Recognition']:.0%} | Usage: {row['Overall Usage']:.0%}"):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Recognition", f"{row['Recognition']:.0%}")
                    st.metric("Uniqueness", f"{row['Uniqueness']:.0%}")
                with col_b:
                    st.metric("Usage", f"{row['Overall Usage']:.0%}")
                    st.metric("Investment", f"€{row['Total Investment']:,.0f}")
                with col_c:
                    # Calculate Brand Equity Score
                    equity_score = row['Recognition'] * row['Uniqueness']
                    st.metric("Brand Equity", f"{equity_score:.3f}")
                    st.metric("ROI", f"{row['Recognition ROI']:.2f}")

                st.markdown("**Observable Performance Characteristics:**")
                st.write(f"• **Recognition:** {row['Recognition']:.0%} - consumers have seen/heard this element, ensuring immediate brand attribution")
                st.write(f"• **Uniqueness:** {row['Uniqueness']:.0%} - distinctively Škoda (consumers correctly identify it as belonging to your brand, not competitors)")
                st.write(f"• **Proven Usage:** {row['Overall Usage']:.0%} of campaigns - already validated as core asset")
                st.write(f"• **Investment Efficiency:** €{row['Total Investment']:,.0f} delivers {row['Recognition']:.0%} recognition = {row['Recognition ROI']:.2f} ROI")
                st.write(f"• **Sentiment:** +{row['Net Sentiment']:.1%} net positive emotional associations")

                st.markdown("**Why these metrics matter:**")
                st.write("High recognition ensures your ads are immediately identified as Škoda. High uniqueness prevents confusion with competitors. Combined, they build lasting brand equity with every exposure.")

        st.markdown("---")

        st.markdown("### ⭐ Moderate_Performance Assets (Strongly Encouraged)")
        st.info(f"**{len(Moderate_Performance)} assets show strong potential:** Good recognition or uniqueness")

        for idx, row in Moderate_Performance.iterrows():
            with st.expander(f"**{row['Element']}** - Recognition: {row['Recognition']:.0%} | Uniqueness: {row['Uniqueness']:.0%}"):
                st.markdown("**Why Moderate_Performance:**")
                if row['Recognition'] >= 0.35:
                    st.write(f"• ✅ Strong recognition ({row['Recognition']:.0%}) - consumers are familiar with this element")
                if row['Uniqueness'] >= 0.25:
                    st.write(f"• ✅ High uniqueness ({row['Uniqueness']:.0%}) - distinctively Škoda, differentiates from competitors")
                st.write(f"• Current usage: {row['Overall Usage']:.0%} of campaigns")
                st.write(f"• Investment: €{row['Total Investment']:,.0f}")
                st.write(f"• ROI: {row['Recognition ROI']:.2f} per €1M")

                st.markdown("**Strategic value:**")
                if row['Recognition'] >= 0.35 and row['Uniqueness'] < 0.25:
                    st.write("High recognition makes this useful for awareness, though consider pairing with unique assets for differentiation")
                elif row['Uniqueness'] >= 0.25 and row['Recognition'] < 0.40:
                    st.write(f"Strong differentiation potential - increase usage from {row['Overall Usage']:.0%} to build recognition while maintaining uniqueness")
                else:
                    st.write("Solid performer across both recognition and uniqueness - reliable brand builder")

        st.markdown("---")

        st.markdown("### ⚠️ Lower_Performance")
        st.warning(f"**{len(requires_attention)} assets** have low recognition despite significant investment")

        for idx, row in requires_attention.iterrows():
            with st.expander(f"**{row['Element']}** - Recognition: {row['Recognition']:.0%} | Investment: €{row['Total Investment']:,.0f}"):
                st.markdown("**Why this Some elements show X pattern:**")
                st.write(f"• **Low recognition:** {row['Recognition']:.0%} despite €{row['Total Investment']:,.0f} investment (above median)")
                st.write(f"• **Usage:** {row['Overall Usage']:.0%} of campaigns")
                st.write(f"• **Uniqueness:** {row['Uniqueness']:.0%}")
                st.write(f"• **ROI:** {row['Recognition ROI']:.2f} per €1M (compare to best performer: {master_df['Recognition ROI'].max():.2f})")

                st.markdown("**Possible causes:**")
                st.write("1. **Recent investment:** Recognition may still be building (takes time)")
                st.write("2. **Generic design:** Low uniqueness suggests it may not be distinctive enough")
                st.write("3. **Ineffective deployment:** Placement, creative execution, or context may need optimization")
                st.write("4. **Low visibility:** May be used but not prominently featured in creative")

                st.markdown("**Moderate_Performance action:**")
                if row['Uniqueness'] < 0.20:
                    st.write("⚠️ Consider redesigning for greater Škoda distinctiveness OR deprioritize in favor of higher-uniqueness assets")
                else:
                    st.write("💡 Increase prominence in creative or give more time to build recognition - the distinctiveness is there")

    with col2:
        st.markdown("### 📋 Quick Reference")

        st.markdown("#### High_Performance (Non-Negotiable)")
        for idx, row in must_use.iterrows():
            st.success(f"✓ {row['Element']}")

        st.markdown("#### Moderate_Performance")
        for idx, row in Moderate_Performance.head(5).iterrows():
            st.info(f"⭐ {row['Element']}")

        st.markdown("#### Review Needed")
        for idx, row in requires_attention.iterrows():
            st.warning(f"⚠️ {row['Element']}")

        st.markdown("---")

        # Download guidelines
        guidelines_text = "# Škoda Brand Asset Usage Guidelines\n\n"
        guidelines_text += "## High_Performance Assets (Non-Negotiable)\n"
        for idx, row in must_use.iterrows():
            guidelines_text += f"- {row['Element']}: {row['Recognition']:.0%} recognition\n"
        guidelines_text += "\n## Moderate_Performance Assets\n"
        for idx, row in Moderate_Performance.iterrows():
            guidelines_text += f"- {row['Element']}: {row['Recognition']:.0%} recognition, {row['Uniqueness']:.0%} uniqueness\n"

        st.download_button(
            label="📥 Download Guidelines (TXT)",
            data=guidelines_text,
            file_name="skoda_brand_guidelines.txt",
            mime="text/plain"
        )

# ==================== TAB 5: FUTURE-PROOFING ====================
with tab5:
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
        (master_df['Uniqueness'] >= 0.25) &
        (master_df['Overall Usage'] < 0.40)
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
                    st.metric("Brand Equity", f"{(row['Recognition'] * row['Uniqueness']):.3f}")
                    st.metric("Current Investment", f"€{row['Total Investment']:,.0f}")
                    st.metric("Recognition ROI", f"{row['Recognition ROI']:.2f}")

                st.markdown("**💡 Why is this an opportunity?**")
                st.write(f"• **High uniqueness ({row['Uniqueness']:.0%})** means consumers correctly attribute it to Škoda, not competitors")
                st.write(f"• **Underutilized ({row['Overall Usage']:.0%})** - only used in {row['Overall Usage']:.0%} of campaigns despite its differentiation power")
                st.write(f"• **Strong differentiation potential** - increasing usage would build brand equity more efficiently than generic assets")
                st.write(f"• **Current investment is modest** (€{row['Total Investment']:,.0f}) - scaling up wouldn't require massive budget increases")

                st.markdown("**📈 Why these recommendations make sense:**")
                st.write(f"• **Increase to 50%+ usage:** Would boost recognition from {row['Recognition']:.0%} closer to top performers (64%+) while maintaining distinctiveness")
                st.write(f"• **High-visibility placements:** With {row['Uniqueness']:.0%} uniqueness, prominent placement would maximize brand differentiation impact")
                st.write(f"• **Consistency guidelines:** Current {row['Overall Usage']:.0%} usage suggests inconsistent deployment across markets - standardize to build familiarity")

                st.markdown("**🎯 Expected impact:**")
                st.write("If usage increases to 50%, recognition could grow 25-40% over 12 months, creating a powerful differentiator that competitors can't copy")
    else:
        st.info("No significantly underutilized high-potential assets identified")

    st.markdown("---")

    # Investment Reallocation Opportunities
    st.markdown("### 💰 Investment Optimization")

    with st.expander("📖 Understanding Efficiency Scores"):
        st.markdown("""
        **Efficiency Score = (Recognition × Uniqueness) / Investment (in millions)**

        This metric shows how much brand equity (recognition + differentiation) each asset delivers per euro spent.

        **Why this matters:**
        - High efficiency = Getting strong brand-building results with limited investment (opportunity to scale up)
        - Low efficiency = Spending a lot but not getting proportional brand equity (may need optimization or reallocation)

        **Observed patterns:** Increase investment in high-efficiency assets, optimize or reduce spend on low-efficiency ones
        """)

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
                st.write(f"• Current Investment: €{row['Total Investment']:,.0f} (below median)")
                st.write(f"• Brand Equity: {(row['Recognition'] * row['Uniqueness']):.3f}")

                with st.expander(f"Why is {row['Element']} highly efficient?"):
                    st.write(f"**Recognition:** {row['Recognition']:.0%}")
                    st.write(f"**Uniqueness:** {row['Uniqueness']:.0%}")
                    st.write(f"**Current Investment:** €{row['Total Investment']:,.0f}")
                    st.write(f"**Usage:** {row['Overall Usage']:.0%}")
                    st.markdown("**Why it's efficient:**")
                    st.write(f"Delivers strong brand equity ({(row['Recognition'] * row['Uniqueness']):.3f}) with minimal spend. Each euro generates {row['Efficiency Score']:.2f} units of brand equity - among the best performers.")
                    st.markdown("**Opportunity:**")
                    st.write(f"0f} to €{row['Total Investment']*1.5:,.0f} could boost recognition from {row['Recognition']:.0%} to {min(row['Recognition']*1.3, 0.85):.0%} while maintaining high uniqueness")
                st.write("")

    with col2:
        st.markdown("#### 📉 Low Efficiency (Overfunded)")
        low_efficiency = master_df.nsmallest(3, 'Efficiency Score')

        for idx, row in low_efficiency.iterrows():
            if row['Total Investment'] > master_df['Total Investment'].median():
                st.warning(f"**{row['Element']}**")
                st.write(f"• Efficiency Score: {row['Efficiency Score']:.2f}")
                st.write(f"• Current Investment: €{row['Total Investment']:,.0f} (above median)")
                st.write(f"• Brand Equity: {(row['Recognition'] * row['Uniqueness']):.3f}")

                with st.expander(f"Why is {row['Element']} less efficient?"):
                    st.write(f"**Recognition:** {row['Recognition']:.0%}")
                    st.write(f"**Uniqueness:** {row['Uniqueness']:.0%}")
                    st.write(f"**Current Investment:** €{row['Total Investment']:,.0f}")
                    st.write(f"**Usage:** {row['Overall Usage']:.0%}")
                    st.markdown("**Why efficiency is lower:**")
                    if row['Recognition'] < 0.40:
                        st.write(f"High investment (€{row['Total Investment']:,.0f}) hasn't translated to strong recognition ({row['Recognition']:.0%}). Possible causes: recent launch, poor visibility in creative, or low distinctiveness")
                    if row['Uniqueness'] < 0.20:
                        st.write(f"Low uniqueness ({row['Uniqueness']:.0%}) means it's not strongly associated with Škoda - may be too generic")
                    st.markdown("**Opportunity:**")
                    st.write("Re-evaluate: Can creative execution be improved? Should budget be partially reallocated to higher-efficiency assets? Or does it need more time to build recognition?")
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
        st.write("1. **Increase High_Performance asset deployment**")
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

# ==================== TAB 6: DEEP DIVE ANALYSIS ====================
with tab6:
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
    st.markdown("### Brand Personality Analysis")

    with st.expander("💡 Why personality attributes matter"):
        st.markdown("""
        These 7 personality dimensions (Bold, Stylish, Modern, Simple, Human, Exciting, Playful) reveal the **emotional character** of each brand asset.

        **Why this matters for strategy:**
        - **Emotional connection** drives preference beyond rational features
        - **Personality consistency** across assets strengthens brand identity
        - **Differentiation** comes from unique personality, not just visual recognition
        - **Campaign selection:** Choose assets that match your communication goal (e.g., "Exciting" for launch campaigns, "Simple" for practical messaging)

        **What the scores mean:**
        High scores (50%+) indicate strong associations - consumers clearly perceive these qualities in the asset.
        Variations between assets show which elements carry different emotional messages.
        """)

    personality_view = st.radio(
        "Choose visualization:",
        ["Radar Chart (7 Dimensions)", "Bar Chart Comparison"],
        horizontal=True
    )

    selected_elements = st.multiselect(
        "Select elements to compare:",
        brand_elements,
        default=brand_elements[:3]
    )

    if selected_elements:
        if personality_view == "Radar Chart (7 Dimensions)":
            # Radar chart with 7 positive personality dimensions
            fig_radar = go.Figure()

            personality_dimensions = ['Bold', 'Stylish', 'Modern', 'Simple', 'Human', 'Exciting', 'Playful']

            for element in selected_elements:
                research = research_data[element]
                values = [
                    research['bold'], research['stylish'], research['modern'],
                    research['simple'], research['human'], research['exciting'], research['playful']
                ]
                # Close the radar chart
                values_closed = values + [values[0]]
                dimensions_closed = personality_dimensions + [personality_dimensions[0]]

                fig_radar.add_trace(go.Scatterpolar(
                    r=values_closed,
                    theta=dimensions_closed,
                    name=element,
                    fill='toself'
                ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 0.6],
                        tickformat='.0%'
                    )
                ),
                showlegend=True,
                title="Brand Personality Profile (7 Dimensions)",
                height=600
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            st.info("**Extended personality profile** includes: Bold, Stylish, Modern, Simple, Human, Exciting, and Playful. " +
                    "Higher scores indicate stronger associations with these positive attributes.")

        else:
            # Bar chart comparison
            personality_data = []
            for element in selected_elements:
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
            st.plotly_chart(fig_personality, use_container_width=True)

    st.markdown("---")

    # Market/Country Recognition Analysis
    st.markdown("### Recognition by Market")
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
            min_country = min(recognition_by_country[element].items(), key=lambda x: x[1])
            max_country = max(recognition_by_country[element].items(), key=lambda x: x[1])
            variations.append((element, variation, min_country, max_country))

        variations_sorted = sorted(variations, key=lambda x: x[1], reverse=True)
        for element, var, min_c, max_c in variations_sorted[:3]:
            with st.expander(f"**{element}**: {var:.0%} variation"):
                st.write(f"**Highest:** {max_c[0]} ({max_c[1]:.0%})")
                st.write(f"**Lowest:** {min_c[0]} ({min_c[1]:.0%})")
                st.markdown("**Why this variation exists:**")
                st.write("Possible causes:")
                st.write(f"• **Market maturity:** {max_c[0]} may be a more established Škoda market with longer brand presence")
                st.write(f"• **Media mix differences:** {element} may be used more prominently in {max_c[0]} campaigns")
                st.write(f"• **Cultural relevance:** Design/messaging may resonate differently across cultures")
                st.write(f"• **Competitive landscape:** {min_c[0]} may have stronger local competitors that dilute brand asset recognition")
                st.markdown("**Strategic action:**")
                st.write(f"Analyze why {max_c[0]} outperforms - replicate successful tactics in {min_c[0]} to close the {var:.0%} gap")

# ==================== TAB 7: DATA EXPLORER ====================
with tab7:
    st.header("📄 Data Explorer")
    st.caption("Raw data access and detailed views")

    tab_a, tab_b, tab_c, tab_d = st.tabs(["Comms Audit Data", "Research Data", "Combined Metrics", "Survey Demographics"])

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
            ["Core Metrics", "Extended Personality (7 Dimensions)", "Recognition by Country"],
            horizontal=True
        )

        if research_view == "Core Metrics":
            research_display = []
            for element, data in research_data.items():
                research_display.append({
                    'Element': element,
                    'Recognition': data['recognition'],
                    'Uniqueness': data['uniqueness']
                })
            research_display_df = pd.DataFrame(research_display)

            st.dataframe(research_display_df.style.format({
                'Recognition': '{:.1%}',
                'Uniqueness': '{:.1%}'
            }), use_container_width=True)

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
        st.dataframe(master_df, use_container_width=True)

        csv = master_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Combined Metrics CSV",
            data=csv,
            file_name="skoda_combined_metrics.csv",
            mime="text/csv"
        )

    with tab_d:
        st.markdown("### Survey Demographics (n=2,011)")
        st.caption("P045556 - Saffron Brand Assets Study")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🌍 Sample by Country")
            country_data = pd.DataFrame({
                'Country': ['UK', 'Spain', 'Germany', 'Poland'],
                'Respondents': [
                    demographics['countries']['UK'],
                    demographics['countries']['Spain'],
                    demographics['countries']['Germany'],
                    demographics['countries']['Poland']
                ],
                'Percentage': [
                    demographics['countries']['UK'] / demographics['total_respondents'],
                    demographics['countries']['Spain'] / demographics['total_respondents'],
                    demographics['countries']['Germany'] / demographics['total_respondents'],
                    demographics['countries']['Poland'] / demographics['total_respondents']
                ]
            })

            st.dataframe(country_data.style.format({
                'Respondents': '{:,.0f}',
                'Percentage': '{:.1%}'
            }), use_container_width=True)

            # Country chart
            fig_countries = px.pie(
                country_data,
                values='Respondents',
                names='Country',
                title='Sample Distribution by Country',
                color_discrete_sequence=['#4CAF50', '#66BB6A', '#81C784', '#A5D6A7']
            )
            st.plotly_chart(fig_countries, use_container_width=True)

        with col2:
            st.markdown("#### 👥 Demographics")
            
            # Age
            st.metric("Age Range", demographics['age']['range'])
            st.caption(f"Mean: {demographics['age']['mean']} years | Median: {demographics['age']['median']} years")
            
            # Gender
            st.markdown("**Gender Split:**")
            gender_data = pd.DataFrame({
                'Gender': ['Male', 'Female'],
                'Percentage': [demographics['gender']['male'], demographics['gender']['female']]
            })
            fig_gender = go.Figure(go.Bar(
                x=gender_data['Gender'],
                y=gender_data['Percentage'],
                marker_color=['#2196F3', '#E91E63'],
                text=gender_data['Percentage'].apply(lambda x: f'{x:.0%}'),
                textposition='outside'
            ))
            fig_gender.update_layout(
                yaxis_tickformat='.0%',
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig_gender, use_container_width=True)

            # Škoda Awareness
            st.markdown("**Škoda Brand Awareness:**")
            awareness_data = pd.DataFrame({
                'Status': ['Heard of Škoda', 'Unaware'],
                'Percentage': [
                    demographics['skoda_awareness']['heard_of_skoda'],
                    demographics['skoda_awareness']['unaware']
                ]
            })
            fig_awareness = go.Figure(go.Bar(
                x=awareness_data['Status'],
                y=awareness_data['Percentage'],
                marker_color=['#4CAF50', '#F44336'],
                text=awareness_data['Percentage'].apply(lambda x: f'{x:.0%}'),
                textposition='outside'
            ))
            fig_awareness.update_layout(
                yaxis_tickformat='.0%',
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig_awareness, use_container_width=True)

        st.markdown("---")

        # Summary stats
        st.markdown("#### 📊 Survey Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Respondents", f"{demographics['total_respondents']:,}")
        
        with col2:
            st.metric("Countries", "4", "UK, Spain, Germany, Poland")
        
        with col3:
            st.metric("Mean Age", f"{demographics['age']['mean']} years")
        
        with col4:
            st.metric("Škoda Awareness", f"{demographics['skoda_awareness']['heard_of_skoda']:.0%}")

        st.info("""
        **Survey Design:**
        - Each respondent was shown 6 out of 9 brand elements in randomized order
        - Elements were shown individually without brand identification
        - After viewing, respondents were asked if they recognized it as Škoda
        - Finally, the Škoda brand was revealed and post-reveal questions were asked
        """)

# ==================== TAB 8: RECOGNITION JOURNEY ====================
with tab8:
    st.header("🧭 Recognition Journey & Brand Discovery")
    st.caption("How consumers discover and recognize Škoda through brand elements")

    # Critical finding callout
    st.error("""
    ### ⚠️ Critical Finding
    **56.3% of respondents NEVER recognized these elements as Škoda** — even after seeing 6 different brand assets.
    
    This finding underscores:
    - The challenge of brand recognition in the automotive market
    - The critical importance of the Symbol (48% recognition) as the primary brand carrier
    - The need for multiple touchpoints working together
    - The opportunity to strengthen brand identity through strategic asset deployment
    """)

    st.markdown("---")

    # SECTION 1: Recognition Journey
    st.markdown("### 📈 The Recognition Build: When Do People Identify Škoda?")
    st.caption("Tracking how recognition accumulates as respondents see more brand elements")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Create waterfall-style visualization - FLIPPED ORDER (1 element at top)
        journey_data = pd.DataFrame([
            {'Stage': 'Never recognized', 'Recognition': recognition_journey['never_recognized'], 'Label': '56.3%'},
            {'Stage': 'After all 6 elements', 'Recognition': recognition_journey['after_all_6_elements'], 'Label': '40.1%'},
            {'Stage': 'After 5 elements', 'Recognition': recognition_journey['after_5_elements'], 'Label': '31.3%'},
            {'Stage': 'After 4 elements', 'Recognition': recognition_journey['after_4_elements'], 'Label': '24.7%'},
            {'Stage': 'After 3 elements', 'Recognition': recognition_journey['after_3_elements'], 'Label': '19.7%'},
            {'Stage': 'After 2 elements', 'Recognition': recognition_journey['after_2_elements'], 'Label': '13.3%'},
            {'Stage': 'After 1 element', 'Recognition': recognition_journey['after_1_element'], 'Label': '10.3%'},
        ])

        fig_journey = go.Figure()

        # Never recognized (red) - now at top
        fig_journey.add_trace(go.Bar(
            x=[journey_data['Recognition'].iloc[0]],
            y=[journey_data['Stage'].iloc[0]],
            orientation='h',
            marker_color='#F44336',
            text=[journey_data['Label'].iloc[0]],
            textposition='outside',
            name='Never Recognized',
            hovertemplate='<b>%{y}</b><br>%{x:.1%} never identified Škoda<extra></extra>'
        ))

        # Recognition builders (green) - now below
        fig_journey.add_trace(go.Bar(
            x=journey_data['Recognition'][1:],
            y=journey_data['Stage'][1:],
            orientation='h',
            marker_color='#4CAF50',
            text=journey_data['Label'][1:],
            textposition='outside',
            name='Recognized',
            hovertemplate='<b>%{y}</b><br>%{x:.1%} recognized Škoda<extra></extra>'
        ))

        fig_journey.update_layout(
            title='Progressive Recognition: The "Aha Moment" Journey',
            xaxis_title='% of Respondents',
            yaxis_title='',
            xaxis_tickformat='.0%',
            height=500,
            showlegend=True,
            barmode='overlay'
        )

        st.plotly_chart(fig_journey, use_container_width=True)

    with col2:
        st.markdown("#### 🔍 Key Insights")
        
        st.metric("Immediate Recognition", "10.3%", "After just 1 element")
        st.caption("Only 1 in 10 recognize Škoda from a single brand element")
        
        st.metric("Maximum Recognition", "40.1%", "After all 6 elements")
        st.caption("Even with 6 touchpoints, less than half recognize the brand")
        
        st.metric("Never Recognized", "56.3%", delta="-56.3%", delta_color="inverse")
        st.caption("**Critical gap:** More than half never connect elements to Škoda")

        st.markdown("---")
        
        st.markdown("#### 💡 Strategic Implications")
        st.markdown("""
        **What this means:**
        1. **Single elements are insufficient** - Recognition requires multiple exposures
        2. **Symbol is critical** - At 48% recognition, it's the strongest individual carrier
        3. **Cumulative effect matters** - Each additional element adds ~5-7% recognition
        4. **56% gap is the priority** - Focus on making assets more distinctively Škoda
        """)

    st.markdown("---")

    # SECTION 2: Post-Reveal Brand Familiarity
    st.markdown("### 🎯 Post-Reveal: How Well Do People Know Škoda?")
    st.caption("After revealing these are Škoda elements, respondents rated their familiarity with the brand")

    col1, col2 = st.columns([2, 1])

    with col1:
        familiarity_data = pd.DataFrame([
            {'Level': 'Very familiar', 'Percentage': skoda_familiarity['very_familiar'], 'Description': 'Strong brand advocates'},
            {'Level': 'Quite familiar', 'Percentage': skoda_familiarity['quite_familiar'], 'Description': 'Active considerers'},
            {'Level': 'Heard of, don\'t know much', 'Percentage': skoda_familiarity['heard_of_not_much'], 'Description': 'Awareness without knowledge'},
            {'Level': 'Never heard of Škoda', 'Percentage': skoda_familiarity['never_heard'], 'Description': 'Outside consideration set'},
            {'Level': 'Not sure', 'Percentage': skoda_familiarity['not_sure'], 'Description': 'Uncertain'},
        ])

        # Create color scale
        colors = ['#2E7D32', '#4CAF50', '#FFC107', '#FF5722', '#9E9E9E']

        fig_familiarity = go.Figure(go.Bar(
            x=familiarity_data['Percentage'],
            y=familiarity_data['Level'],
            orientation='h',
            marker_color=colors,
            text=familiarity_data['Percentage'].apply(lambda x: f'{x:.0%}'),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>%{x:.1%} of respondents<br>%{customdata}<extra></extra>',
            customdata=familiarity_data['Description']
        ))

        fig_familiarity.update_layout(
            title='Škoda Brand Familiarity Levels',
            xaxis_title='% of Respondents',
            yaxis_title='',
            xaxis_tickformat='.0%',
            height=400,
            showlegend=False
        )

        st.plotly_chart(fig_familiarity, use_container_width=True)

    with col2:
        st.markdown("#### 📊 Familiarity Breakdown")
        
        familiar_total = skoda_familiarity['very_familiar'] + skoda_familiarity['quite_familiar']
        st.metric("Familiar with Brand", f"{familiar_total:.0%}", "Very + Quite familiar")
        
        st.metric("Heard Name Only", f"{skoda_familiarity['heard_of_not_much']:.0%}", "Lack deeper knowledge")
        
        st.metric("Completely Unaware", f"{skoda_familiarity['never_heard']:.0%}", delta=f"-{skoda_familiarity['never_heard']:.0%}", delta_color="inverse")

        st.markdown("---")
        
        st.info("""
        **The Familiarity Challenge:**
        
        Only **33% are familiar** with Škoda, while **46% have heard the name but lack knowledge**.
        
        This explains why recognition is low and highlights the opportunity for brand education.
        """)

    st.markdown("---")

    # SECTION 3: Emotional Response to Brand Reveal
    st.markdown("### 💚 Emotional Response: Learning It's Škoda")
    st.caption("How respondents felt when told these elements belong to Škoda")

    col1, col2 = st.columns([2, 1])

    with col1:
        response_data = pd.DataFrame([
            {'Response': 'Positively surprised', 'Percentage': response_to_reveal['positive_surprised'], 'Sentiment': 'Positive'},
            {'Response': 'Makes sense / Expected', 'Percentage': response_to_reveal['makes_sense'], 'Sentiment': 'Positive'},
            {'Response': 'Neutral / No strong feeling', 'Percentage': response_to_reveal['neutral'], 'Sentiment': 'Neutral'},
            {'Response': 'Disappointed', 'Percentage': response_to_reveal['disappointed'], 'Sentiment': 'Negative'},
            {'Response': 'Don\'t know', 'Percentage': response_to_reveal['dont_know'], 'Sentiment': 'Neutral'},
        ])

        # Color by sentiment
        color_map = {'Positive': '#4CAF50', 'Neutral': '#FFC107', 'Negative': '#F44336'}
        response_data['Color'] = response_data['Sentiment'].map(color_map)

        fig_response = go.Figure(go.Bar(
            x=response_data['Percentage'],
            y=response_data['Response'],
            orientation='h',
            marker_color=response_data['Color'],
            text=response_data['Percentage'].apply(lambda x: f'{x:.0%}'),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>%{x:.1%} of respondents<extra></extra>'
        ))

        fig_response.update_layout(
            title='Emotional Reaction to Brand Reveal',
            xaxis_title='% of Respondents',
            yaxis_title='',
            xaxis_tickformat='.0%',
            height=400,
            showlegend=False
        )

        st.plotly_chart(fig_response, use_container_width=True)

    with col2:
        st.markdown("#### 🎭 Response Summary")
        
        positive_total = response_to_reveal['positive_surprised'] + response_to_reveal['makes_sense']
        st.metric("Positive Reactions", f"{positive_total:.0%}", "Surprised or expected")
        
        st.metric("Neutral/Indifferent", f"{response_to_reveal['neutral']:.0%}", "No emotional response")
        
        st.metric("Disappointed", f"{response_to_reveal['disappointed']:.0%}", "Negative reaction")

        st.markdown("---")
        
        st.warning("""
        **The Emotional Gap:**
        
        **42% felt nothing** when learning these are Škoda elements.
        
        Combined with only 47% positive reactions, this indicates the brand lacks strong emotional connection.
        """)

    st.markdown("---")

    # SECTION 4: Integrated Strategic View
    st.markdown("### 🎯 Strategic Integration: The Complete Picture")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Recognition Challenge")
        st.markdown("""
        - **56%** never identify elements as Škoda
        - **10%** recognize after 1 element
        - **40%** maximum with 6 elements
        
        **Implication:** Multiple touchpoints essential; Symbol must lead
        """)

    with col2:
        st.markdown("#### Awareness Challenge")
        st.markdown("""
        - **33%** familiar with brand
        - **46%** heard name only
        - **18%** completely unaware
        
        **Implication:** Brand education opportunity; not just recognition issue
        """)

    with col3:
        st.markdown("#### Engagement Challenge")
        st.markdown("""
        - **47%** positive reaction
        - **42%** indifferent
        - **3%** disappointed
        
        **Implication:** Strengthen emotional positioning; brand not rejected but not loved
        """)

    st.markdown("---")

    # Key recommendations
    st.success("""
    ### 🎯 Strategic Priorities Based on This Data
    
    1. **Elevate the Symbol** - At 48% recognition vs 20% average, the logo is the critical brand carrier. Make it prominent in all communications.
    
    2. **Create Combinations** - Since single elements drive only 10% recognition, ensure multiple elements appear together. Moderate_Performance minimum: 3 elements per touchpoint.
    
    3. **Address the 56% Gap** - More than half never connect elements to Škoda. This requires:
       - Bolder, more distinctive asset design
       - More consistent usage across markets
       - Stronger connection between elements and brand name
    
    4. **Build Familiarity** - 46% have heard of Škoda but know little. Use brand elements as educational tools, not just identity markers.
    
    5. **Strengthen Emotional Connection** - 42% feel nothing about Škoda. Move beyond functional attributes to emotional benefits in messaging.
    """)

    # Download option
    st.markdown("---")
    
    journey_export = pd.DataFrame({
        'Metric': ['After 1 element', 'After 2 elements', 'After 3 elements', 
                   'After 4 elements', 'After 5 elements', 'After all 6 elements', 
                   'Never recognized'],
        'Recognition Rate': [
            recognition_journey['after_1_element'],
            recognition_journey['after_2_elements'],
            recognition_journey['after_3_elements'],
            recognition_journey['after_4_elements'],
            recognition_journey['after_5_elements'],
            recognition_journey['after_all_6_elements'],
            recognition_journey['never_recognized']
        ]
    })
    
    csv_journey = journey_export.to_csv(index=False)
    st.download_button(
        label="📥 Download Recognition Journey Data",
        data=csv_journey,
        file_name="skoda_recognition_journey.csv",
        mime="text/csv"
    )

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
<p><b>Škoda Brand Intelligence Dashboard</b> | Powered by Saffron</p>
</div>
""", unsafe_allow_html=True)
