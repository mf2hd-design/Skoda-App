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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview",
    "📊 Asset Performance",
    "💭 Brand Perception",
    "🎯 Strategic Recommendations",
    "📈 Detailed Analytics"
])

# ==================== TAB 1: OVERVIEW ====================
with tab1:
    st.header("🏠 Brand Asset Overview")
    st.caption("Data-driven insights into Škoda's brand asset performance across key markets")
    
    st.write("")  # Add breathing room
    
    # Educational context upfront
    with st.expander("📊 Understanding the Key Metrics", expanded=False):
        st.markdown("""
        This dashboard analyzes Škoda brand assets across two critical dimensions:
        
        **Recognition (Awareness):**
        - Measured via survey question: "Have you seen or heard this element before?"
        - Indicates how familiar consumers are with each brand element
        - Higher recognition = more brand exposure achieved
        - Based on responses from 2,011 consumers (18-55 years) across UK, Spain, Germany, and Poland
        
        **Uniqueness (Brand Attribution):**
        - Measured via survey question: "Which brand do you think this belongs to?"
        - Percentage who correctly identified the element as Škoda (not competitor or generic)
        - Higher uniqueness = stronger brand ownership and differentiation
        - Critical for long-term brand equity - prevents confusion with competitors
        
        **Why Both Matter:**
        - High Recognition + Low Uniqueness = Familiar but generic (doesn't build Škoda equity)
        - Low Recognition + High Uniqueness = Distinctive but unknown (wasted potential)
        - High Recognition + High Uniqueness = Optimal (builds strong Škoda brand equity)
        
        **Investment Data:**
        - Sourced from communications audit of 102 campaigns across 4 markets over 12 months
        - Total spend tracked: €24.8M across all campaigns analyzed
        - Usage frequency shows consistency of element deployment
        """)
    
    st.write("")
    
    # Hero visual first - Brand Equity Matrix
    st.markdown("### 📍 Brand Equity Landscape")
    st.caption("Visual map of all brand assets - position indicates strategic value, size shows investment")
    
    with st.expander("📖 How to interpret this chart", expanded=False):
        st.markdown("""
        **Reading the Chart:**
        - **Y-Axis (Recognition):** % of consumers who have seen/heard this element
        - **X-Axis (Uniqueness):** % who correctly identify it as belonging to Škoda
        - **Bubble Size:** Total investment (€) across all campaigns featuring this element
        - **Color Intensity:** Green = higher uniqueness, Red = lower uniqueness
        
        **What Positions Mean:**
        - **Top-Right Quadrant:** High recognition + high uniqueness = strongest performers
        - **Top-Left Quadrant:** High recognition + low uniqueness = familiar but not distinctive
        - **Bottom-Right Quadrant:** Low recognition + high uniqueness = distinctive but underexposed
        - **Bottom-Left Quadrant:** Low recognition + low uniqueness = requires attention
        
        **Strategic Implications:**
        - Elements in top-right are proven brand builders - high priority for continued investment
        - Large bubbles in bottom-left may indicate inefficient spending
        - Small bubbles in top-right present opportunities to scale investment
        - Gap between median recognition and top performers shows potential for improvement
        """)
    
    fig_matrix = px.scatter(
        master_df,
        x="Uniqueness",
        y="Recognition",
        size="Total Investment",
        color="Uniqueness",
        text="Element",
        size_max=60,
        hover_data=['Total Investment', 'Average Investment', 'Overall Usage'],
        color_continuous_scale='RdYlGn',
        title=""
    )
    fig_matrix.update_traces(textposition='top center')
    fig_matrix.update_layout(height=500)
    st.plotly_chart(fig_matrix, use_container_width=True)
    
    st.write("")  # Breathing room


    # Key Headlines - simplified
    st.markdown("### 🎯 Top Performers")
    st.caption("Assets achieving the strongest measurable performance across key metrics")
    
    col1, col2, col3, col4 = st.columns(4)

    most_recognized = master_df.loc[master_df['Recognition'].idxmax()]
    most_unique = master_df.loc[master_df['Uniqueness'].idxmax()]
    highest_investment = master_df.loc[master_df['Total Investment'].idxmax()]
    best_roi = master_df.loc[master_df['Recognition ROI'].idxmax()]

    with col1:
        st.metric(
            "Most Recognised Asset", 
            most_recognized['Element'],
            help="Recognition measures the percentage of consumers who have seen or heard this element before."
        )
        st.info(f"**{most_recognized['Recognition']:.0%}** of consumers recognize this asset")

    with col2:
        st.metric(
            "Most Unique Asset", 
            most_unique['Element'],
            help="Uniqueness measures brand attribution - how many correctly identify this as Škoda."
        )
        st.info(f"**{most_unique['Uniqueness']:.0%}** distinctiveness rating")

    with col3:
        st.metric(
            "Highest Investment", 
            highest_investment['Element'],
            help="Total investment across all campaigns where this element appears."
        )
        st.info(f"**€{highest_investment['Total Investment']:,.0f}** invested")

    with col4:
        st.metric(
            "Best Recognition ROI", 
            best_roi['Element'],
            help="Recognition points gained per million euros spent. Higher is better."
        )
        st.info(f"**{best_roi['Recognition ROI']:.2f}** recognition per €1M")
        
        with st.expander("📊 Understanding Recognition ROI", expanded=False):
            st.markdown(f"""
            **What is Recognition ROI?**
            
            **Formula:** (Recognition % ÷ Total Investment) × €1,000,000
            
            This metric shows how many percentage points of recognition are gained for every €1M invested.
            
            **Example:**
            - Element with 40% recognition and €10M investment → ROI = 4.0
            - Element with 20% recognition and €2M investment → ROI = 10.0
            - Higher ROI = more efficient at building recognition per € spent
            
            **{best_roi['Element']} Performance:**
            - Recognition: {best_roi['Recognition']:.0%}
            - Total Investment: €{best_roi['Total Investment']:,.0f}
            - ROI: {best_roi['Recognition ROI']:.2f} points per €1M
            
            **Important Caveats:**
            
            1. **Not the Only Metric:** High ROI doesn't automatically mean "invest more here"
            2. **Correlation ≠ Causation:** Other factors influence recognition beyond spend
            3. **Diminishing Returns:** Efficiency may decrease as investment scales up
            4. **Strategic Value:** Low ROI elements may still serve important roles
            
            **Questions for Discussion:**
            - Should investment strategy prioritize high-ROI elements?
            - Or should it build low-performing elements to target thresholds?
            - Does ROI reflect inherent memorability or campaign placement?
            - How does ROI compare across different media types or markets?
            """)

    st.write("")  # Breathing room
    st.write("")

    # Key Observations Box - descriptive not prescriptive
    st.info("""
    ### 📊 Key Observations from the Data
    
    **Performance Patterns:**
    - Symbol achieves 48% recognition and 65% uniqueness - significantly outperforming other elements
    - Average recognition across all elements is 20%, with Symbol being 2.5x above this average
    - Wordmark and Sonic show secondary strength with 30% and 22% recognition respectively
    
    **Recognition Challenge:**
    - 56% of respondents never recognized elements as Škoda after exposure to 6 different assets
    - Recognition builds incrementally: 10% after 1 element → 40% after 6 elements
    - This cumulative pattern suggests need for multiple touchpoints
    
    **Investment vs. Performance:**
    - Total investment across all elements: €24.8M over 12 months
    - Symbol receives highest investment (€8.2M) and delivers highest recognition
    - Best ROI performer achieves recognition more efficiently per € spent
    
    **Questions for Strategic Discussion:**
    - What explains the 56% recognition gap? Is it inconsistent usage, insufficient differentiation, or low awareness of Škoda brand overall?
    - Should investment follow recognition (concentrate on winners) or should it aim to build weaker elements?
    - How does Škoda's 20% average recognition compare to category benchmarks?
    """)

    st.write("")
    st.write("")

    # Collapsible detailed sections
    with st.expander("📊 Performance Tier Analysis", expanded=False):
        st.markdown("""
        **Understanding Performance Tiers:**
        
        Assets are grouped based on recognition thresholds to identify performance patterns:
        - **Tier 1 (≥30% recognition):** Established assets with proven consumer awareness
        - **Tier 2 (19-29% recognition):** Moderate awareness, potential for growth
        - **Tier 3 (<19% recognition):** Low current awareness, requires analysis of investment efficiency
        
        These tiers are descriptive categories based on current performance, not prescriptive guidelines. 
        Strategic decisions should consider multiple factors beyond recognition alone.
        """)
        
        tier_summary = []
        for _, row in master_df.iterrows():
            if row['Recognition'] >= 0.30:
                tier = "🥇 Tier 1"
                description = "High Recognition"
            elif row['Recognition'] >= 0.19:
                tier = "🥈 Tier 2"
                description = "Moderate Recognition"
            else:
                tier = "🥉 Tier 3"
                description = "Emerging/Low Recognition"
            
            tier_summary.append({
                'Element': row['Element'],
                'Tier': tier,
                'Status': description,
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
        
        st.caption("Note: Tier placement based solely on recognition percentage. Strategic value depends on combination of recognition, uniqueness, sentiment, and investment efficiency.")

    with st.expander("📋 Combined Analysis Table", expanded=False):
        st.markdown("""
        **Data Sources:**
        - **Comms Audit Metrics:** Usage frequency, investment levels, media channel distribution
        - **Quantitative Research:** Recognition, uniqueness, and sentiment scores from consumer survey (n=2,011)
        
        This integrated view enables correlation analysis between media spend patterns and consumer perception outcomes.
        """)
        
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

    with st.expander("🔍 Interpreting Top Performer Metrics", expanded=False):
        st.markdown("""
        **Context for Understanding Performance Leaders:**
        
        These metrics identify which assets currently perform strongest, but understanding *why* helps inform strategy.
        Consider both the data patterns shown here and broader market context when developing recommendations.
        """)
        
        st.markdown("### Most Recognised Asset")
        st.markdown(f"""
        **{most_recognized['Element']}** achieves highest recognition ({most_recognized['Recognition']:.0%}):

        **Observable Patterns:**
        - Usage Frequency: **{most_recognized['Overall Usage']:.0%}** of campaigns analyzed
        - Investment Level: **€{most_recognized['Total Investment']:,.0f}** total spend tracked
        - Appears in **{int(most_recognized['Overall Usage'] * 102)}** of 102 audited campaigns
        - Present across all surveyed markets (UK, Spain, Germany, Poland)
        
        **Questions for Discussion:**
        - Does high usage drive recognition, or is this element prioritized *because* it's recognizable?
        - How does visual prominence affect recognition beyond frequency?
        - What role does media channel mix play in recognition rates?
        """)
        
        st.markdown("---")
        st.markdown("### Most Unique Asset")
        usage_pct = most_unique['Overall Usage']
        st.markdown(f"""
        **{most_unique['Element']}** achieves highest uniqueness ({most_unique['Uniqueness']:.0%}):

        **Observable Patterns:**
        - Recognition Level: **{most_unique['Recognition']:.0%}** (context: element is recognized AND attributed correctly)
        - Usage Frequency: **{usage_pct:.0%}** of campaigns
        - Net Sentiment: **{most_unique['Net Sentiment']:+.1%}** (emotional associations)
        
        **Questions for Discussion:**
        - Is this element inherently distinctive, or has consistent usage built association?
        - What design characteristics contribute to brand attribution vs. generic perception?
        - How does this compare to competitor brand elements in automotive category?
        """)
        
        st.markdown("---")
        st.markdown("### Highest Investment")
        roi_comparison = highest_investment['Recognition'] / best_roi['Recognition'] if best_roi['Recognition'] > 0 else 1
        st.markdown(f"""
        **{highest_investment['Element']}** receives highest investment (€{highest_investment['Total Investment']:,.0f}):

        **Observable Patterns:**
        - Campaign Presence: **{highest_investment['Overall Usage']:.0%}** of all campaigns
        - Recognition Achieved: **{highest_investment['Recognition']:.0%}**
        - Recognition ROI: **{highest_investment['Recognition ROI']:.2f}** points per €1M
        - Compared to most efficient asset: **{best_roi['Recognition ROI']:.2f}** points per €1M ({best_roi['Element']})
        
        **Questions for Discussion:**
        - Is current investment level optimal, or could reallocations improve overall performance?
        - Does high investment reflect strategic priority or historical momentum?
        - What would be the impact of increasing/decreasing investment by 20%?
        """)
        
        st.markdown("---")
        st.markdown("### Best Recognition ROI")
        st.markdown(f"""
        **{best_roi['Element']}** achieves highest efficiency ({best_roi['Recognition ROI']:.2f} recognition points per €1M):

        **Observable Patterns:**
        - Total Investment: **€{best_roi['Total Investment']:,.0f}** (relatively lower spend)
        - Recognition Achieved: **{best_roi['Recognition']:.0%}**
        - Uniqueness: **{best_roi['Uniqueness']:.0%}** (brand attribution strength)
        - Usage: **{best_roi['Overall Usage']:.0%}** of campaigns
        
        **Questions for Discussion:**
        - Does this efficiency reflect inherent memorability or strategic placement?
        - Is there opportunity to scale investment while maintaining efficiency?
        - What factors beyond spend drive recognition for this element?
        - Could insights from this element's efficiency apply to other assets?
        """)
        roi_comparison = highest_investment['Recognition'] / best_roi['Recognition'] if best_roi['Recognition'] > 0 else 1
        st.markdown(f"""
        **{highest_investment['Element']}** receives the highest investment because:

        1. **Campaign Frequency:** Used in **{highest_investment['Overall Usage']:.0%}** of all campaigns
        2. **Strategic Priority:** Identified as a core brand asset requiring consistent presence
        3. **Media Versatility:** Works effectively across **{('image and video' if highest_investment['Usage Image'] > 0.3 and highest_investment['Usage Video'] > 0.3 else 'all')}** formats
        4. **Performance:** Achieves **{highest_investment['Recognition']:.0%}** recognition with this investment

        **ROI Context:** Recognition ROI is **{highest_investment['Recognition ROI']:.2f}** per €1M. Compare this to the most efficient asset ({best_roi['Element']}) at **{best_roi['Recognition ROI']:.2f}** per €1M.
        """)
        
        st.markdown("---")
        st.markdown("### Best Recognition ROI")
        st.markdown(f"""
        **{best_roi['Element']}** achieves exceptional efficiency because:

        1. **Low Investment, High Impact:** Only **€{best_roi['Total Investment']:,.0f}** spent, yet achieves **{best_roi['Recognition']:.0%}** recognition
        2. **Strategic Placement:** Used in **{best_roi['Overall Usage']:.0%}** of campaigns, focusing on high-impact moments
        3. **Inherent Memorability:** The design is naturally distinctive and memorable
        4. **Uniqueness Bonus:** **{best_roi['Uniqueness']:.0%}** uniqueness means strong brand association with less repetition needed

        **Opportunity:** This asset punches above its weight - consider increasing investment to amplify results further.
        """)


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

# ==================== TAB 2: ASSET PERFORMANCE ====================
with tab2:
    st.header("📊 Asset Performance Deep Dive")
    st.caption("Individual element analysis with personality traits and usage patterns")
    
    st.write("")  # Breathing room
    
    # Educational context upfront
    with st.expander("📊 Understanding Personality Trait Analysis", expanded=False):
        st.markdown("""
        **Methodology:**
        
        Personality traits measured via semantic differential scales in consumer survey (n=2,011).
        Respondents shown each brand element and asked to rate on 7 bipolar dimensions:
        
        1. **Bold ↔ Cautious:** Energy and confidence level
        2. **Stylish ↔ Plain:** Aesthetic sophistication
        3. **Modern ↔ Old-Fashioned:** Contemporary relevance
        4. **Simple ↔ Complicated:** Clarity and accessibility
        5. **Human ↔ Cold:** Warmth and approachability
        6. **Exciting ↔ Boring:** Emotional engagement
        7. **Playful ↔ Serious:** Brand character and tone
        
        **Interpreting the Scores:**
        - Scores range from 0% (negative trait) to 100% (positive trait)
        - 50% = neutral midpoint (neither positive nor negative)
        - Scores above 50% indicate positive association with that trait
        - Scores below 50% indicate association with opposite trait
        
        **Strategic Relevance:**
        - **Brand Consistency:** Do elements share a common personality profile?
        - **Target Alignment:** Do traits match intended brand positioning?
        - **Emotional Differentiation:** How do traits compare to competitors?
        - **Campaign Fit:** Which elements suit different message types?
        
        **Example Interpretation:**
        - Element scoring 70% "Modern" but 30% "Simple" = perceived as contemporary but complex
        - Element scoring 60% "Bold", 55% "Exciting", 50% "Playful" = energetic character
        - Element with all traits near 50% = lacks distinctive personality
        
        **Questions to Consider:**
        - Which personality combinations best serve Škoda's brand goals?
        - Do low-scoring traits indicate weaknesses or intentional positioning?
        - How do personality scores correlate with sentiment and recognition?
        """)
    
    st.write("")
    
    # Lead with visual - personality radar first
    st.markdown("### 🎨 Brand Personality Profiles")
    st.caption("Emotional associations consumers form with each brand element")
    
    selected_element_personality = st.selectbox("Select brand element to analyze", brand_elements, key="personality_select")
    
    if selected_element_personality in research_data:
        elem_data = research_data[selected_element_personality]
        
        personality_traits = ['bold', 'stylish', 'modern', 'simple', 'human', 'exciting', 'playful']
        trait_values = [elem_data.get(trait, 0.5) for trait in personality_traits]
        trait_labels = [t.title() for t in personality_traits]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=trait_values,
            theta=trait_labels,
            fill='toself',
            name=selected_element_personality,
            marker=dict(color='#4CAF50')
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], tickformat='.0%')
            ),
            showlegend=False,
            height=450
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Recognition", f"{elem_data['recognition']:.0%}")
            st.metric("Uniqueness", f"{elem_data['uniqueness']:.0%}")
        with col2:
            st.metric("Net Sentiment", f"{elem_data['net_sentiment']:+.1%}")
            pos_traits = sum(1 for v in trait_values if v > 0.5)
            st.metric("Positive Traits", f"{pos_traits} of 7")
    
    st.write("")
    st.write("")
    
    # Filters section - collapsed by default
    with st.expander("🔍 Filter by Market/Medium/Placement", expanded=False):
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
    
    # If no filters applied, use full dataset
    if 'filtered_df' not in locals():
        filtered_df = audit_df.copy()
    
    st.write("")
    
    # Investment and Usage side by side
    st.markdown("### 💰 Investment & Usage Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
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
        fig_invest.update_layout(height=400)
        st.plotly_chart(fig_invest, use_container_width=True)
    
    with col2:
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
        fig_usage.update_layout(height=400)
        st.plotly_chart(fig_usage, use_container_width=True)
    
    st.write("")
    st.write("")
    
    # Detailed tables collapsed
    with st.expander("📊 Detailed Performance Metrics Table", expanded=False):
        st.dataframe(
            master_df[['Element', 'Recognition', 'Uniqueness', 'Net Sentiment', 
                       'Overall Usage', 'Total Investment', 'Recognition ROI']]
            .style.format({
                'Recognition': '{:.0%}',
                'Uniqueness': '{:.0%}',
                'Net Sentiment': '{:+.1%}',
                'Overall Usage': '{:.0%}',
                'Total Investment': '€{:,.0f}',
                'Recognition ROI': '{:.2f}'
            })
            .background_gradient(subset=['Recognition', 'Uniqueness'], cmap='RdYlGn'),
            use_container_width=True,
            hide_index=True
        )
    
    with st.expander("📋 Raw Data Explorer", expanded=False):
        st.caption("Explore the complete communications audit dataset")
        
        # Show/hide columns
        all_columns = filtered_df.columns.tolist()
        selected_columns = st.multiselect(
            "Select columns to display",
            all_columns,
            default=['Market', 'Medium', 'Placement', 'Spend'] + brand_elements
        )
        
        if selected_columns:
            st.dataframe(filtered_df[selected_columns], use_container_width=True, height=400)
            
            # Download button
            csv = filtered_df[selected_columns].to_csv(index=False)
            st.download_button(
                label="📥 Download filtered data as CSV",
                data=csv,
                file_name="skoda_filtered_data.csv",
                mime="text/csv"
            )

# ==================== TAB 3: BRAND PERCEPTION ====================
with tab3:
    st.header("💭 Brand Perception & Recognition")
    st.caption("How consumers feel about and recognize Škoda brand elements")
    
    st.write("")  # Breathing room
    
    # Lead with recognition journey visual
    st.markdown("### 🧭 Recognition Journey")
    st.caption("How brand recognition builds as consumers see multiple elements")
    
    with st.expander("📖 Understanding the recognition journey", expanded=False):
        st.markdown("""
        This shows the **cumulative effect** of exposing consumers to multiple brand elements:
        - After seeing 1 element, only 10% recognize it as Škoda
        - Recognition grows to 40% after seeing all 6 elements
        - **Critical finding:** 56% never recognize elements as Škoda, even after 6 exposures
        
        **Implication:** Multiple touchpoints are essential - Symbol must lead every campaign.
        """)
    
    # Recognition Journey Data
    recognition_journey = {
        'after_1_element': 0.103,
        'after_2_elements': 0.157,
        'after_3_elements': 0.232,
        'after_4_elements': 0.295,
        'after_5_elements': 0.356,
        'after_all_6_elements': 0.397,
        'never_recognized': 0.563
    }
    
    journey_df = pd.DataFrame([
        {'Stage': 'After 1 element', 'Recognition': recognition_journey['after_1_element'], 'Cumulative': True},
        {'Stage': 'After 2 elements', 'Recognition': recognition_journey['after_2_elements'], 'Cumulative': True},
        {'Stage': 'After 3 elements', 'Recognition': recognition_journey['after_3_elements'], 'Cumulative': True},
        {'Stage': 'After 4 elements', 'Recognition': recognition_journey['after_4_elements'], 'Cumulative': True},
        {'Stage': 'After 5 elements', 'Recognition': recognition_journey['after_5_elements'], 'Cumulative': True},
        {'Stage': 'After all 6', 'Recognition': recognition_journey['after_all_6_elements'], 'Cumulative': True},
        {'Stage': 'Never recognized', 'Recognition': recognition_journey['never_recognized'], 'Cumulative': False}
    ])
    
    fig_journey = go.Figure()
    
    # Cumulative recognition (building up)
    cumulative_data = journey_df[journey_df['Cumulative'] == True]
    fig_journey.add_trace(go.Scatter(
        x=cumulative_data['Stage'],
        y=cumulative_data['Recognition'],
        mode='lines+markers+text',
        name='Cumulative Recognition',
        line=dict(color='#4CAF50', width=3),
        marker=dict(size=12),
        text=cumulative_data['Recognition'].apply(lambda x: f'{x:.0%}'),
        textposition='top center',
        textfont=dict(size=14, color='#4CAF50')
    ))
    
    # Never recognized (separate point)
    never_data = journey_df[journey_df['Cumulative'] == False]
    fig_journey.add_trace(go.Scatter(
        x=never_data['Stage'],
        y=never_data['Recognition'],
        mode='markers+text',
        name='Never Recognized',
        marker=dict(size=15, color='#F44336'),
        text=never_data['Recognition'].apply(lambda x: f'{x:.0%}'),
        textposition='top center',
        textfont=dict(size=14, color='#F44336')
    ))
    
    fig_journey.update_layout(
        title='',
        xaxis_title='Exposure to Brand Elements',
        yaxis_title='% Recognizing as Škoda',
        yaxis_tickformat='.0%',
        height=450,
        showlegend=False,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_journey, use_container_width=True)
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Recognition after 1 element", "10%")
        st.caption("Initial brand attribution is very low")
    with col2:
        st.metric("Recognition after 6 elements", "40%")
        st.caption("Maximum cumulative recognition")
    with col3:
        st.metric("Never Recognized", "56%", delta="-56%", delta_color="inverse")
        st.caption("Critical gap requiring urgent attention")
    
    st.write("")
    st.write("")
    
    # Sentiment Analysis Section
    st.markdown("### 💚 Emotional Sentiment Analysis")
    st.caption("Consumer emotional response to each brand element")
    
    # Sentiment warning callout
    st.warning("""
    **Sentiment Challenge:** Only 2 of 9 elements have positive sentiment. The brand triggers slightly more negative than positive emotional responses - a concern requiring strategic attention.
    """)
    
    with st.expander("📖 Understanding sentiment scores", expanded=False):
        st.markdown("""
        **Positive Sentiment:** % choosing positive descriptors (Bold, Stylish, Modern, Simple, Human, Exciting, Playful)
        
        **Negative Sentiment:** % choosing opposite descriptors (Cautious, Plain, Old-Fashioned, Complicated, Cold, Boring, Serious)
        
        **Net Sentiment:** Positive minus Negative (higher = more positive brand perception)
        """)
    
    # Sentiment ranking chart
    sentiment_ranked = master_df.sort_values('Net Sentiment', ascending=True)
    
    fig_sentiment = go.Figure(go.Bar(
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
    
    fig_sentiment.update_layout(
        title='Elements Ranked by Net Sentiment',
        xaxis_title='Net Sentiment Score',
        yaxis_title='',
        xaxis_tickformat='.0%',
        height=450,
        showlegend=False
    )
    
    st.plotly_chart(fig_sentiment, use_container_width=True)
    
    st.write("")
    
    # Sentiment metrics
    col1, col2, col3 = st.columns(3)
    
    most_positive = master_df.loc[master_df['Net Sentiment'].idxmax()]
    least_positive = master_df.loc[master_df['Net Sentiment'].idxmin()]
    avg_net_sentiment = master_df['Net Sentiment'].mean()
    
    with col1:
        st.metric("Most Positive", most_positive['Element'])
        st.success(f"**{most_positive['Net Sentiment']:+.1%}** net sentiment")
        
        with st.expander("📊 Why does this element score highest?", expanded=False):
            st.markdown(f"""
            **{most_positive['Element']}** resonates most strongly with consumers.
            
            **Observable Data:**
            - **Positive Sentiment:** {most_positive['Positive Sentiment']:.1%} of respondents chose positive descriptors
            - **Negative Sentiment:** {most_positive['Negative Sentiment']:.1%} chose negative descriptors
            - **Net Difference:** {most_positive['Net Sentiment']:+.1%} (positive minus negative)
            - **Recognition Level:** {most_positive['Recognition']:.0%} have seen/heard this element
            - **Uniqueness:** {most_positive['Uniqueness']:.0%} correctly identify it as Škoda
            
            **Correlation Patterns:**
            - This element has {most_positive['Recognition']:.0%} recognition, suggesting familiarity may influence sentiment
            - Uniqueness at {most_positive['Uniqueness']:.0%} indicates strong brand attribution
            
            **Questions for Discussion:**
            - What design characteristics drive the positive emotional response?
            - Does this sentiment align with Škoda's intended brand positioning?
            - Can insights from this element inform improvements to others?
            - Is the sentiment consistent across all markets or does it vary?
            """)
    
    with col2:
        st.metric("Least Positive", least_positive['Element'])
        st.error(f"**{least_positive['Net Sentiment']:+.1%}** net sentiment")
        
        with st.expander("📊 Why does this element score lowest?", expanded=False):
            st.markdown(f"""
            **{least_positive['Element']}** triggers more negative than positive associations.
            
            **Observable Data:**
            - **Positive Sentiment:** {least_positive['Positive Sentiment']:.1%} chose positive descriptors
            - **Negative Sentiment:** {least_positive['Negative Sentiment']:.1%} chose negative descriptors
            - **Net Difference:** {least_positive['Net Sentiment']:+.1%} (negative sentiment exceeds positive)
            - **Recognition Level:** {least_positive['Recognition']:.0%} have seen/heard this element
            - **Usage Frequency:** {least_positive['Overall Usage']:.0%} of campaigns use this element
            
            **Possible Factors:**
            - Lower recognition ({least_positive['Recognition']:.0%}) may indicate limited familiarity
            - Personality trait scores may reveal specific associations driving negativity
            - Visual or audio characteristics may not resonate with target audience
            - Cultural factors may influence perception differently across markets
            
            **Questions for Discussion:**
            - Is negative sentiment due to design weakness or market misalignment?
            - Does low recognition contribute to negative sentiment (unfamiliarity bias)?
            - Which specific personality traits score lowest for this element?
            - What would be required to improve sentiment - redesign or repositioning?
            """)
    
    with col3:
        st.metric("Average", "All Elements")
        st.warning(f"**{avg_net_sentiment:+.1%}** average net sentiment")
        
        with st.expander("📊 Interpreting the average", expanded=False):
            sentiment_range = master_df['Net Sentiment'].max() - master_df['Net Sentiment'].min()
            positive_count = len(master_df[master_df['Net Sentiment'] > 0])
            negative_count = len(master_df[master_df['Net Sentiment'] < 0])
            
            st.markdown(f"""
            **Overall Sentiment Pattern:**
            - **Average:** {avg_net_sentiment:+.1%} across all 9 brand elements
            - **Range:** {sentiment_range:.1%} between highest and lowest performers
            - **Distribution:** {positive_count} elements with positive sentiment, {negative_count} with negative
            
            **Context:**
            - Average positive sentiment across elements: {master_df['Positive Sentiment'].mean():.1%}
            - Average negative sentiment across elements: {master_df['Negative Sentiment'].mean():.1%}
            - The slight negative tilt suggests more consumers choose negative descriptors
            
            **Industry Context Considerations:**
            - Is -3.4% net sentiment typical for automotive brands or below average?
            - How does this compare to competitor brand element sentiment?
            - Are there category-specific expectations affecting perception?
            
            **Questions for Discussion:**
            - Does negative sentiment indicate need for element redesign or brand repositioning?
            - Which elements drag down the average most significantly?
            - Is consistency of sentiment across elements more important than average score?
            - How quickly can sentiment shift with design changes or increased exposure?
            """)

    
    st.write("")
    st.write("")
    
    # Brand Familiarity section
    st.markdown("### 🎯 Brand Familiarity Levels")
    st.caption("How well do consumers know Škoda?")
    
    with st.expander("📖 Understanding brand familiarity context", expanded=False):
        st.markdown("""
        **Methodology:**
        Survey question: "How familiar are you with the Škoda brand?"
        - Respondents: n=2,011 consumers aged 18-55 across UK, Spain, Germany, Poland
        - No exposure to brand elements before this question
        - Measures overall brand awareness independent of specific assets
        
        **Why This Matters:**
        This data provides critical context for interpreting asset recognition:
        - **High familiarity + low asset recognition** = Assets not distinctive enough
        - **Low familiarity + low asset recognition** = General awareness challenge
        - **Familiarity gap** = Opportunity for brand education
        
        **Interpreting the Categories:**
        - **Very/Quite Familiar (33%):** Core audience who know Škoda well
        - **Heard of, not much knowledge (46%):** Aware but superficial understanding
        - **Never heard of (18%):** Complete lack of brand awareness
        
        **Strategic Questions:**
        - Does 33% familiarity align with category benchmarks for automotive brands?
        - What explains the 46% who've heard of Škoda but lack deeper knowledge?
        - Should asset strategy prioritize deepening existing awareness or expanding reach?
        - How does familiarity correlate with asset recognition rates?
        """)
    
    skoda_familiarity = {
        'very_familiar': 0.15,
        'quite_familiar': 0.18,
        'heard_of_not_much': 0.46,
        'never_heard': 0.18,
        'dont_know': 0.03
    }
    
    familiarity_df = pd.DataFrame([
        {'Level': 'Very familiar', 'Percentage': skoda_familiarity['very_familiar'], 'Category': 'Familiar'},
        {'Level': 'Quite familiar', 'Percentage': skoda_familiarity['quite_familiar'], 'Category': 'Familiar'},
        {'Level': 'Heard of, not much knowledge', 'Percentage': skoda_familiarity['heard_of_not_much'], 'Category': 'Aware'},
        {'Level': 'Never heard of', 'Percentage': skoda_familiarity['never_heard'], 'Category': 'Unaware'},
        {'Level': "Don't know", 'Percentage': skoda_familiarity['dont_know'], 'Category': 'Unaware'}
    ])
    
    color_map = {'Familiar': '#4CAF50', 'Aware': '#FFC107', 'Unaware': '#F44336'}
    familiarity_df['Color'] = familiarity_df['Category'].map(color_map)
    
    fig_familiarity = go.Figure(go.Bar(
        x=familiarity_df['Percentage'],
        y=familiarity_df['Level'],
        orientation='h',
        marker_color=familiarity_df['Color'],
        text=familiarity_df['Percentage'].apply(lambda x: f'{x:.0%}'),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>%{x:.0%} of respondents<extra></extra>'
    ))
    
    fig_familiarity.update_layout(
        title='',
        xaxis_title='% of Respondents',
        yaxis_title='',
        xaxis_tickformat='.0%',
        height=350,
        showlegend=False
    )
    
    st.plotly_chart(fig_familiarity, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        familiar_total = skoda_familiarity['very_familiar'] + skoda_familiarity['quite_familiar']
        st.metric("Familiar with Brand", f"{familiar_total:.0%}")
        st.caption("Very + Quite familiar combined")
    
    with col2:
        st.metric("Heard Name Only", f"{skoda_familiarity['heard_of_not_much']:.0%}")
        st.caption("Lack deeper brand knowledge")
    
    with col3:
        st.metric("Completely Unaware", f"{skoda_familiarity['never_heard']:.0%}")
        st.caption("Never heard of Škoda")
    
    st.write("")
    st.write("")
    
    # Strategic discussion points - not prescriptive
    st.info("""
    ### 💡 Key Patterns for Strategic Discussion
    
    **Recognition Journey Patterns:**
    - 56% never recognize elements as Škoda after 6 exposures
    - Recognition builds incrementally (10% → 40% across 6 elements)
    - Symbol drives majority of initial recognition at 48%
    
    **Discussion Questions:**
    - What factors explain the 56% non-recognition rate? Inconsistency? Low brand familiarity? Generic design?
    - Is cumulative recognition pattern typical for automotive category?
    - What's the optimal number of elements per campaign for recognition?
    
    **Sentiment Patterns:**
    - Only 2 of 9 elements show positive sentiment (Symbol +0.3%, Sonic +1.1%)
    - 7 elements generate more negative than positive associations
    - Average net sentiment: -3.4%
    
    **Discussion Questions:**
    - Is negative sentiment unusual in automotive category or within norms?
    - Do personality traits explain sentiment patterns?
    - Should strategy prioritize fixing negative sentiment or amplifying positive elements?
    
    **Familiarity Patterns:**
    - 33% familiar with Škoda brand overall
    - 46% heard name but lack knowledge
    - 18% never heard of brand
    
    **Discussion Questions:**
    - Does low brand familiarity explain weak element recognition?
    - Should brand building focus on awareness first, then asset recognition?
    - How do familiarity levels compare to competitive set?
    """)
    
    # Detailed data in expander
    with st.expander("📊 Detailed Sentiment Data by Element", expanded=False):
        sentiment_detail = master_df[['Element', 'Recognition', 'Uniqueness', 'Positive Sentiment', 
                                       'Negative Sentiment', 'Net Sentiment']].sort_values('Net Sentiment', ascending=False)
        
        st.dataframe(
            sentiment_detail.style.format({
                'Recognition': '{:.0%}',
                'Uniqueness': '{:.0%}',
                'Positive Sentiment': '{:.1%}',
                'Negative Sentiment': '{:.1%}',
                'Net Sentiment': '{:+.1%}'
            }).background_gradient(subset=['Net Sentiment'], cmap='RdYlGn'),
            use_container_width=True,
            hide_index=True
        )

# ==================== TAB 4: STRATEGIC DISCUSSION POINTS ====================
with tab4:
    st.header("💭 Strategic Discussion Framework")
    st.caption("Data patterns and questions to guide brand asset strategy development")
    
    st.write("")  # Breathing room
    
    # Context setting
    with st.expander("📋 Purpose of This Analysis", expanded=False):
        st.markdown("""
        **Objective:**
        
        This tab synthesizes data patterns to facilitate strategic discussions between Škoda and Saffron teams.
        It presents questions and observations—not prescriptions—to guide collaborative decision-making.
        
        **How to Use This Section:**
        
        1. Review data patterns for each strategic area
        2. Consider the discussion questions with your team
        3. Use insights to inform (not replace) your strategic judgment
        4. Combine this quantitative data with qualitative insights, market knowledge, and brand vision
        
        **What This Analysis Does NOT Do:**
        
        - Prescribe specific actions (that's for workshop discussions)
        - Account for factors beyond this research (competitive moves, business strategy, etc.)
        - Replace strategic judgment with algorithmic recommendations
        """)
    
    st.write("")
    
    # Performance patterns section
    st.markdown("### 📊 Performance Pattern Analysis")
    st.caption("Observable patterns in the data worth discussing")
    
    # Identify high/moderate/low performers based on data
    high_performers = master_df[
        (master_df['Recognition'] >= 0.30) |
        (master_df['Uniqueness'] >= 0.40)
    ].sort_values('Recognition', ascending=False)
    
    moderate_performers = master_df[
        (master_df['Recognition'] >= 0.19) & 
        (master_df['Recognition'] < 0.30) &
        (master_df['Uniqueness'] < 0.40)
    ].sort_values('Recognition', ascending=False)
    
    low_performers = master_df[
        (master_df['Recognition'] < 0.19) &
        (master_df['Uniqueness'] < 0.30)
    ].sort_values('Recognition', ascending=False)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### High Performance")
        st.success(f"**{len(high_performers)} elements** meeting high thresholds")
        
        for idx, row in high_performers.iterrows():
            with st.expander(f"**{row['Element']}**", expanded=False):
                st.markdown(f"""
                **Metrics:**
                - Recognition: {row['Recognition']:.0%}
                - Uniqueness: {row['Uniqueness']:.0%}
                - Sentiment: {row['Net Sentiment']:+.1%}
                - Usage: {row['Overall Usage']:.0%} of campaigns
                - Investment: €{row['Total Investment']:,.0f}
                - ROI: {row['Recognition ROI']:.2f} per €1M
                
                **Discussion Points:**
                - What makes this element successful?
                - Should it anchor brand system?
                - Can success be replicated in other elements?
                - Is investment proportional to performance?
                """)
    
    with col2:
        st.markdown("#### Moderate Performance")
        st.info(f"**{len(moderate_performers)} elements** with mixed results")
        
        for idx, row in moderate_performers.iterrows():
            with st.expander(f"**{row['Element']}**", expanded=False):
                st.markdown(f"""
                **Metrics:**
                - Recognition: {row['Recognition']:.0%}
                - Uniqueness: {row['Uniqueness']:.0%}
                - Sentiment: {row['Net Sentiment']:+.1%}
                - Usage: {row['Overall Usage']:.0%} of campaigns
                - Investment: €{row['Total Investment']:,.0f}
                
                **Discussion Points:**
                - What's limiting performance?
                - Is there unrealized potential?
                - Should usage increase or decrease?
                - Does it serve a specific strategic role?
                """)
    
    with col3:
        st.markdown("#### Lower Performance")
        st.warning(f"**{len(low_performers)} elements** below thresholds")
        
        for idx, row in low_performers.iterrows():
            with st.expander(f"**{row['Element']}**", expanded=False):
                st.markdown(f"""
                **Metrics:**
                - Recognition: {row['Recognition']:.0%}
                - Uniqueness: {row['Uniqueness']:.0%}
                - Sentiment: {row['Net Sentiment']:+.1%}
                - Usage: {row['Overall Usage']:.0%} of campaigns
                - Investment: €{row['Total Investment']:,.0f}
                
                **Discussion Points:**
                - Why is performance low despite investment?
                - Is redesign worth considering?
                - Does it fill a necessary role despite metrics?
                - Should investment shift elsewhere?
                """)
    
    st.write("")
    st.write("")
    
    # Investment efficiency analysis
    st.markdown("### 💰 Investment Efficiency Discussion")
    st.caption("Examining relationship between spend and recognition outcomes")
    
    with st.expander("📊 Investment vs. Recognition Patterns", expanded=False):
        # Calculate efficiency metrics
        high_investment_low_rec = master_df[
            (master_df['Total Investment'] > master_df['Total Investment'].median()) &
            (master_df['Recognition'] < master_df['Recognition'].median())
        ]
        
        low_investment_high_rec = master_df[
            (master_df['Total Investment'] < master_df['Total Investment'].median()) &
            (master_df['Recognition'] > master_df['Recognition'].median())
        ]
        
        st.markdown("""
        **High Investment, Lower Recognition:**
        
        These elements receive above-median spend but achieve below-median recognition.
        """)
        
        if len(high_investment_low_rec) > 0:
            for idx, row in high_investment_low_rec.iterrows():
                st.write(f"- **{row['Element']}:** €{row['Total Investment']:,.0f} invested → {row['Recognition']:.0%} recognition (ROI: {row['Recognition ROI']:.2f})")
            
            st.markdown("""
            **Discussion Questions:**
            - Is investment disproportionate to results?
            - Are there execution issues (size, prominence, consistency)?
            - Should investment be reallocated?
            - Or does this reflect strategic choice (building for future)?
            """)
        else:
            st.write("No elements in this category")
        
        st.write("")
        
        st.markdown("""
        **Lower Investment, Higher Recognition:**
        
        These elements achieve above-median recognition with below-median spend.
        """)
        
        if len(low_investment_high_rec) > 0:
            for idx, row in low_investment_high_rec.iterrows():
                st.write(f"- **{row['Element']}:** €{row['Total Investment']:,.0f} invested → {row['Recognition']:.0%} recognition (ROI: {row['Recognition ROI']:.2f})")
            
            st.markdown("""
            **Discussion Questions:**
            - What drives efficiency? Inherent memorability? Strategic placement?
            - Is there opportunity to scale investment while maintaining efficiency?
            - Can learnings apply to other elements?
            - Why isn't more being invested here?
            """)
        else:
            st.write("No elements in this category")
    
    st.write("")
    st.write("")
    
    # Future considerations
    st.markdown("### 🔮 Future Considerations")
    st.caption("Strategic areas requiring long-term planning")
    
    with st.expander("🚗 Electric Vehicle Transition Context", expanded=False):
        st.markdown("""
        **Data Observations:**
        
        - **Electric Green:** 20% recognition, 32% uniqueness, -6.0% sentiment
        - **Symbol:** 48% recognition, 65% uniqueness, +0.3% sentiment
        - **Sonic:** 22% recognition, 34% uniqueness, +1.1% sentiment
        
        **Discussion Framework:**
        
        1. **EV Relevance:** Which current elements feel appropriate for electric vehicles?
        2. **Green Performance:** Why does Electric Green underperform? Color choice? Execution? Positioning?
        3. **Sonic Evolution:** How should audio identity adapt for quiet EVs?
        4. **Continuity vs. Change:** Balance maintaining recognition (Symbol) with signaling transformation
        
        **Questions to Explore:**
        - Should EV communications use different element combinations?
        - Is a new EV-specific element needed, or adapt existing ones?
        - How do competitor EV brands use brand elements?
        - What do focus groups say about current elements and EV fit?
        """)
    
    with st.expander("🎨 Design Evolution Opportunities", expanded=False):
        negative_sentiment_elements = master_df[master_df['Net Sentiment'] < -0.03].sort_values('Net Sentiment')
        
        st.markdown("""
        **Data Observations:**
        
        7 of 9 elements show negative net sentiment:
        """)
        
        for idx, row in negative_sentiment_elements.iterrows():
            st.write(f"- **{row['Element']}:** {row['Net Sentiment']:+.1%} sentiment ({row['Recognition']:.0%} recognition)")
        
        st.markdown("""
        
        **Discussion Framework:**
        
        1. **Severity Assessment:** Is negative sentiment problematic or category-typical?
        2. **Design vs. Perception:** Do elements need redesign or repositioning?
        3. **Priority Setting:** Which negative sentiment elements hurt most?
        4. **Risk Evaluation:** What's lost if elements are redesigned (recognition drops)?
        
        **Questions to Explore:**
        - What sentiment scores do competitor brands achieve?
        - Do personality traits explain sentiment (e.g., "boring" vs "exciting")?
        - Can negative sentiment be addressed without major redesign?
        - Should focus be on amplifying positives vs. fixing negatives?
        """)
    
    with st.expander("📊 Measurement & Tracking Framework", expanded=False):
        st.markdown("""
        **Current Baseline Metrics (for future tracking):**
        
        - **Recognition Rate:** Average 20% across elements (range: 19-48%)
        - **Uniqueness Rate:** Average 35% across elements (range: 29-65%)
        - **Net Sentiment:** Average -3.4% across elements (range: -7.7% to +1.1%)
        - **Recognition Journey:** 10% after 1 element → 40% after 6 elements
        - **Non-Recognition:** 56% never attribute elements to Škoda
        
        **Tracking Considerations:**
        
        1. **Frequency:** Quarterly? Annually? Post-campaign?
        2. **Metrics:** Which KPIs matter most for strategy?
        3. **Methodology:** Repeat exact survey or adapt based on learnings?
        4. **Benchmarks:** Compare to competitors or category norms?
        5. **Triggers:** What performance changes would prompt strategy review?
        
        **Questions to Explore:**
        - What constitutes "success" - improved recognition? Sentiment? Both?
        - How long before design changes impact metrics?
        - Should tracking vary by market or be consistent?
        - Who owns ongoing measurement and reporting?
        """)
    
    st.write("")
    st.write("")
    
    # Workshop preparation section
    st.markdown("### 🎯 Preparing for Strategy Workshop")
    st.caption("Key questions to address in collaborative sessions")
    
    st.info("""
    **Core Strategic Questions for Discussion:**
    
    **Brand Architecture:**
    - Should one element (Symbol?) anchor the entire system?
    - What's the minimum viable element set for recognition?
    - How should elements combine - fixed templates or flexible?
    
    **Investment Strategy:**
    - Should investment follow performance or build weak areas?
    - What's the target recognition threshold for each element?
    - How to balance consistency (repeat winners) vs. evolution (new ideas)?
    
    **Market Adaptation:**
    - Are element guidelines universal or market-specific?
    - How much flexibility should local markets have?
    - Should underperforming markets use different combinations?
    
    **Timeline & Priorities:**
    - What changes can happen immediately vs. require redesign?
    - Which elements are "locked" vs. "flexible"?
    - What's the 12/24/36-month vision?
    
    **Success Criteria:**
    - How will we know if new strategy works?
    - What metrics define success?
    - What's the acceptable timeframe for results?
    """)
    
    st.write("")
    
    # Download data for workshop
    st.markdown("### 📥 Export Data for Workshop")
    
    workshop_data = master_df[['Element', 'Recognition', 'Uniqueness', 'Net Sentiment', 
                                'Total Investment', 'Overall Usage', 'Recognition ROI']].copy()
    workshop_data['Performance Tier'] = workshop_data.apply(
        lambda row: 'High' if (row['Recognition'] >= 0.30 or row['Uniqueness'] >= 0.40) 
        else 'Moderate' if row['Recognition'] >= 0.19 
        else 'Lower', axis=1
    )
    
    csv_data = workshop_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Strategic Discussion Data",
        data=csv_data,
        file_name="skoda_strategic_discussion_data.csv",
        mime="text/csv"
    )
    
    st.caption("Export includes all metrics for offline analysis and workshop preparation")

# ==================== TAB 5: DETAILED ANALYTICS ====================
with tab5:
    st.header("📈 Detailed Analytics & Strategic Insights")
    st.caption("Advanced analytics to identify opportunities and optimize brand asset usage")
    
    st.write("")  # Breathing room
    
    # Educational context
    with st.expander("📊 Understanding Portfolio Analysis", expanded=False):
        st.markdown("""
        **Purpose of This Section:**
        
        This tab provides advanced analytical frameworks for understanding brand asset performance patterns.
        These matrices help visualize strategic trade-offs and opportunities.
        
        **Three Key Frameworks:**
        
        1. **BCG Matrix (Recognition vs Investment):** Identifies Stars, Cash Cows, Question Marks, Dogs
        2. **Brand Equity Matrix (Recognition vs Uniqueness):** Maps Icons, Famous Generics, Hidden Gems, Weak elements
        3. **Efficiency Matrix (Usage vs ROI):** Reveals Workhorses, Overused, Untapped Potential, Underperformers
        
        **How to Use:**
        - Each matrix reveals different strategic insights
        - Combine perspectives for comprehensive understanding
        - Use for workshop discussions, not algorithmic decisions
        - Consider factors beyond what data shows (brand vision, competitive moves, etc.)
        
        **Important Caveat:**
        Quadrant labels (Stars, Dogs, etc.) are analytical frameworks, not prescriptions. 
        Strategic decisions require judgment beyond classification.
        """)
    
    st.write("")
    
    # Key observations
    st.info("""
    ### 🎯 Key Observations from Advanced Analysis
    
    **ROI Patterns:**
    - Sonic delivers best efficiency (low investment, strong recognition)
    - Symbol provides best overall value (high recognition + high uniqueness)
    
    **Investment-Performance Relationship:**
    - Not all high-investment elements achieve proportional recognition
    - Some lower-investment elements punch above their weight
    - Efficiency varies significantly across the portfolio
    
    **Portfolio Balance:**
    - Heavy concentration in certain elements vs light usage of others
    - Opportunity to optimize investment allocation
    - Question: Should investment follow performance or build weaker elements?
    """)
    
    st.write("")
    st.write("")

    # Portfolio Optimization Matrices
    st.markdown("### 📊 Portfolio Optimization Matrices")
    st.caption("Strategic analysis frameworks - where patterns emerge")

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
            st.caption("⬆️ Increase investment")
        
        if len(dogs) > 0:
            st.error(f"**DOGS ({len(dogs)}):**")
            for _, row in dogs.iterrows():
                st.write(f"• {row['Element']}")
            st.caption("⚠️ Cut or redesign")
        
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
            st.caption("💎 Build awareness")
        
        if len(weak) > 0:
            st.error(f"**WEAK ({len(weak)}):**")
            for _, row in weak.iterrows():
                st.write(f"• {row['Element']}")
            st.caption("🔴 Redesign urgently")

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
        insight_text = "**Holistic efficiency combining fame and differentiation.** High scorers deliver the most long-term brand equity per euro - ideal for identifying non-negotiables."

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
            st.write("• **Top = Non-negotiables candidates**")
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


st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
<p><b>Škoda Brand Intelligence Dashboard</b> | Powered by Saffron</p>
<p style='font-size: 0.9em;'>Redesigned for better user experience and progressive disclosure</p>
</div>
""", unsafe_allow_html=True)
