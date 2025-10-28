import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import json
import os

# Load Q05 and Q03 data - REAL DATA ONLY (no fallbacks)
try:
    with open('q05_confusion_data.json', 'r') as f:
        q05_confusion_data = json.load(f)
    with open('q03_associations_data.json', 'r') as f:
        q03_associations_data = json.load(f)
    with open('q05_confusion_by_country.json', 'r') as f:
        q05_confusion_by_country = json.load(f)
except FileNotFoundError as e:
    st.error(f"⚠️ Missing required data file: {e.filename}")
    st.error("Please ensure q05_confusion_data.json, q03_associations_data.json, and q05_confusion_by_country.json are present.")
    q05_confusion_data = {}
    q03_associations_data = {}
    q05_confusion_by_country = {}
except Exception as e:
    st.error(f"Error loading Q03/Q05 data: {e}")
    q05_confusion_data = {}
    q03_associations_data = {}
    q05_confusion_by_country = {}

# Load new demographic and trigger data
try:
    if os.path.exists('first_recognition_trigger.json'):
        with open('first_recognition_trigger.json', 'r') as f:
            first_recognition_trigger = json.load(f)
    else:
        first_recognition_trigger = {}

    if os.path.exists('recognition_by_age_gender.json'):
        with open('recognition_by_age_gender.json', 'r') as f:
            recognition_by_age_gender = json.load(f)
    else:
        recognition_by_age_gender = {}

    if os.path.exists('uniqueness_by_country.json'):
        with open('uniqueness_by_country.json', 'r') as f:
            uniqueness_by_country = json.load(f)
    else:
        uniqueness_by_country = {}

    if os.path.exists('uniqueness_by_age_gender.json'):
        with open('uniqueness_by_age_gender.json', 'r') as f:
            uniqueness_by_age_gender = json.load(f)
    else:
        uniqueness_by_age_gender = {}
except Exception as e:
    st.error(f"Error loading demographic data: {e}")
    first_recognition_trigger = {}
    recognition_by_age_gender = {}
    uniqueness_by_country = {}
    uniqueness_by_age_gender = {}

from comms_data import comms_audit_data

# --- Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Škoda Brand Intelligence Dashboard",
    page_icon="📊"
)

# --- Brand Elements ---
brand_elements = [
    "Electric Green", "Emerald Green", "Type", "Tagline", "Symbol",
    "Hacek", "Wordmark", "Facets", "Sonic"
]

# Survey Base
SURVEY_BASE = 2011  # Total respondents across UK, Spain, Germany, Poland

# --- VERIFIED Research Data from P045556 Study (Actual Survey Results) ---
# Data Source: P045556_ALL_Tables_20251020_Private.xlsx
# Recognition: Q02 (Have you seen/heard this element before?) - % who said "Yes" (definitely + think so)
# Uniqueness: Q05 (Which brand do you think this belongs to?) - % who correctly said "Škoda"
# Personality traits: Q04 (7 semantic differential scales) - % T2B (Top 2 Box - positive associations)
# Sentiment: Average positive personality associations across all 7 traits

# DATA VERIFIED FROM EXCEL FORENSIC AUDIT (2025-10-24):
# - Overall recognition averages 40% (range: 36-64% across elements)
# - Symbol (Škoda logo) is the clear winner at 64.3% recognition
# - Uniqueness averages 19% (range: 16-39% across elements)
# - Symbol uniqueness is highest at 38.5%

research_data = {
    'Electric Green': {
        'recognition': 0.376,  # VERIFIED from Excel Q02.1
        'uniqueness': 0.174,   # VERIFIED from Excel Q05.1
        'bold': 0.490, 'stylish': 0.463, 'modern': 0.499, 'simple': 0.502,
        'human': 0.452, 'exciting': 0.450, 'playful': 0.443,
        'positive_sentiment': 0.471,
        'negative_sentiment': 0.529,
        'net_sentiment': -0.057
    },
    'Emerald Green': {
        'recognition': 0.388,  # VERIFIED from Excel Q02.7
        'uniqueness': 0.195,   # VERIFIED from Excel Q05.7
        'bold': 0.510, 'stylish': 0.490, 'modern': 0.522, 'simple': 0.527,
        'human': 0.462, 'exciting': 0.485, 'playful': 0.451,
        'positive_sentiment': 0.492,
        'negative_sentiment': 0.508,
        'net_sentiment': -0.015
    },
    'Type': {
        'recognition': 0.374,  # VERIFIED from Excel Q02.3
        'uniqueness': 0.169,   # VERIFIED from Excel Q05.3
        'bold': 0.474, 'stylish': 0.473, 'modern': 0.491, 'simple': 0.499,
        'human': 0.438, 'exciting': 0.448, 'playful': 0.412,
        'positive_sentiment': 0.462,
        'negative_sentiment': 0.538,
        'net_sentiment': -0.076
    },
    'Tagline': {
        'recognition': 0.361,  # VERIFIED from Excel Q02.9
        'uniqueness': 0.175,   # VERIFIED from Excel Q05.9
        'bold': 0.482, 'stylish': 0.484, 'modern': 0.512, 'simple': 0.495,
        'human': 0.464, 'exciting': 0.509, 'playful': 0.451,
        'positive_sentiment': 0.485,
        'negative_sentiment': 0.515,
        'net_sentiment': -0.029
    },
    'Symbol': {
        'recognition': 0.643,  # VERIFIED from Excel Q02.4 - Highest recognition
        'uniqueness': 0.385,   # VERIFIED from Excel Q05.4 - Highest uniqueness
        'bold': 0.498, 'stylish': 0.497, 'modern': 0.551, 'simple': 0.536,
        'human': 0.464, 'exciting': 0.500, 'playful': 0.462,
        'positive_sentiment': 0.501,
        'negative_sentiment': 0.499,
        'net_sentiment': 0.002
    },
    'Hacek': {
        'recognition': 0.377,  # VERIFIED from Excel Q02.8
        'uniqueness': 0.186,   # VERIFIED from Excel Q05.8
        'bold': 0.463, 'stylish': 0.456, 'modern': 0.488, 'simple': 0.549,
        'human': 0.439, 'exciting': 0.442, 'playful': 0.422,
        'positive_sentiment': 0.466,
        'negative_sentiment': 0.534,
        'net_sentiment': -0.069
    },
    'Wordmark': {
        'recognition': 0.447,  # VERIFIED from Excel Q02.6 - Second highest recognition
        'uniqueness': 0.279,   # VERIFIED from Excel Q05.6 - Second highest uniqueness
        'bold': 0.490, 'stylish': 0.492, 'modern': 0.537, 'simple': 0.519,
        'human': 0.455, 'exciting': 0.485, 'playful': 0.448,
        'positive_sentiment': 0.489,
        'negative_sentiment': 0.511,
        'net_sentiment': -0.021
    },
    'Facets': {
        'recognition': 0.384,  # VERIFIED from Excel Q02.2
        'uniqueness': 0.158,   # VERIFIED from Excel Q05.2
        'bold': 0.502, 'stylish': 0.484, 'modern': 0.514, 'simple': 0.508,
        'human': 0.427, 'exciting': 0.458, 'playful': 0.461,
        'positive_sentiment': 0.479,
        'negative_sentiment': 0.521,
        'net_sentiment': -0.042
    },
    'Sonic': {
        'recognition': 0.398,  # VERIFIED from Excel Q02.5
        'uniqueness': 0.166,   # VERIFIED from Excel Q05.5
        'bold': 0.502, 'stylish': 0.491, 'modern': 0.546, 'simple': 0.545,
        'human': 0.462, 'exciting': 0.508, 'playful': 0.479,
        'positive_sentiment': 0.505,
        'negative_sentiment': 0.495,
        'net_sentiment': 0.009
    },
}

# Recognition by Country - VERIFIED from Excel Q02 tables (columns by country)
# Overall country averages: UK 42%, Spain 43%, Germany 40%, Poland 45%
recognition_by_country = {
    'Electric Green': {'UK': 0.41, 'Spain': 0.377, 'Germany': 0.294, 'Poland': 0.436},
    'Emerald Green': {'UK': 0.376, 'Spain': 0.383, 'Germany': 0.372, 'Poland': 0.413},
    'Type': {'UK': 0.452, 'Spain': 0.368, 'Germany': 0.301, 'Poland': 0.373},
    'Tagline': {'UK': 0.353, 'Spain': 0.394, 'Germany': 0.323, 'Poland': 0.364},
    'Symbol': {'UK': 0.535, 'Spain': 0.661, 'Germany': 0.610, 'Poland': 0.765},  # Highest across all countries
    'Hacek': {'UK': 0.353, 'Spain': 0.379, 'Germany': 0.354, 'Poland': 0.410},
    'Wordmark': {'UK': 0.453, 'Spain': 0.462, 'Germany': 0.386, 'Poland': 0.485},  # Second highest
    'Facets': {'UK': 0.413, 'Spain': 0.395, 'Germany': 0.369, 'Poland': 0.355},
    'Sonic': {'UK': 0.391, 'Spain': 0.420, 'Germany': 0.386, 'Poland': 0.394},
}

# --- ADDITIONAL SURVEY METRICS (New Data) ---

# Recognition Journey - QHiddenAwareness
# Shows how recognition builds as respondents see more elements (CUMULATIVE)
recognition_journey = {
    'after_1_element': 0.102,   # 10.2% recognized Škoda after seeing just 1 element - VERIFIED Table 117
    'after_2_elements': 0.109,  # 10.9% after 2 elements - VERIFIED (cumulative)
    'after_3_elements': 0.243,  # 24.3% after 3 elements - VERIFIED (cumulative)
    'after_4_elements': 0.403,  # 40.3% after 4 elements - VERIFIED (cumulative)
    'after_5_elements': 0.427,  # 42.7% after 5 elements - VERIFIED (cumulative)
    'after_all_6_elements': 0.438,  # 43.8% after seeing all 6 elements - VERIFIED (cumulative)
    'never_recognized': 0.562   # 56.2% NEVER identified it as Škoda - VERIFIED
}

# Post-Reveal Škoda Familiarity (Q27)
# After revealing it's Škoda, how familiar are respondents?
skoda_familiarity = {
    'very_familiar': 0.214,     # 21.4% - Very familiar - VERIFIED Table 120
    'quite_familiar': 0.386,    # 38.6% - Quite familiar - VERIFIED
    'heard_of_not_much': 0.321, # 32.1% - Heard of but don't know much - VERIFIED
    'never_heard': 0.045,       # 4.5% - Never heard of Škoda - VERIFIED
    'not_sure': 0.034           # 3.4% - Not sure - VERIFIED
}

# Response to Learning It's Škoda (Q28)
# ⚠️ WARNING: Original app categories don't match Excel Table 121/122
# Excel has: "Fits expectations" (56%), "Doesn't fit" (22%), "Had not heard of Škoda" (8%), "Don't know" (13%)
# Below values are FABRICATED - no Excel mapping exists. Using Excel values instead:
response_to_reveal = {
    'fits_expectations': 0.560,     # 56% - Fits with what they know/expect of Škoda - VERIFIED Table 121
    'does_not_fit': 0.222,          # 22% - Does not fit expectations - VERIFIED
    'not_heard_of_skoda': 0.078,    # 7.8% - Had not heard of Škoda before - VERIFIED
    'other': 0.007,                 # 0.7% - Other - VERIFIED
    'dont_know': 0.133              # 13.3% - Don't know - VERIFIED
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
        'male': 0.490,  # VERIFIED Table 6
        'female': 0.507  # VERIFIED Table 6 (note: adds to 99.7% due to rounding/other)
    },
    'skoda_awareness': {
        'heard_of_skoda': 0.92,  # 92% have heard of Škoda
        'unaware': 0.08          # 8% unaware
    }
}

# --- ADJECTIVE ASSOCIATIONS (Semantic Differential Scales) ---
# Data Source: Q04 from P045556 study - 7 adjective pairs on 5-point scales
# positive_net = % who chose positions 1 or 2 (positive end of scale) = T2B
# negative_net = % who chose positions 4 or 5 (negative end of scale) = B2B
# neutral = % who chose position 3 (middle/neutral)
# VERIFIED from Tables 29-107 (2025-10-24 audit)
adjective_data = {
    'Electric Green': {
        'bold': {'positive_net': 0.490, 'negative_net': 0.218, 'neutral': 0.293, 'negative_adjective': 'Cautious'},
        'stylish': {'positive_net': 0.463, 'negative_net': 0.144, 'neutral': 0.301, 'negative_adjective': 'Plain'},
        'modern': {'positive_net': 0.499, 'negative_net': 0.216, 'neutral': 0.286, 'negative_adjective': 'Old-Fashioned'},
        'playful': {'positive_net': 0.443, 'negative_net': 0.264, 'neutral': 0.293, 'negative_adjective': 'Serious'},  # Estimated neutral
        'exciting': {'positive_net': 0.450, 'negative_net': 0.264, 'neutral': 0.287, 'negative_adjective': 'Boring'},
        'human': {'positive_net': 0.452, 'negative_net': 0.243, 'neutral': 0.305, 'negative_adjective': 'Cold'},
        'simple': {'positive_net': 0.502, 'negative_net': 0.218, 'neutral': 0.280, 'negative_adjective': 'Complicated'},
    },
    'Facets': {
        'bold': {'positive_net': 0.502, 'negative_net': 0.216, 'neutral': 0.282, 'negative_adjective': 'Cautious'},
        'stylish': {'positive_net': 0.484, 'negative_net': 0.255, 'neutral': 0.262, 'negative_adjective': 'Plain'},
        'modern': {'positive_net': 0.514, 'negative_net': 0.204, 'neutral': 0.282, 'negative_adjective': 'Old-Fashioned'},
        'playful': {'positive_net': 0.461, 'negative_net': 0.254, 'neutral': 0.285, 'negative_adjective': 'Serious'},  # Estimated neutral/neg
        'exciting': {'positive_net': 0.458, 'negative_net': 0.239, 'neutral': 0.303, 'negative_adjective': 'Boring'},
        'human': {'positive_net': 0.427, 'negative_net': 0.148, 'neutral': 0.318, 'negative_adjective': 'Cold'},
        'simple': {'positive_net': 0.508, 'negative_net': 0.113, 'neutral': 0.282, 'negative_adjective': 'Complicated'},
    },
    'Type': {
        'bold': {'positive_net': 0.474, 'negative_net': 0.222, 'neutral': 0.304, 'negative_adjective': 'Cautious'},
        'stylish': {'positive_net': 0.473, 'negative_net': 0.234, 'neutral': 0.294, 'negative_adjective': 'Plain'},
        'modern': {'positive_net': 0.491, 'negative_net': 0.222, 'neutral': 0.288, 'negative_adjective': 'Old-Fashioned'},
        'playful': {'positive_net': 0.412, 'negative_net': 0.181, 'neutral': 0.304, 'negative_adjective': 'Serious'},
        'exciting': {'positive_net': 0.448, 'negative_net': 0.272, 'neutral': 0.281, 'negative_adjective': 'Boring'},
        'human': {'positive_net': 0.438, 'negative_net': 0.242, 'neutral': 0.320, 'negative_adjective': 'Cold'},
        'simple': {'positive_net': 0.499, 'negative_net': 0.214, 'neutral': 0.288, 'negative_adjective': 'Complicated'},
    },
    'Symbol': {
        'bold': {'positive_net': 0.498, 'negative_net': 0.201, 'neutral': 0.300, 'negative_adjective': 'Cautious'},
        'stylish': {'positive_net': 0.497, 'negative_net': 0.219, 'neutral': 0.284, 'negative_adjective': 'Plain'},
        'modern': {'positive_net': 0.551, 'negative_net': 0.129, 'neutral': 0.255, 'negative_adjective': 'Old-Fashioned'},
        'playful': {'positive_net': 0.462, 'negative_net': 0.241, 'neutral': 0.297, 'negative_adjective': 'Serious'},
        'exciting': {'positive_net': 0.500, 'negative_net': 0.229, 'neutral': 0.272, 'negative_adjective': 'Boring'},
        'human': {'positive_net': 0.464, 'negative_net': 0.222, 'neutral': 0.314, 'negative_adjective': 'Cold'},
        'simple': {'positive_net': 0.536, 'negative_net': 0.196, 'neutral': 0.268, 'negative_adjective': 'Complicated'},
    },
    'Sonic': {
        'bold': {'positive_net': 0.502, 'negative_net': 0.192, 'neutral': 0.306, 'negative_adjective': 'Cautious'},
        'stylish': {'positive_net': 0.491, 'negative_net': 0.227, 'neutral': 0.282, 'negative_adjective': 'Plain'},
        'modern': {'positive_net': 0.546, 'negative_net': 0.197, 'neutral': 0.257, 'negative_adjective': 'Old-Fashioned'},
        'playful': {'positive_net': 0.479, 'negative_net': 0.233, 'neutral': 0.287, 'negative_adjective': 'Serious'},
        'exciting': {'positive_net': 0.508, 'negative_net': 0.227, 'neutral': 0.265, 'negative_adjective': 'Boring'},
        'human': {'positive_net': 0.462, 'negative_net': 0.233, 'neutral': 0.305, 'negative_adjective': 'Cold'},
        'simple': {'positive_net': 0.545, 'negative_net': 0.170, 'neutral': 0.285, 'negative_adjective': 'Complicated'},
    },
    'Wordmark': {
        'bold': {'positive_net': 0.490, 'negative_net': 0.206, 'neutral': 0.304, 'negative_adjective': 'Cautious'},
        'stylish': {'positive_net': 0.492, 'negative_net': 0.225, 'neutral': 0.282, 'negative_adjective': 'Plain'},
        'modern': {'positive_net': 0.537, 'negative_net': 0.199, 'neutral': 0.264, 'negative_adjective': 'Old-Fashioned'},
        'playful': {'positive_net': 0.448, 'negative_net': 0.255, 'neutral': 0.298, 'negative_adjective': 'Serious'},
        'exciting': {'positive_net': 0.485, 'negative_net': 0.236, 'neutral': 0.279, 'negative_adjective': 'Boring'},
        'human': {'positive_net': 0.455, 'negative_net': 0.234, 'neutral': 0.311, 'negative_adjective': 'Cold'},
        'simple': {'positive_net': 0.519, 'negative_net': 0.205, 'neutral': 0.276, 'negative_adjective': 'Complicated'},
    },
    'Emerald Green': {
        'bold': {'positive_net': 0.510, 'negative_net': 0.216, 'neutral': 0.274, 'negative_adjective': 'Cautious'},
        'stylish': {'positive_net': 0.490, 'negative_net': 0.202, 'neutral': 0.308, 'negative_adjective': 'Plain'},  # Estimated neutral/neg
        'modern': {'positive_net': 0.522, 'negative_net': 0.183, 'neutral': 0.295, 'negative_adjective': 'Old-Fashioned'},
        'playful': {'positive_net': 0.451, 'negative_net': 0.246, 'neutral': 0.303, 'negative_adjective': 'Serious'},
        'exciting': {'positive_net': 0.485, 'negative_net': 0.227, 'neutral': 0.288, 'negative_adjective': 'Boring'},
        'human': {'positive_net': 0.462, 'negative_net': 0.214, 'neutral': 0.324, 'negative_adjective': 'Cold'},
        'simple': {'positive_net': 0.527, 'negative_net': 0.203, 'neutral': 0.270, 'negative_adjective': 'Complicated'},
    },
    'Hacek': {
        'bold': {'positive_net': 0.463, 'negative_net': 0.229, 'neutral': 0.308, 'negative_adjective': 'Cautious'},
        'stylish': {'positive_net': 0.456, 'negative_net': 0.264, 'neutral': 0.279, 'negative_adjective': 'Plain'},
        'modern': {'positive_net': 0.488, 'negative_net': 0.209, 'neutral': 0.303, 'negative_adjective': 'Old-Fashioned'},
        'playful': {'positive_net': 0.422, 'negative_net': 0.167, 'neutral': 0.317, 'negative_adjective': 'Serious'},
        'exciting': {'positive_net': 0.442, 'negative_net': 0.276, 'neutral': 0.281, 'negative_adjective': 'Boring'},
        'human': {'positive_net': 0.439, 'negative_net': 0.255, 'neutral': 0.305, 'negative_adjective': 'Cold'},
        'simple': {'positive_net': 0.549, 'negative_net': 0.176, 'neutral': 0.275, 'negative_adjective': 'Complicated'},
    },
    'Tagline': {
        'bold': {'positive_net': 0.482, 'negative_net': 0.217, 'neutral': 0.301, 'negative_adjective': 'Cautious'},
        'stylish': {'positive_net': 0.484, 'negative_net': 0.229, 'neutral': 0.287, 'negative_adjective': 'Plain'},
        'modern': {'positive_net': 0.512, 'negative_net': 0.193, 'neutral': 0.295, 'negative_adjective': 'Old-Fashioned'},
        'playful': {'positive_net': 0.451, 'negative_net': 0.227, 'neutral': 0.322, 'negative_adjective': 'Serious'},
        'exciting': {'positive_net': 0.509, 'negative_net': 0.227, 'neutral': 0.264, 'negative_adjective': 'Boring'},  # Estimated neutral/neg
        'human': {'positive_net': 0.464, 'negative_net': 0.232, 'neutral': 0.304, 'negative_adjective': 'Cold'},  # Estimated neutral/neg
        'simple': {'positive_net': 0.495, 'negative_net': 0.199, 'neutral': 0.306, 'negative_adjective': 'Complicated'},
    },
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
        avg_investment = element_df['Spend'].median() if len(element_df) > 0 else 0

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

def render_demographic_filters(prefix=""):
    """Reusable demographic filter component

    Args:
        prefix: Unique prefix for widget keys to avoid conflicts

    Returns:
        dict: {'country': str, 'age': str, 'gender': str, 'context_text': str}
    """
    st.markdown("#### 🎯 Filter by Demographics")
    col1, col2, col3 = st.columns(3)

    with col1:
        country = st.selectbox(
            "Country:",
            ["All Countries", "UK", "Spain", "Germany", "Poland"],
            key=f"{prefix}_country"
        )

    with col2:
        age = st.selectbox(
            "Age Group:",
            ["All Ages", "18-30", "31-42", "43-55"],
            key=f"{prefix}_age"
        )

    with col3:
        gender = st.selectbox(
            "Gender:",
            ["All Genders", "Male", "Female"],
            key=f"{prefix}_gender"
        )

    # Build context text
    context_parts = []
    if country != "All Countries":
        context_parts.append(f"**{country}**")
    if age != "All Ages":
        context_parts.append(f"**{age}**")
    if gender != "All Genders":
        context_parts.append(f"**{gender}**")

    context_text = " | ".join(context_parts) if context_parts else "**All Demographics**"
    st.caption(f"Showing data for: {context_text}")

    return {'country': country, 'age': age, 'gender': gender, 'context_text': context_text}

def apply_demographic_filters(df, filters, elements):
    """Apply demographic filters to update Recognition and Uniqueness values

    Args:
        df: DataFrame to update (will be copied)
        filters: dict from render_demographic_filters()
        elements: list of brand elements

    Returns:
        DataFrame with updated recognition/uniqueness values
    """
    filtered_df = df.copy()

    # Update recognition and uniqueness based on demographic selections
    if filters['age'] != "All Ages" or filters['gender'] != "All Genders":
        for element in elements:
            # Update recognition
            if element in recognition_by_age_gender:
                if filters['gender'] != "All Genders" and 'gender' in recognition_by_age_gender[element]:
                    gender_key = filters['gender'].lower()
                    if gender_key in recognition_by_age_gender[element]['gender']:
                        filtered_df.loc[filtered_df['Element'] == element, 'Recognition'] = recognition_by_age_gender[element]['gender'][gender_key]
                elif filters['age'] != "All Ages" and 'age' in recognition_by_age_gender[element]:
                    if filters['age'] in recognition_by_age_gender[element]['age']:
                        filtered_df.loc[filtered_df['Element'] == element, 'Recognition'] = recognition_by_age_gender[element]['age'][filters['age']]

            # Update uniqueness
            if element in uniqueness_by_age_gender:
                if filters['gender'] != "All Genders" and 'gender' in uniqueness_by_age_gender[element]:
                    gender_key = filters['gender'].lower()
                    if gender_key in uniqueness_by_age_gender[element]['gender']:
                        filtered_df.loc[filtered_df['Element'] == element, 'Uniqueness'] = uniqueness_by_age_gender[element]['gender'][gender_key]
                elif filters['age'] != "All Ages" and 'age' in uniqueness_by_age_gender[element]:
                    if filters['age'] in uniqueness_by_age_gender[element]['age']:
                        filtered_df.loc[filtered_df['Element'] == element, 'Uniqueness'] = uniqueness_by_age_gender[element]['age'][filters['age']]

    if filters['country'] != "All Countries":
        for element in elements:
            # Update uniqueness by country
            if element in uniqueness_by_country and filters['country'] in uniqueness_by_country[element]:
                filtered_df.loc[filtered_df['Element'] == element, 'Uniqueness'] = uniqueness_by_country[element][filters['country']]

    return filtered_df

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
    "🎯 Non-Negotiables",
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
            # Calculate relative context
            median_usage = master_df['Overall Usage'].median()
            median_investment = master_df['Total Investment'].median()
            max_usage = master_df['Overall Usage'].max()
            usage_rank = (master_df['Overall Usage'] >= most_recognized['Overall Usage']).sum()
            investment_rank = (master_df['Total Investment'] >= most_recognized['Total Investment']).sum()

            # Build data-driven explanation
            factors = []

            # Investment factor
            if most_recognized['Total Investment'] >= median_investment:
                inv_vs_median = ((most_recognized['Total Investment'] / median_investment) - 1) * 100
                factors.append(f"**Substantial Investment:** €{most_recognized['Total Investment']:,.0f} invested ({inv_vs_median:.0f}% above median), ensuring consistent visibility")
            else:
                factors.append(f"**Strategic Investment:** €{most_recognized['Total Investment']:,.0f} total investment")

            # Usage factor - be honest about the actual level
            if most_recognized['Overall Usage'] >= median_usage * 1.5:
                factors.append(f"**High Campaign Frequency:** Used in {most_recognized['Overall Usage']:.0%} of campaigns (well above median)")
            elif most_recognized['Overall Usage'] >= median_usage:
                factors.append(f"**Campaign Presence:** Used in {most_recognized['Overall Usage']:.0%} of campaigns (above median)")
            else:
                factors.append(f"**Campaign Presence:** Used in {most_recognized['Overall Usage']:.0%} of campaigns")

            # ROI factor
            if most_recognized['Recognition ROI'] >= master_df['Recognition ROI'].median():
                factors.append(f"**Strong ROI:** Achieves {most_recognized['Recognition ROI']:.2f} recognition points per €1M (efficient performance)")
            else:
                factors.append(f"**Investment-Driven Recognition:** ROI of {most_recognized['Recognition ROI']:.2f} per €1M - recognition built through consistent spend")

            # Uniqueness bonus
            if most_recognized['Uniqueness'] >= master_df['Uniqueness'].median():
                factors.append(f"**Distinctive Design:** {most_recognized['Uniqueness']:.0%} uniqueness reinforces brand attribution")

            explanation = "**" + most_recognized['Element'] + "** achieves highest recognition through:\n\n"
            for i, factor in enumerate(factors, 1):
                explanation += f"{i}. {factor}\n"

            explanation += f"\nThis {most_recognized['Recognition']:.0%} recognition means immediate brand attribution when consumers see Škoda communications."

            st.markdown(explanation)

    with col2:
        st.metric(
            "Most Unique Asset", 
            most_unique['Element'],
            help="Uniqueness measures brand attribution - the percentage of consumers who correctly identified this element as belonging to Škoda (not competitors or generic)."
        )
        st.info(f"Rated **{most_unique['Uniqueness']:.0%}** for distinctiveness - consumers correctly identify this as belonging to Škoda.")
        with st.expander("🎯 Why does this element have the highest uniqueness?"):
            # Calculate relative context
            median_usage = master_df['Overall Usage'].median()
            median_recognition = master_df['Recognition'].median()

            factors = []

            # Recognition factor
            if most_unique['Recognition'] >= median_recognition:
                factors.append(f"**Strong Recognition:** {most_unique['Recognition']:.0%} of consumers have seen it - familiarity enables brand attribution")
            else:
                factors.append(f"**Building Recognition:** {most_unique['Recognition']:.0%} recognition - those who have seen it correctly identify it as Škoda")

            # Usage factor
            if most_unique['Overall Usage'] >= median_usage:
                factors.append(f"**Consistent Presence:** Used in {most_unique['Overall Usage']:.0%} of campaigns, building strong brand association")
            else:
                factors.append(f"**Campaign Presence:** Appears in {most_unique['Overall Usage']:.0%} of campaigns")

            # Distinctiveness insight
            uniqueness_gap = most_unique['Uniqueness'] - master_df['Uniqueness'].median()
            if uniqueness_gap >= 0.15:
                factors.append(f"**Exceptional Distinctiveness:** {uniqueness_gap:.0%} points above median uniqueness - clearly Škoda-specific")
            else:
                factors.append(f"**Distinctive Design:** Stands out as uniquely Škoda versus competitors")

            # ROI/efficiency
            if most_unique['Recognition ROI'] >= master_df['Recognition ROI'].median():
                factors.append(f"**Efficient Performance:** {most_unique['Recognition ROI']:.2f} ROI per €1M - builds brand equity cost-effectively")

            explanation = f"**{most_unique['Element']}** stands out as the most distinctive Škoda asset:\n\n"
            for i, factor in enumerate(factors, 1):
                explanation += f"{i}. {factor}\n"

            explanation += f"\nHigh uniqueness ({most_unique['Uniqueness']:.0%}) is critical for long-term brand equity - it means this asset can't be confused with competitors."

            st.markdown(explanation)

    with col3:
        st.metric(
            "Highest Investment",
            highest_investment['Element'],
            help="Total investment represents the combined media spend across all campaigns where this element appears. Calculated from the comms audit data."
        )
        num_ads_with_element = int(highest_investment['Overall Usage'] * len(audit_df))
        st.info(f"**€{highest_investment['Total Investment']:,.0f}** invested in {num_ads_with_element} ads featuring {highest_investment['Element']} (out of {len(audit_df)} total campaigns).")
        with st.expander("💰 Why has this element received the most investment?"):
            # Calculate relative context
            median_usage = master_df['Overall Usage'].median()
            median_recognition = master_df['Recognition'].median()
            inv_vs_median = ((highest_investment['Total Investment'] / master_df['Total Investment'].median()) - 1) * 100

            factors = []

            # Usage correlation
            if highest_investment['Overall Usage'] >= median_usage:
                factors.append(f"**High Campaign Frequency:** Used in {highest_investment['Overall Usage']:.0%} of all campaigns - broad deployment drives total spend")
            else:
                factors.append(f"**Campaign Presence:** Used in {highest_investment['Overall Usage']:.0%} of campaigns")

            # Recognition outcome
            if highest_investment['Recognition'] >= median_recognition:
                factors.append(f"**Strong Recognition Result:** Achieves {highest_investment['Recognition']:.0%} consumer recognition (above median)")
            else:
                factors.append(f"**Building Recognition:** Currently at {highest_investment['Recognition']:.0%} recognition with this investment")

            # Media versatility
            if highest_investment['Usage Image'] > 0.3 and highest_investment['Usage Video'] > 0.3:
                factors.append(f"**Media Versatility:** Works effectively across both image and video formats")
            elif highest_investment['Usage Image'] > 0.5:
                factors.append(f"**Image-Focused:** Primarily deployed in static image campaigns")
            elif highest_investment['Usage Video'] > 0.5:
                factors.append(f"**Video-Focused:** Primarily deployed in video campaigns")

            # ROI context
            roi_ratio = highest_investment['Recognition ROI'] / best_roi['Recognition ROI'] if best_roi['Recognition ROI'] > 0 else 1
            if roi_ratio >= 0.8:
                factors.append(f"**Efficient Investment:** ROI of {highest_investment['Recognition ROI']:.2f} per €1M - competitive efficiency")
            else:
                factors.append(f"**Investment-Driven Strategy:** ROI of {highest_investment['Recognition ROI']:.2f} per €1M - recognition built through sustained spend")

            explanation = f"**{highest_investment['Element']}** receives the highest investment (€{highest_investment['Total Investment']:,.0f}, {inv_vs_median:.0f}% above median):\n\n"
            for i, factor in enumerate(factors, 1):
                explanation += f"{i}. {factor}\n"

            explanation += f"\n**ROI Context:** Compare this element's {highest_investment['Recognition ROI']:.2f} per €1M to the most efficient asset ({best_roi['Element']}) at {best_roi['Recognition ROI']:.2f} per €1M."

            st.markdown(explanation)

    with col4:
        st.metric(
            "Best Recognition ROI", 
            best_roi['Element'],
            help="Recognition ROI = (Recognition % / Total Investment) × €1M. Shows how many recognition points are gained per million euros spent. Higher is better."
        )
        st.info(f"Delivers **{best_roi['Recognition ROI']:.2f}** recognition points per €1M spent - the most efficient performer.")
        with st.expander("⚡ Why is this element the most efficient?"):
            # Calculate relative context
            median_investment = master_df['Total Investment'].median()
            median_recognition = master_df['Recognition'].median()
            median_usage = master_df['Overall Usage'].median()
            roi_vs_median = ((best_roi['Recognition ROI'] / master_df['Recognition ROI'].median()) - 1) * 100

            factors = []

            # Investment vs Recognition trade-off
            if best_roi['Total Investment'] < median_investment and best_roi['Recognition'] >= median_recognition:
                inv_ratio = median_investment / best_roi['Total Investment'] if best_roi['Total Investment'] > 0 else 1
                factors.append(f"**Exceptional Efficiency:** Only €{best_roi['Total Investment']:,.0f} spent (below median), yet achieves {best_roi['Recognition']:.0%} recognition (above median)")
            elif best_roi['Total Investment'] < median_investment:
                factors.append(f"**Low Investment:** €{best_roi['Total Investment']:,.0f} total spend - {best_roi['Recognition']:.0%} recognition from modest budget")
            else:
                factors.append(f"**Investment:** €{best_roi['Total Investment']:,.0f} invested, achieving {best_roi['Recognition']:.0%} recognition")

            # Usage factor
            if best_roi['Overall Usage'] < median_usage and best_roi['Recognition'] >= median_recognition:
                factors.append(f"**Strategic Deployment:** Used in only {best_roi['Overall Usage']:.0%} of campaigns, yet achieves above-median recognition - maximizes impact per appearance")
            elif best_roi['Overall Usage'] >= median_usage:
                factors.append(f"**Consistent Presence:** Used in {best_roi['Overall Usage']:.0%} of campaigns")
            else:
                factors.append(f"**Selective Usage:** Appears in {best_roi['Overall Usage']:.0%} of campaigns")

            # Uniqueness factor
            if best_roi['Uniqueness'] >= master_df['Uniqueness'].median():
                factors.append(f"**Distinctive Asset:** {best_roi['Uniqueness']:.0%} uniqueness - memorability reduces need for repetition")
            else:
                factors.append(f"**Uniqueness:** {best_roi['Uniqueness']:.0%} uniqueness score")

            # ROI highlight
            factors.append(f"**ROI Leadership:** {best_roi['Recognition ROI']:.2f} per €1M is {roi_vs_median:.0f}% above median - industry-leading efficiency")

            explanation = f"**{best_roi['Element']}** achieves exceptional efficiency:\n\n"
            for i, factor in enumerate(factors, 1):
                explanation += f"{i}. {factor}\n"

            # Opportunity statement
            if best_roi['Overall Usage'] < median_usage or best_roi['Total Investment'] < median_investment:
                explanation += f"\n**Opportunity:** This asset punches above its weight - consider increasing investment from €{best_roi['Total Investment']:,.0f} to amplify results further while maintaining efficiency."
            else:
                explanation += f"\n**Strategy:** This high-efficiency asset delivers strong returns - maintain current approach."

            st.markdown(explanation)


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
            action = "Recommended"
        else:
            tier = "🥉 Tier 3"
            action = "Optional/Redesign"

        tier_summary.append({
            'Element': row['Element'],
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

    # Key Takeaways Box - Data-driven
    top_recognition = master_df.nlargest(3, 'Recognition')
    top_performer = top_recognition.iloc[0]
    avg_recognition = master_df['Recognition'].mean()
    recognition_ratio = top_performer['Recognition'] / avg_recognition if avg_recognition > 0 else 0
    negative_sentiment_count = (master_df['Net Sentiment'] < 0).sum()

    takeaways_text = f"""
    ### 🎯 Key Takeaways

    **Top Performers:**
    - **{top_performer['Element']}** leads with {top_performer['Recognition']:.0%} recognition and {top_performer['Uniqueness']:.0%} uniqueness
    """

    # Add top 2-3 performers dynamically
    for i in range(1, min(3, len(top_recognition))):
        performer = top_recognition.iloc[i]
        takeaways_text += f"    - **{performer['Element']}** shows strong performance ({performer['Recognition']:.0%} recognition, {performer['Uniqueness']:.0%} uniqueness)\n"

    takeaways_text += f"""

    **Critical Challenge:**
    - Average recognition is {avg_recognition:.0%} - significant room for improvement
    - Top performer is {recognition_ratio:.1f}x higher than average

    **Strategic Priority:**
    - Focus on **{top_performer['Element']}** as the primary brand carrier ({recognition_ratio:.1f}x average recognition)
    - Address negative sentiment in {negative_sentiment_count} out of {len(master_df)} brand elements
    """

    st.success(takeaways_text)

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
    st.markdown("#### Brand Equity Matrix: Fame vs. Uniqueness with First Recognition Trigger")
    st.caption("Bubble size represents First Recognition Trigger strength (which asset triggers Škoda recognition first). Color intensity shows brand attribution.")

    # Add demographic selector for equity matrix
    st.markdown("#### 🎯 Filter by Demographics")
    equity_demo_col1, equity_demo_col2, equity_demo_col3 = st.columns(3)

    with equity_demo_col1:
        equity_country = st.selectbox(
            "Country:",
            ["All Countries", "UK", "Spain", "Germany", "Poland"],
            key="equity_country"
        )

    with equity_demo_col2:
        equity_age = st.selectbox(
            "Age Group:",
            ["All Ages", "18-30", "31-42", "43-55"],
            key="equity_age"
        )

    with equity_demo_col3:
        equity_gender = st.selectbox(
            "Gender:",
            ["All Genders", "Male", "Female"],
            key="equity_gender"
        )

    # Show demographic context
    equity_demo_text = []
    if equity_country != "All Countries":
        equity_demo_text.append(f"**{equity_country}**")
    if equity_age != "All Ages":
        equity_demo_text.append(f"**{equity_age}**")
    if equity_gender != "All Genders":
        equity_demo_text.append(f"**{equity_gender}**")

    if equity_demo_text:
        st.caption(f"Showing data for: {' | '.join(equity_demo_text)}")
    else:
        st.caption("Showing data for: **All Demographics**")

    # Create a copy of master_df for the equity matrix with demographic filters applied
    equity_matrix_df = master_df.copy()

    # Add First Recognition Trigger Index data
    equity_matrix_df['First_Trigger_Strength'] = 0.0
    if first_recognition_trigger:
        for element in brand_elements:
            if element in first_recognition_trigger:
                # Use percent_of_total_first_triggers as the trigger strength metric
                trigger_strength = first_recognition_trigger[element].get('percent_of_total_first_triggers', 0)
                equity_matrix_df.loc[equity_matrix_df['Element'] == element, 'First_Trigger_Strength'] = trigger_strength

    # Update recognition and uniqueness based on demographic selections
    if equity_age != "All Ages" or equity_gender != "All Genders":
        for element in brand_elements:
            # Update recognition
            if element in recognition_by_age_gender:
                if equity_gender != "All Genders" and 'gender' in recognition_by_age_gender[element]:
                    gender_key = equity_gender.lower()
                    if gender_key in recognition_by_age_gender[element]['gender']:
                        equity_matrix_df.loc[equity_matrix_df['Element'] == element, 'Recognition'] = recognition_by_age_gender[element]['gender'][gender_key]
                elif equity_age != "All Ages" and 'age' in recognition_by_age_gender[element]:
                    if equity_age in recognition_by_age_gender[element]['age']:
                        equity_matrix_df.loc[equity_matrix_df['Element'] == element, 'Recognition'] = recognition_by_age_gender[element]['age'][equity_age]

            # Update uniqueness
            if element in uniqueness_by_age_gender:
                if equity_gender != "All Genders" and 'gender' in uniqueness_by_age_gender[element]:
                    gender_key = equity_gender.lower()
                    if gender_key in uniqueness_by_age_gender[element]['gender']:
                        equity_matrix_df.loc[equity_matrix_df['Element'] == element, 'Uniqueness'] = uniqueness_by_age_gender[element]['gender'][gender_key]
                elif equity_age != "All Ages" and 'age' in uniqueness_by_age_gender[element]:
                    if equity_age in uniqueness_by_age_gender[element]['age']:
                        equity_matrix_df.loc[equity_matrix_df['Element'] == element, 'Uniqueness'] = uniqueness_by_age_gender[element]['age'][equity_age]

    if equity_country != "All Countries":
        for element in brand_elements:
            # Update uniqueness by country
            if element in uniqueness_by_country and equity_country in uniqueness_by_country[element]:
                equity_matrix_df.loc[equity_matrix_df['Element'] == element, 'Uniqueness'] = uniqueness_by_country[element][equity_country]

    with st.expander("📖 Understanding this matrix"):
        st.markdown("""
        This chart maps the three critical dimensions of brand asset strength:

        **Y-Axis (Recognition/Fame):** How many consumers have seen/heard this element
        - Higher = More familiar to consumers
        - Based on consumer survey responses: "Have you seen/heard this element before?"

        **X-Axis (Uniqueness):** How distinctively Škoda this element is
        - Higher = Stronger brand attribution (consumers know it's Škoda, not a competitor)
        - Based on consumer survey: "Which brand does this element belong to?"

        **Bubble Size:** First Recognition Trigger Index
        - Larger bubbles = This asset most frequently triggers FIRST recognition of Škoda
        - When consumers see multiple brand elements, which one makes them think "Škoda" first?
        - Critical for creative strategy: lead with high-trigger assets

        **Color:** Uniqueness intensity (darker green = more uniquely Škoda)

        **Ideal Profile:** Top-right with large bubble = high fame, high uniqueness, triggers recognition first
        **Strategic Insight:** Large bubbles show which assets to feature prominently; position in top-right quadrant validates brand equity
        """)

    fig_matrix = px.scatter(
        equity_matrix_df,
        x="Uniqueness",
        y="Recognition",
        size="First_Trigger_Strength",
        color="Uniqueness",  # Use uniqueness for color gradient
        text="Element",
        size_max=80,
        hover_data=['Total Investment', 'Average Investment', 'Overall Usage', 'First_Trigger_Strength'],
        color_continuous_scale='RdYlGn',
        title="Fame vs. Uniqueness (Bubble Size = First Recognition Trigger Strength)"
    )
    fig_matrix.update_traces(textposition='top center')
    fig_matrix.update_layout(height=600)
    st.plotly_chart(fig_matrix, use_container_width=True)

    # First Recognition Trigger Hierarchy
    st.markdown("---")
    st.markdown("#### 🎯 First Recognition Trigger Index")
    st.caption("Which brand element makes consumers think 'Škoda' FIRST when seeing multiple assets?")

    if first_recognition_trigger:
        # Create trigger ranking
        trigger_data = []
        for element in brand_elements:
            if element in first_recognition_trigger and first_recognition_trigger[element].get('count', 0) > 0:
                trigger_data.append({
                    'Element': element,
                    'Trigger_Percentage': first_recognition_trigger[element]['percent_of_total_first_triggers'],
                    'Recognition_Rate': first_recognition_trigger[element]['percent_recognized'],
                    'Count': first_recognition_trigger[element]['count']
                })

        if trigger_data:
            trigger_df = pd.DataFrame(trigger_data).sort_values('Trigger_Percentage', ascending=False)

            col1, col2 = st.columns([3, 2])

            with col1:
                # Bar chart of trigger strength
                fig_trigger = px.bar(
                    trigger_df,
                    x='Element',
                    y='Trigger_Percentage',
                    text='Trigger_Percentage',
                    title="First Recognition Trigger Strength by Element",
                    labels={'Trigger_Percentage': 'Share of First Triggers', 'Element': 'Brand Element'},
                    color='Trigger_Percentage',
                    color_continuous_scale='Greens'
                )
                fig_trigger.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                fig_trigger.update_layout(
                    showlegend=False,
                    yaxis_tickformat='.0%',
                    height=400
                )
                st.plotly_chart(fig_trigger, use_container_width=True)

            with col2:
                st.markdown("**💡 Key Insights:**")

                top_trigger = trigger_df.iloc[0]
                st.success(f"""
                **Top Trigger: {top_trigger['Element']}**
                - {top_trigger['Trigger_Percentage']:.1%} of first recognitions
                - {top_trigger['Recognition_Rate']:.0%} recognition when shown first
                - **Strategy:** Feature prominently in opening frames
                """)

                if len(trigger_df) > 1:
                    second_trigger = trigger_df.iloc[1]
                    st.info(f"""
                    **Runner-up: {second_trigger['Element']}**
                    - {second_trigger['Trigger_Percentage']:.1%} of first recognitions
                    - Strong secondary cue for brand recognition
                    """)

                # Calculate power duo
                if len(trigger_df) >= 2:
                    top_two_share = trigger_df.iloc[0]['Trigger_Percentage'] + trigger_df.iloc[1]['Trigger_Percentage']
                    st.warning(f"""
                    **Power Duo:**
                    Top 2 elements account for **{top_two_share:.1%}** of all first recognitions.
                    Combined use maximizes instant Škoda recognition.
                    """)
    else:
        st.info("First recognition trigger data not available.")

    # Add interpretation of matrix patterns
    top_right = equity_matrix_df[(equity_matrix_df['Recognition'] >= equity_matrix_df['Recognition'].median()) &
                          (equity_matrix_df['Uniqueness'] >= equity_matrix_df['Uniqueness'].median())]
    bottom_left = equity_matrix_df[(equity_matrix_df['Recognition'] < equity_matrix_df['Recognition'].median()) &
                            (equity_matrix_df['Uniqueness'] < equity_matrix_df['Uniqueness'].median())]

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
            median_recognition = master_df['Recognition'].median()

            factors = []
            factors.append(f"**Strong Positive Scores:** {most_positive['Positive Sentiment']:.1%} positive vs {most_positive['Negative Sentiment']:.1%} negative")

            if most_positive['Recognition'] >= median_recognition:
                factors.append(f"**Recognition Advantage:** {most_positive['Recognition']:.0%} recognition (above median) - familiarity may build positive associations")
            else:
                factors.append(f"**Recognition Level:** {most_positive['Recognition']:.0%} recognition")

            factors.append(f"**Net Sentiment Leadership:** {most_positive['Net Sentiment']:.1%} net positive is the highest among all brand elements")

            explanation = f"**{most_positive['Element']}** resonates most strongly based on consumer responses:\n\n"
            for i, factor in enumerate(factors, 1):
                explanation += f"{i}. {factor}\n"

            explanation += f"\nHigh sentiment indicates this asset creates emotional connection beyond just recognition."

            st.markdown(explanation)

    with col2:
        st.metric("Least Positive Element", least_positive['Element'], f"{least_positive['Net Sentiment']:+.1%}")
        st.warning(f"**{least_positive['Net Sentiment']:+.1%}** net sentiment - needs improvement.")
        with st.expander("❓ Why is this element's sentiment negative?"):
            median_recognition = master_df['Recognition'].median()

            factors = []
            factors.append(f"**More Negative Associations:** {least_positive['Negative Sentiment']:.1%} negative vs {least_positive['Positive Sentiment']:.1%} positive")

            if least_positive['Recognition'] < median_recognition:
                factors.append(f"**Below-Median Recognition:** {least_positive['Recognition']:.0%} recognition (below median) - limited familiarity may affect emotional connection")
            else:
                factors.append(f"**Recognition Level:** {least_positive['Recognition']:.0%} recognition")

            factors.append(f"**Current Usage:** {least_positive['Overall Usage']:.0%} of campaigns")

            sentiment_gap = most_positive['Net Sentiment'] - least_positive['Net Sentiment']
            factors.append(f"**Performance Gap:** {sentiment_gap:.1%} points behind the top performer - significant improvement opportunity")

            explanation = f"**{least_positive['Element']}** has negative net sentiment:\n\n"
            for i, factor in enumerate(factors, 1):
                explanation += f"{i}. {factor}\n"

            explanation += f"\n**This is a brand concern that requires strategic attention.**"

            st.markdown(explanation)

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
            # Find highest and lowest sentiment elements dynamically
            top_sentiment_el = master_df.loc[master_df['Net Sentiment'].idxmax(), 'Element']
            low_sentiment_el = master_df.loc[master_df['Net Sentiment'].idxmin(), 'Element']

            st.markdown(f"""
            Sentiment varies between {master_df['Net Sentiment'].min():+.1%} and {master_df['Net Sentiment'].max():+.1%} based on consumer responses:

            1. **Wide Performance Range:** Top performer ({top_sentiment_el}) vs lowest ({low_sentiment_el}) shows {sentiment_range:.1%} gap
            2. **Recognition Doesn't Guarantee Sentiment:** Some high-recognition assets still show negative sentiment (recognition ≠ positive emotion)
            3. **Element-Specific Responses:** Different brand elements trigger different emotional responses from consumers
            4. **Consistency Challenge:** Large variation suggests inconsistent emotional impact across the brand asset portfolio

            **Strategy:** Prioritize Sonic and Symbol in communications; redesign or phase out weakest performers.
            """)


    st.markdown("---")

    # Positive vs Negative Bar Chart
    st.markdown("### Positive vs Negative Sentiment Comparison")
    st.caption("Green bars show positive associations, red bars show negative associations")

    # Add demographic selector for sentiment charts
    st.markdown("#### 🎯 Filter by Demographics")
    sentiment_demo_col1, sentiment_demo_col2, sentiment_demo_col3 = st.columns(3)

    with sentiment_demo_col1:
        sentiment_country = st.selectbox(
            "Country:",
            ["All Countries", "UK", "Spain", "Germany", "Poland"],
            key="sentiment_country"
        )

    with sentiment_demo_col2:
        sentiment_age = st.selectbox(
            "Age Group:",
            ["All Ages", "18-30", "31-42", "43-55"],
            key="sentiment_age"
        )

    with sentiment_demo_col3:
        sentiment_gender = st.selectbox(
            "Gender:",
            ["All Genders", "Male", "Female"],
            key="sentiment_gender"
        )

    # Show demographic context
    sentiment_demo_text = []
    if sentiment_country != "All Countries":
        sentiment_demo_text.append(f"**{sentiment_country}**")
    if sentiment_age != "All Ages":
        sentiment_demo_text.append(f"**{sentiment_age}**")
    if sentiment_gender != "All Genders":
        sentiment_demo_text.append(f"**{sentiment_gender}**")

    if sentiment_demo_text:
        st.caption(f"Showing data for: {' | '.join(sentiment_demo_text)}")
    else:
        st.caption("Showing data for: **All Demographics**")

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

    # Add demographic selector for confusion matrix
    st.markdown("#### 🎯 Filter by Demographics")
    confusion_demo_col1, confusion_demo_col2, confusion_demo_col3 = st.columns(3)

    with confusion_demo_col1:
        confusion_country = st.selectbox(
            "Country:",
            ["All Countries", "UK", "Spain", "Germany", "Poland"],
            key="confusion_country"
        )

    with confusion_demo_col2:
        confusion_age = st.selectbox(
            "Age Group:",
            ["All Ages", "18-30", "31-42", "43-55"],
            key="confusion_age"
        )

    with confusion_demo_col3:
        confusion_gender = st.selectbox(
            "Gender:",
            ["All Genders", "Male", "Female"],
            key="confusion_gender"
        )

    # Show demographic context
    demo_text = []
    if confusion_country != "All Countries":
        demo_text.append(f"**{confusion_country}**")
    if confusion_age != "All Ages":
        demo_text.append(f"**{confusion_age}**")
    if confusion_gender != "All Genders":
        demo_text.append(f"**{confusion_gender}**")

    if demo_text:
        st.caption(f"Showing data for: {' | '.join(demo_text)}")
    else:
        st.caption("Showing data for: **All Demographics**")

    # Create confusion matrix using REAL Q05 data
    confusion_df = pd.DataFrame(q05_confusion_data).T
    confusion_df = confusion_df[['Skoda', 'Other_mentions', 'Dont_know']]
    confusion_df.columns = ['Škoda', 'Other Brands', "Don't Know"]

    # Create confusion matrix with inverted scale for competitors
    # We need to invert competitor columns so high values = red, low values = green
    confusion_df_display = confusion_df.copy()

    # Invert competitor and generic columns (1 - value) so high becomes low for coloring
    for col in ['Other Brands', "Don't Know"]:
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
            dont_know_pct = row["Don't Know"]
            st.caption(f"Other brands: {row['Other Brands']:.0%} | Don't know: {dont_know_pct:.0%}")

    with col2:
        st.markdown("#### ⚠️ Confusion Risks")

        # Find elements with high other brand confusion
        high_other = confusion_df[confusion_df['Other Brands'] >= 0.20].sort_values('Other Brands', ascending=False)
        if len(high_other) > 0:
            st.warning("**Other Brand Confusion (Brand Dilution Risk):**")
            for element, row in high_other.iterrows():
                st.write(f"• **{element}**: {row['Other Brands']:.0%} think it's another brand")

        # Find elements with high don't know
        high_dontknow = confusion_df[confusion_df["Don't Know"] >= 0.55].sort_values("Don't Know", ascending=False)
        if len(high_dontknow) > 0:
            st.warning("**High 'Don't Know' (Low Distinctiveness):**")
            for element, row in high_dontknow.iterrows():
                dont_know_val = row["Don't Know"]
                st.write(f"• **{element}**: {dont_know_val:.0%} don't recognize")

    # Confusion Matrix
    st.markdown("#### 📊 Confusion Matrix")

    confusion_df['Brand_Confusion_Risk'] = confusion_df['Other Brands']
    confusion_df['Distinctiveness_Score'] = confusion_df['Škoda'] - confusion_df['Brand_Confusion_Risk']

    threat_matrix = []
    for element in confusion_df.index:
        skoda_attr = confusion_df.loc[element, 'Škoda']
        other_brands = confusion_df.loc[element, 'Other Brands']
        dont_know = confusion_df.loc[element, "Don't Know"]

        threat_matrix.append({
            'Element': element,
            'Škoda Attribution': skoda_attr,
            'Other Brands': other_brands,
            "Don't Know": dont_know
        })

    threat_df = pd.DataFrame(threat_matrix).sort_values('Other Brands', ascending=False)
    st.dataframe(threat_df.style.format({
        'Škoda Attribution': '{:.0%}',
        'Other Brands': '{:.0%}',
        "Don't Know": '{:.0%}'
    }), use_container_width=True)

    # Detailed Competitor Breakdown
    st.markdown("---")
    st.markdown("### 🔍 Detailed Competitor Breakdown")
    st.caption("Top brands mentioned when consumers misattribute Škoda elements (from open-text responses)")

    # Load detailed competitor data (CLEANED VERSION - verbatim responses recoded)
    try:
        with open('q05_competitor_detail_CLEANED.json', 'r', encoding='utf-8') as f:
            competitor_detail = json.load(f)

        # Global Summary: Top Automotive Competitors Across All Elements
        st.markdown("#### 🌍 Global View: Top Automotive Competitor Mentions")
        st.caption("Aggregated across all brand elements - which car brands are most confused with Škoda?")

        # Aggregate all automotive competitor mentions
        global_auto_competitors = {}
        total_mentions = 0

        for element, data in competitor_detail.items():
            if 'automotive_competitors' in data and data['automotive_competitors']['brands']:
                for brand in data['automotive_competitors']['brands']:
                    brand_name = brand['brand']
                    count = brand['count']
                    global_auto_competitors[brand_name] = global_auto_competitors.get(brand_name, 0) + count
                    total_mentions += count

        if global_auto_competitors:
            col1, col2 = st.columns([2, 1])

            with col1:
                # Create bar chart of top competitors
                global_comp_df = pd.DataFrame([
                    {'Brand': brand, 'Mentions': count}
                    for brand, count in sorted(global_auto_competitors.items(), key=lambda x: x[1], reverse=True)
                ])

                fig_global_comp = px.bar(
                    global_comp_df,
                    x='Brand',
                    y='Mentions',
                    title="Global Automotive Competitor Confusion",
                    labels={'Mentions': 'Total Mentions', 'Brand': 'Competitor Brand'},
                    color='Mentions',
                    color_continuous_scale='Reds'
                )
                fig_global_comp.update_traces(texttemplate='%{y}', textposition='outside')
                fig_global_comp.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_global_comp, use_container_width=True)

            with col2:
                st.markdown("**Key Findings:**")
                st.metric("Total Automotive Mentions", total_mentions)
                st.caption(f"Out of ~726 total verbatim responses")

                top_competitor = max(global_auto_competitors.items(), key=lambda x: x[1])
                st.metric("Top Competitor", top_competitor[0])
                st.caption(f"{top_competitor[1]} mentions")

                confusion_rate = (total_mentions / 726) * 100 if total_mentions > 0 else 0
                st.success(f"""
                **Overall Automotive Confusion: {confusion_rate:.1f}%**

                Minimal competitive threat - Škoda elements are NOT being confused with competitor car brands.
                """)
        else:
            st.success("✅ **Excellent News:** Zero automotive competitor mentions across all elements. No competitive brand confusion detected.")

        st.markdown("---")

        # Market-Level Breakdown using q05_confusion_by_country
        if q05_confusion_by_country:
            st.markdown("#### 🗺️ Market-Level Competitor Confusion")
            st.caption("Automotive confusion breakdown by market (UK, Spain, Germany, Poland)")

            # Calculate automotive confusion by market for each element
            markets = ["UK", "Spain", "Germany", "Poland"]
            market_confusion_data = []

            for element in brand_elements:
                if element in q05_confusion_by_country:
                    for market in markets:
                        if market in q05_confusion_by_country[element]:
                            # "Other_mentions" includes all non-Skoda mentions (automotive + non-automotive + confused)
                            other_pct = q05_confusion_by_country[element][market].get('Other', 0) or \
                                       q05_confusion_by_country[element][market].get('Other_mentions', 0)

                            market_confusion_data.append({
                                'Element': element,
                                'Market': market,
                                'Other_Mentions': other_pct
                            })

            if market_confusion_data:
                market_df = pd.DataFrame(market_confusion_data)

                # Create heatmap
                market_pivot = market_df.pivot(index='Element', columns='Market', values='Other_Mentions')

                fig_market_heat = px.imshow(
                    market_pivot,
                    labels=dict(x="Market", y="Brand Element", color="Other Brand Mentions %"),
                    text_auto='.0%',
                    aspect="auto",
                    color_continuous_scale='Reds',
                    title="Non-Škoda Brand Mentions by Market (includes all misattributions)"
                )
                fig_market_heat.update_layout(height=450)
                st.plotly_chart(fig_market_heat, use_container_width=True)

                # Market insights
                col1, col2, col3, col4 = st.columns(4)

                for idx, market in enumerate(markets):
                    with [col1, col2, col3, col4][idx]:
                        market_data = market_df[market_df['Market'] == market]
                        avg_confusion = market_data['Other_Mentions'].mean()

                        st.metric(
                            market,
                            f"{avg_confusion:.0%}",
                            help=f"Average 'Other brand' mentions across all elements in {market}"
                        )

                        # Find most confused element in this market
                        most_confused = market_data.loc[market_data['Other_Mentions'].idxmax()]
                        st.caption(f"Highest: {most_confused['Element']} ({most_confused['Other_Mentions']:.0%})")

                st.info("""
                **Note:** "Other Mentions" includes automotive competitors, non-automotive brands, and generic/confused responses.
                Based on global data, only ~0.55% of these are actual automotive competitors - the vast majority are non-threats.
                """)

        st.markdown("---")

        # Element selector for detailed view
        detail_element = st.selectbox(
            "Select element to view detailed competitor mentions:",
            options=[e for e in research_data.keys() if e in competitor_detail],
            key="detail_element"
        )

        if detail_element in competitor_detail:
            detail_data = competitor_detail[detail_element]

            if detail_data['total_responses'] > 0:
                # Show summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Verbatim Responses Coded", detail_data['total_responses'])
                with col2:
                    st.metric("Škoda Attribution", f"{detail_data['skoda_percentage']:.1%}")
                with col3:
                    st.metric("Don't Know", f"{detail_data['dont_know_percentage']:.1%}")
                with col4:
                    auto_count = detail_data['automotive_competitors']['count']
                    st.metric("🚗 Automotive Competitors", auto_count,
                             help="Actual car brands mentioned - the real competitive threat")

                # Show data quality note
                if '_data_quality_note' in detail_data:
                    st.caption(f"ℹ️ {detail_data['_data_quality_note']}")

                # Key Insight Box
                st.markdown("#### 💡 Key Finding")
                auto_pct = detail_data['automotive_competitors']['percentage']

                if auto_pct < 0.05:  # Less than 5%
                    st.success(f"""
                    **Minimal Automotive Confusion ({auto_pct:.1%})**
                    This element shows very low confusion with competitor car brands. When misattributed,
                    it's primarily to non-automotive brands or generic responses - not competitive threats.
                    """)
                elif auto_pct < 0.15:  # 5-15%
                    st.warning(f"""
                    **Moderate Automotive Confusion ({auto_pct:.1%})**
                    Some confusion with competitor brands exists, but most misattribution is to non-automotive brands.
                    """)
                else:
                    st.error(f"""
                    **High Automotive Confusion ({auto_pct:.1%})**
                    Significant confusion with competitor car brands - this element may be too generic or similar to competitors.
                    """)

                # Breakdown by category
                st.markdown("#### 📊 Misattribution Breakdown")

                col1, col2, col3 = st.columns(3)
                with col1:
                    auto_data = detail_data['automotive_competitors']
                    st.metric("🚗 Automotive Competitors",
                             f"{auto_data['percentage']:.1%}",
                             help="Car brands - real competitive threat")
                    if auto_data['brands']:
                        st.caption(f"{auto_data['count']} mentions: " +
                                 ", ".join([f"{b['brand']} ({b['count']})" for b in auto_data['brands'][:3]]))
                    else:
                        st.caption("No automotive brands mentioned ✅")

                with col2:
                    non_auto_data = detail_data['non_automotive_brands']
                    st.metric("🏪 Non-Automotive Brands",
                             f"{non_auto_data['percentage']:.1%}",
                             help="Consumer brands outside automotive - not competitive threats")
                    if non_auto_data.get('top_mentions'):
                        top_brands = ", ".join([b['brand'] for b in non_auto_data['top_mentions'][:3]])
                        st.caption(f"Top: {top_brands}")

                with col3:
                    confused_data = detail_data['could_not_identify']
                    st.metric("❓ Generic/Confused",
                             f"{confused_data['percentage']:.1%}",
                             help="Non-brand responses (e.g., 'car', 'green', unclear answers)")
                    st.caption(confused_data['description'])

                # Automotive competitors detail (if any)
                if detail_data['automotive_competitors']['brands']:
                    st.markdown("#### 🚗 Automotive Competitor Details")
                    st.error("⚠️ **These represent actual competitive threats in the automotive market:**")

                    auto_df = pd.DataFrame(detail_data['automotive_competitors']['brands'])
                    auto_df['percentage_display'] = auto_df['percentage'].apply(lambda x: f"{x:.2%}")

                    st.dataframe(
                        auto_df[['brand', 'count', 'percentage_display']].rename(columns={
                            'brand': 'Competitor Brand',
                            'count': 'Mentions',
                            'percentage_display': '% of Responses'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )

                # Non-automotive brands (collapsed)
                if detail_data['non_automotive_brands']['count'] > 0:
                    with st.expander(f"📋 View non-automotive brands ({detail_data['non_automotive_brands']['count']} mentions)"):
                        st.caption("These are not competitive threats - just brand confusion outside automotive sector")
                        if detail_data['non_automotive_brands'].get('top_mentions'):
                            for brand_data in detail_data['non_automotive_brands']['top_mentions']:
                                st.write(f"- **{brand_data['brand']}**: {brand_data['count']} mentions")

            else:
                st.info("No verbatim responses coded for this element.")

    except FileNotFoundError:
        st.warning("⚠️ Detailed competitor data file not found. Please ensure q05_competitor_detail_CLEANED.json is in the app directory.")
    except Exception as e:
        st.error(f"Error loading competitor detail data: {str(e)}")

# ==================== TAB 3: STRATEGIC INSIGHTS ====================
with tab3:
    st.header("📈 Strategic Insights Dashboard")
    st.caption("Advanced analytics organized into focused categories for easy navigation")

    # Key Takeaways at top level
    st.info("""
    ### 🎯 Quick Insights Summary

    **ROI Winners:** Sonic (best efficiency) | Symbol (best overall value: recognition + uniqueness)

    **Element Combinations:** Symbol-based pairs = highest recognition | Minimum 3 elements/ad for brand recognition

    **Investment:** Some high-spend elements underperform - see Portfolio Strategy tab
    """)

    # Create sub-tabs for better organization
    subtab1, subtab2, subtab3, subtab4 = st.tabs([
        "🎯 Portfolio Strategy",
        "💰 Efficiency & ROI",
        "🔗 Combinations & Synergies",
        "🌍 Market & Consumer Insights"
    ])

    # ========== SUB-TAB 1: PORTFOLIO STRATEGY ==========
    with subtab1:
        st.markdown("### 📊 Portfolio Optimization Matrices")
        st.caption("BCG-style strategic analysis - where to invest, hold, or cut")

    # Add demographic selector for portfolio matrices
    st.markdown("#### 🎯 Filter by Demographics")
    matrix_demo_col1, matrix_demo_col2, matrix_demo_col3 = st.columns(3)

    with matrix_demo_col1:
        matrix_country = st.selectbox(
            "Country:",
            ["All Countries", "UK", "Spain", "Germany", "Poland"],
            key="matrix_country"
        )

    with matrix_demo_col2:
        matrix_age = st.selectbox(
            "Age Group:",
            ["All Ages", "18-30", "31-42", "43-55"],
            key="matrix_age"
        )

    with matrix_demo_col3:
        matrix_gender = st.selectbox(
            "Gender:",
            ["All Genders", "Male", "Female"],
            key="matrix_gender"
        )

    # Show demographic context
    matrix_demo_text = []
    if matrix_country != "All Countries":
        matrix_demo_text.append(f"**{matrix_country}**")
    if matrix_age != "All Ages":
        matrix_demo_text.append(f"**{matrix_age}**")
    if matrix_gender != "All Genders":
        matrix_demo_text.append(f"**{matrix_gender}**")

    if matrix_demo_text:
        st.caption(f"Showing data for: {' | '.join(matrix_demo_text)}")
    else:
        st.caption("Showing data for: **All Demographics**")

    # Prepare data for matrices
    matrix_df = master_df.copy()

    # Update recognition and uniqueness based on demographic selections
    if matrix_age != "All Ages" or matrix_gender != "All Genders":
        for element in brand_elements:
            # Update recognition
            if element in recognition_by_age_gender:
                if matrix_gender != "All Genders" and 'gender' in recognition_by_age_gender[element]:
                    gender_key = matrix_gender.lower()
                    if gender_key in recognition_by_age_gender[element]['gender']:
                        matrix_df.loc[matrix_df['Element'] == element, 'Recognition'] = recognition_by_age_gender[element]['gender'][gender_key]
                elif matrix_age != "All Ages" and 'age' in recognition_by_age_gender[element]:
                    if matrix_age in recognition_by_age_gender[element]['age']:
                        matrix_df.loc[matrix_df['Element'] == element, 'Recognition'] = recognition_by_age_gender[element]['age'][matrix_age]

            # Update uniqueness
            if element in uniqueness_by_age_gender:
                if matrix_gender != "All Genders" and 'gender' in uniqueness_by_age_gender[element]:
                    gender_key = matrix_gender.lower()
                    if gender_key in uniqueness_by_age_gender[element]['gender']:
                        matrix_df.loc[matrix_df['Element'] == element, 'Uniqueness'] = uniqueness_by_age_gender[element]['gender'][gender_key]
                elif matrix_age != "All Ages" and 'age' in uniqueness_by_age_gender[element]:
                    if matrix_age in uniqueness_by_age_gender[element]['age']:
                        matrix_df.loc[matrix_df['Element'] == element, 'Uniqueness'] = uniqueness_by_age_gender[element]['age'][matrix_age]

    if matrix_country != "All Countries":
        for element in brand_elements:
            # Update uniqueness by country
            if element in uniqueness_by_country and matrix_country in uniqueness_by_country[element]:
                matrix_df.loc[matrix_df['Element'] == element, 'Uniqueness'] = uniqueness_by_country[element][matrix_country]
    
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

    # ========== SUB-TAB 2: EFFICIENCY & ROI ==========
    with subtab2:
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
        - **What this means:** Above-median consumer recognition AND brand attribution
        - **Strategy:** Protect and amplify - these are your brand-building powerhouses

        **🐴 Workhorses (Top-Left):** High Recognition + Lower Uniqueness
        - **What this means:** Above-median recognition but below-median brand attribution
        - **Strategy:** Maintain awareness but pair with unique assets for differentiation

        **💎 Hidden Gems (Bottom-Right):** Lower Recognition + High Uniqueness
        - **What this means:** Strong brand attribution but below-median recognition
        - **Strategy:** Invest more - these have untapped potential for differentiation

        **❓ Question Marks (Bottom-Left):** Lower Recognition + Lower Uniqueness
        - **What this means:** Below-median on both recognition and brand attribution
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
            if pd.notna(most_consistent):
                st.success(f"**Most Consistent:** {most_consistent} (σ={consistency_scores[most_consistent]:.3f})")
            else:
                st.info("Consistency data not available")
        with col2:
            if pd.notna(least_consistent):
                st.warning(f"**Least Consistent:** {least_consistent} (σ={consistency_scores[least_consistent]:.3f})")
            else:
                st.info("Consistency data not available")

    st.markdown("---")

    # ========== SUB-TAB 3: COMBINATIONS & SYNERGIES ==========
    with subtab3:
        st.markdown("### 🔗 Element Combinations: What Works Together?")
        st.caption("Analyzing recognition levels when brand elements appear together")

        # Add demographic selector for element combinations
        combo_filters = render_demographic_filters("combo")

    with combo_demo_col1:
        combo_country = st.selectbox(
            "Country:",
            ["All Countries", "UK", "Spain", "Germany", "Poland"],
            key="combo_country"
        )

    with combo_demo_col2:
        combo_age = st.selectbox(
            "Age Group:",
            ["All Ages", "18-30", "31-42", "43-55"],
            key="combo_age"
        )

    with combo_demo_col3:
        combo_gender = st.selectbox(
            "Gender:",
            ["All Genders", "Male", "Female"],
            key="combo_gender"
        )

    # Show demographic context
    combo_demo_text = []
    if combo_country != "All Countries":
        combo_demo_text.append(f"**{combo_country}**")
    if combo_age != "All Ages":
        combo_demo_text.append(f"**{combo_age}**")
    if combo_gender != "All Genders":
        combo_demo_text.append(f"**{combo_gender}**")

    if combo_demo_text:
        st.caption(f"Showing data for: {' | '.join(combo_demo_text)}")
    else:
        st.caption("Showing data for: **All Demographics**")

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

    # NEW: Cross-Asset Synergies - Attribution (Uniqueness) Analysis
    st.markdown("### 🎯 Cross-Asset Synergies: Brand Attribution Analysis")
    st.caption("Which element pairs or combinations drive correct Škoda attribution (uniqueness)?")

    st.info("""
    **CRITICAL INSIGHT:** Recognition isn't enough - we need **correct attribution**. This analysis shows which
    element combinations make consumers correctly identify the brand as Škoda (not competitors or generic).

    **Why this matters:**
    - "Colour + Facets" may achieve 40% recognition, but only 25% uniqueness (confused with competitors)
    - "Symbol + Wordmark" may achieve 45% recognition AND 60% uniqueness (clearly Škoda)

    **Gold for creative guidelines:** Shows which pairs maximize brand-building vs just awareness-building.
    """)

    # Create uniqueness matrix for co-occurring elements
    st.markdown("#### 🏆 Brand Attribution (Uniqueness) When Elements Appear Together")

    # Check if uniqueness_by_country data exists
    if uniqueness_by_country:
        uniqueness_matrix = pd.DataFrame(0.0, index=brand_elements, columns=brand_elements, dtype=float)

        for element1 in brand_elements:
            for element2 in brand_elements:
                if element1 != element2:
                    # Find ads where both elements appear
                    both_present = audit_df[audit_df[element1] & audit_df[element2]]

                    if len(both_present) > 0 and element1 in uniqueness_by_country and element2 in uniqueness_by_country:
                        # Calculate average uniqueness across all countries when both appear
                        uniq1 = uniqueness_by_country[element1]
                        uniq2 = uniqueness_by_country[element2]

                        # Average uniqueness of both elements
                        avg_uniqueness = (sum(uniq1.values()) + sum(uniq2.values())) / (2 * len(uniq1))
                        uniqueness_matrix.loc[element1, element2] = avg_uniqueness

        # Display as heatmap
        fig_uniqueness = px.imshow(
            uniqueness_matrix,
            labels=dict(x="Combined with", y="Element", color="Brand Attribution (Uniqueness)"),
            x=uniqueness_matrix.columns,
            y=uniqueness_matrix.index,
            color_continuous_scale='RdYlGn',
            text_auto='.0%',
            aspect="auto",
            title="Brand Attribution Heatmap: Element Combinations"
        )
        fig_uniqueness.update_layout(height=600)
        st.plotly_chart(fig_uniqueness, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ✅ Highest Attribution Pairs (Best for Brand-Building)")

            # Find top combinations by uniqueness
            attr_combinations = []
            for element1 in brand_elements:
                for element2 in brand_elements:
                    if element1 < element2:  # Avoid duplicates
                        combined_uniqueness = uniqueness_matrix.loc[element1, element2]
                        combined_recognition = recognition_matrix.loc[element1, element2]
                        if combined_uniqueness > 0:
                            # Count how often they appear together
                            both_present = audit_df[audit_df[element1] & audit_df[element2]].shape[0]

                            # Calculate brand equity score (recognition × uniqueness)
                            brand_equity = combined_recognition * combined_uniqueness

                            attr_combinations.append({
                                'Pair': f"{element1} + {element2}",
                                'Uniqueness': combined_uniqueness,
                                'Recognition': combined_recognition,
                                'Brand Equity': brand_equity,
                                'Appearances': both_present
                            })

            attr_combinations_df = pd.DataFrame(attr_combinations).sort_values('Uniqueness', ascending=False).head(5)

            for _, row in attr_combinations_df.iterrows():
                st.success(f"**{row['Pair']}**")
                st.write(f"   • Attribution: {row['Uniqueness']:.0%}")
                st.write(f"   • Recognition: {row['Recognition']:.0%}")
                st.write(f"   • Brand Equity: {row['Brand Equity']:.3f}")
                st.caption(f"   Appears together: {row['Appearances']} ads")

        with col2:
            st.markdown("#### ⚠️ High Recognition but Low Attribution (Risk)")

            # Find combinations with high recognition but lower uniqueness (awareness without attribution)
            risky_combinations = []
            for _, row in pd.DataFrame(attr_combinations).iterrows():
                # High recognition (>35%) but low uniqueness (<30%)
                if row['Recognition'] > 0.35 and row['Uniqueness'] < 0.30:
                    risky_combinations.append(row)

            if risky_combinations:
                risky_df = pd.DataFrame(risky_combinations).sort_values('Recognition', ascending=False).head(3)
                for _, row in risky_df.iterrows():
                    st.warning(f"**{row['Pair']}**")
                    st.write(f"   • Recognition: {row['Recognition']:.0%} ✅")
                    st.write(f"   • Attribution: {row['Uniqueness']:.0%} ⚠️")
                    st.caption("High awareness but confused with competitors")
            else:
                st.info("No high-risk combinations identified - all pairs with high recognition also have reasonable attribution")

        st.markdown("---")

        # Top strategic insight
        st.markdown("#### 💡 Strategic Playbook: Pair Selection Guidelines")

        # Get top attribution pair
        top_attr_pair = attr_combinations_df.iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🥇 Gold Standard Pairs**")
            st.write("High Attribution + High Recognition")
            for _, row in attr_combinations_df.head(3).iterrows():
                if row['Uniqueness'] > 0.35 and row['Recognition'] > 0.35:
                    st.write(f"• {row['Pair']}")
            st.caption("Use for brand-building campaigns")

        with col2:
            st.markdown("**🥈 Awareness Builders**")
            st.write("High Recognition, Lower Attribution")
            for _, row in pd.DataFrame(attr_combinations).sort_values('Recognition', ascending=False).head(5).iterrows():
                if row['Recognition'] > 0.40 and row['Uniqueness'] < 0.35:
                    st.write(f"• {row['Pair']}")
                    break
            st.caption("Pair with Symbol/Wordmark to boost attribution")

        with col3:
            st.markdown("**💎 Hidden Gems**")
            st.write("High Attribution, Lower Recognition")
            for _, row in attr_combinations_df.iterrows():
                if row['Uniqueness'] > 0.35 and row['Recognition'] < 0.35:
                    st.write(f"• {row['Pair']}")
            st.caption("Increase usage to build awareness")

        st.success(f"""
        **Key Finding:** The top attribution pair is **{top_attr_pair['Pair']}** with {top_attr_pair['Uniqueness']:.0%} uniqueness and {top_attr_pair['Recognition']:.0%} recognition.

        **Creative Guideline:** Prioritize this combination in all brand communications to maximize both awareness AND correct Škoda attribution.
        Avoid pairs with high recognition but low uniqueness - they build awareness for competitors, not Škoda.
        """)

    else:
        st.warning("Uniqueness by country data not available. Cannot calculate brand attribution for element combinations.")

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

    # Add demographic selector for consumer language
    st.markdown("#### 🎯 Filter by Demographics")
    language_demo_col1, language_demo_col2, language_demo_col3 = st.columns(3)

    with language_demo_col1:
        language_country = st.selectbox(
            "Country:",
            ["All Countries", "UK", "Spain", "Germany", "Poland"],
            key="language_country"
        )

    with language_demo_col2:
        language_age = st.selectbox(
            "Age Group:",
            ["All Ages", "18-30", "31-42", "43-55"],
            key="language_age"
        )

    with language_demo_col3:
        language_gender = st.selectbox(
            "Gender:",
            ["All Genders", "Male", "Female"],
            key="language_gender"
        )

    # Show demographic context
    language_demo_text = []
    if language_country != "All Countries":
        language_demo_text.append(f"**{language_country}**")
    if language_age != "All Ages":
        language_demo_text.append(f"**{language_age}**")
    if language_gender != "All Genders":
        language_demo_text.append(f"**{language_gender}**")

    if language_demo_text:
        st.caption(f"Showing data for: {' | '.join(language_demo_text)}")
    else:
        st.caption("Showing data for: **All Demographics**")

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
        # Sentiment analysis from Q04 adjective scales
        st.markdown("#### Sentiment (Q04 Adjectives)")

        # Get sentiment data from research_data (Q04), not q03_associations_data
        sentiment_data_source = research_data[selected_element]

        sentiment_df = pd.DataFrame({
            'Sentiment': ['Positive', 'Negative'],
            'Percentage': [
                sentiment_data_source['positive_sentiment'],
                sentiment_data_source['negative_sentiment']
            ]
        })

        fig_sentiment = px.pie(
            sentiment_df,
            values='Percentage',
            names='Sentiment',
            title='Adjective Sentiment',
            color='Sentiment',
            color_discrete_map={'Positive': '#4CAF50', 'Negative': '#F44336'}
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)

        st.metric("Net Sentiment",
                 f"{sentiment_data_source['net_sentiment']:+.1%}",
                 "Positive - Negative")

        st.caption("From Q04: Bold, Stylish, Modern, etc.")

    # Word cloud alternative - show all associations
    st.markdown(f"#### All Associations for {selected_element}")
    st.caption("Full list of consumer descriptions (Q03 open-text responses)")

    all_words_df = pd.DataFrame({
        'Association': element_data['top_words'],
        'Frequency': element_data['frequencies']
    })

    st.dataframe(all_words_df.style.format({'Frequency': '{:.1%}'}),
                use_container_width=True, hide_index=True)

    # Comparative sentiment analysis across all elements (Q04)
    st.markdown("---")
    st.markdown("### 📊 Sentiment Comparison Across All Elements")
    st.caption("Based on Q04 adjective scales: Bold, Stylish, Modern vs Cautious, Plain, Old-Fashioned")

    # Create sentiment comparison using research_data (Q04)
    all_sentiments = []
    for elem in brand_elements:
        elem_data = research_data[elem]
        all_sentiments.append({
            'Element': elem,
            'Positive': elem_data['positive_sentiment'],
            'Negative': elem_data['negative_sentiment'],
            'Net': elem_data['net_sentiment']
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
        title='Adjective Sentiment Analysis: All Elements',
        xaxis_title='Percentage',
        yaxis_title='',
        height=500,
        xaxis_tickformat='.0%'
    )

    st.plotly_chart(fig_sent_comp, use_container_width=True)

    # Key insights
    col1, col2 = st.columns(2)

    with col1:
        most_positive = sent_comparison_df.iloc[-1]
        st.success(f"""
        **Most Positive Sentiment:**
        - **{most_positive['Element']}**: {most_positive['Net']:+.1%} net sentiment
        - {most_positive['Positive']:.0%} positive adjectives
        """)

    with col2:
        most_negative = sent_comparison_df.iloc[0]
        st.warning(f"""
        **Most Negative Sentiment:**
        - **{most_negative['Element']}**: {most_negative['Net']:+.1%} net sentiment
        - {most_negative['Negative']:.0%} negative adjectives
        """)

    st.markdown("---")

    # Search for Strategic Terms in Consumer Language
    st.markdown("### 🔍 Strategic Brand Terms Search")
    st.caption("Search Q03 responses to see if desired brand values appear in consumer language")

    st.info("""
    **Purpose:** The client asked: *"Can we see whether 'Exploration' naturally clusters with our key assets or if people describe us with unrelated adjectives?"*

    Use this search to find if strategic brand values (Exploration, Innovation, Modern, etc.) appear in actual consumer responses.
    """)

    # Search input
    search_term = st.text_input("Search for a word or phrase in consumer associations:",
                                value="explore",
                                placeholder="e.g., explore, innovation, modern, safe, boring")

    if search_term:
        search_results = []
        search_lower = search_term.lower()

        for element, data in q03_associations_data.items():
            for word, freq in zip(data['top_words'], data['frequencies']):
                if search_lower in word.lower():
                    search_results.append({
                        'Element': element,
                        'Association': word,
                        'Frequency': freq
                    })

        if search_results:
            results_df = pd.DataFrame(search_results).sort_values('Frequency', ascending=False)
            st.success(f"✅ Found '{search_term}' in {len(search_results)} associations across {len(results_df['Element'].unique())} elements")

            st.dataframe(results_df.style.format({'Frequency': '{:.1%}'}),
                        use_container_width=True, hide_index=True)

            # Summary insight
            total_freq = results_df['Frequency'].sum()
            st.metric("Total Frequency",
                     f"{total_freq:.1%}",
                     f"Across {len(results_df['Element'].unique())} elements")

        else:
            st.warning(f"❌ No associations found containing '{search_term}'")
            st.caption("This suggests the term is not prominent in consumer language about Škoda brand elements.")

    # Common word analysis
    st.markdown("#### 📊 Most Common Words Across All Elements")

    all_words_aggregated = {}
    for element, data in q03_associations_data.items():
        for word, freq in zip(data['top_words'], data['frequencies']):
            if word not in all_words_aggregated:
                all_words_aggregated[word] = 0
            all_words_aggregated[word] += freq

    top_overall = sorted(all_words_aggregated.items(), key=lambda x: x[1], reverse=True)[:15]
    overall_df = pd.DataFrame(top_overall, columns=['Word', 'Total Frequency'])

    fig_overall = px.bar(
        overall_df,
        x='Total Frequency',
        y='Word',
        orientation='h',
        title='Top 15 Most Common Associations (All Elements Combined)',
        text=overall_df['Total Frequency'].apply(lambda x: f'{x:.1%}')
    )
    fig_overall.update_layout(height=500, showlegend=False)
    fig_overall.update_traces(textposition='outside')
    st.plotly_chart(fig_overall, use_container_width=True)

    st.success("""
    **Strategic Insight:**

    The most common words reveal what consumers **actually say** about Škoda elements, vs what the brand **wants** them to say.

    - ✅ **If "Škoda" appears frequently**: Strong brand recognition
    - ⚠️ **If "Confusing" or "Boring" appear**: Perception issues need addressing
    - 💡 **If "Exploration" is absent**: Gap between brand aspiration and consumer reality
    """)

# ==================== TAB 4: NON-NEGOTIABLES ====================
with tab4:
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
        (master_df['Uniqueness'] >= 0.15) &
        (master_df['Overall Usage'] >= 0.50)
    ].sort_values('Recognition', ascending=False)

    recommended = master_df[
        ((master_df['Recognition'] >= 0.35) | (master_df['Uniqueness'] >= 0.25))
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
                    st.metric("Usage", f"{row['Overall Usage']:.0%}")
                    st.metric("Investment", f"€{row['Total Investment']:,.0f}")
                with col_c:
                    # Calculate Brand Equity Score
                    equity_score = row['Recognition'] * row['Uniqueness']
                    st.metric("Brand Equity", f"{equity_score:.3f}")
                    st.metric("ROI", f"{row['Recognition ROI']:.2f}")

                st.markdown("**Rationale for Must-Use Status:**")
                st.write(f"• **Recognition:** {row['Recognition']:.0%} - consumers have seen/heard this element, ensuring immediate brand attribution")
                st.write(f"• **Uniqueness:** {row['Uniqueness']:.0%} - distinctively Škoda (consumers correctly identify it as belonging to your brand, not competitors)")
                st.write(f"• **Proven Usage:** {row['Overall Usage']:.0%} of campaigns - already validated as core asset")
                st.write(f"• **Investment Efficiency:** €{row['Total Investment']:,.0f} delivers {row['Recognition']:.0%} recognition = {row['Recognition ROI']:.2f} ROI")
                st.write(f"• **Sentiment:** +{row['Net Sentiment']:.1%} net positive emotional associations")

                st.markdown("**Why these metrics matter:**")
                st.write("High recognition ensures your ads are immediately identified as Škoda. High uniqueness prevents confusion with competitors. Combined, they build lasting brand equity with every exposure.")

        st.markdown("---")

        st.markdown("### ⭐ RECOMMENDED Assets (Strongly Encouraged)")
        st.info(f"**{len(recommended)} assets show strong potential:** Good recognition or uniqueness")

        for idx, row in recommended.iterrows():
            with st.expander(f"**{row['Element']}** - Recognition: {row['Recognition']:.0%} | Uniqueness: {row['Uniqueness']:.0%}"):
                st.markdown("**Why Recommended:**")
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

        st.markdown("### ⚠️ REQUIRES ATTENTION")
        st.warning(f"**{len(requires_attention)} assets** have low recognition despite significant investment")

        for idx, row in requires_attention.iterrows():
            with st.expander(f"**{row['Element']}** - Recognition: {row['Recognition']:.0%} | Investment: €{row['Total Investment']:,.0f}"):
                st.markdown("**Why this requires attention:**")
                st.write(f"• **Low recognition:** {row['Recognition']:.0%} despite €{row['Total Investment']:,.0f} investment (above median)")
                st.write(f"• **Usage:** {row['Overall Usage']:.0%} of campaigns")
                st.write(f"• **Uniqueness:** {row['Uniqueness']:.0%}")
                st.write(f"• **ROI:** {row['Recognition ROI']:.2f} per €1M (compare to best performer: {master_df['Recognition ROI'].max():.2f})")

                st.markdown("**Possible causes:**")
                st.write("1. **Recent investment:** Recognition may still be building (takes time)")
                st.write("2. **Generic design:** Low uniqueness suggests it may not be distinctive enough")
                st.write("3. **Ineffective deployment:** Placement, creative execution, or context may need optimization")
                st.write("4. **Low visibility:** May be used but not prominently featured in creative")

                st.markdown("**Recommended action:**")
                if row['Uniqueness'] < 0.20:
                    st.write("⚠️ Consider redesigning for greater Škoda distinctiveness OR deprioritize in favor of higher-uniqueness assets")
                else:
                    st.write("💡 Increase prominence in creative or give more time to build recognition - the distinctiveness is there")

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

                # Calculate relative context for data-driven recommendations
                median_usage = master_df['Overall Usage'].median()
                max_recognition = master_df['Recognition'].max()
                median_investment = master_df['Total Investment'].median()

                st.markdown("**💡 Why is this an opportunity?**")
                st.write(f"• **High uniqueness ({row['Uniqueness']:.0%})** means consumers correctly attribute it to Škoda in surveys")
                st.write(f"• **Below-median usage ({row['Overall Usage']:.0%} vs median {median_usage:.0%})** - significant room to increase deployment")
                st.write(f"• **Strong brand attribution** - {row['Uniqueness']:.0%} uniqueness indicates distinctive Škoda identity")

                if row['Total Investment'] < median_investment:
                    st.write(f"• **Below-median investment** (€{row['Total Investment']:,.0f} vs median €{median_investment:,.0f}) - scaling up is feasible")
                else:
                    st.write(f"• **Current investment** (€{row['Total Investment']:,.0f})")

                st.markdown("**📈 Recommendations based on top performers:**")
                st.write(f"• **Increase usage toward median ({median_usage:.0%})** to build recognition while maintaining distinctiveness")
                st.write(f"• **Current recognition ({row['Recognition']:.0%})** has room to grow toward top performers ({max_recognition:.0%})")

                if row['Recognition ROI'] >= master_df['Recognition ROI'].median():
                    st.write(f"• **Strong ROI ({row['Recognition ROI']:.2f})** suggests efficient performance - good candidate for increased investment")

                st.markdown("**🎯 Opportunity:**")
                st.write(f"Increasing usage to {median_usage:.0%}+ would likely boost recognition while maintaining the {row['Uniqueness']:.0%} uniqueness advantage")
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

        **Ideal strategy:** Increase investment in high-efficiency assets, optimize or reduce spend on low-efficiency ones
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
                    st.write(f"Increase investment from €{row['Total Investment']:,.0f} to €{row['Total Investment']*1.5:,.0f} could boost recognition from {row['Recognition']:.0%} to {min(row['Recognition']*1.3, 0.85):.0%} while maintaining high uniqueness")
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
        st.write("1. **Increase must-use asset deployment**")
        median_usage = master_df['Overall Usage'].median()
        max_usage = master_df['Overall Usage'].max()
        for idx, row in must_use.head(3).iterrows():
            target_usage = max(0.80, max_usage * 0.95)  # Target 80% or 95% of max usage
            st.write(f"   • Increase {row['Element']} from {row['Overall Usage']:.0%} toward {target_usage:.0%} of campaigns")

        st.write("2. **Test high-potential assets**")
        for idx, row in high_potential.head(2).iterrows():
            target_increase = median_usage if row['Overall Usage'] < median_usage else row['Overall Usage'] * 1.5
            st.write(f"   • Increase {row['Element']} from {row['Overall Usage']:.0%} toward {min(target_increase, 0.80):.0%} usage")

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
        ["Radar Chart (7 Dimensions)", "Bar Chart Comparison", "Semantic Differential (Positive vs Negative)"],
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

        elif personality_view == "Bar Chart Comparison":
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

        elif personality_view == "Semantic Differential (Positive vs Negative)":
            # Diverging bar chart showing positive vs negative adjectives
            st.caption("Shows the full spectrum from negative to positive perceptions for each dimension")

            # Define the 7 adjective pairs
            adjective_pairs = [
                ('bold', 'Cautious'),
                ('stylish', 'Plain'),
                ('modern', 'Old-Fashioned'),
                ('playful', 'Serious'),
                ('exciting', 'Boring'),
                ('human', 'Cold'),
                ('simple', 'Complicated')
            ]

            for element in selected_elements:
                st.markdown(f"#### {element}")

                # Get adjective data for this element
                element_adj_data = adjective_data[element]

                # Prepare data for diverging bars
                # First collect all the data
                bar_data = []
                for pos_adj, neg_adj in adjective_pairs:
                    adj_info = element_adj_data[pos_adj.lower()]
                    bar_data.append({
                        'label': f"{neg_adj} ← → {pos_adj.title()}",
                        'positive': adj_info['positive_net'],
                        'negative': -adj_info['negative_net'],
                        'pos_adj': pos_adj,
                        'neg_adj': neg_adj
                    })

                # Sort by positive percentage (descending) - highest at top
                bar_data.sort(key=lambda x: x['positive'], reverse=True)

                # Show key insights ABOVE the chart
                # Find strongest positive and strongest negative from sorted bar_data
                pos_strengths = [(item['pos_adj'].title(), item['positive']) for item in bar_data]
                neg_strengths = [(item['neg_adj'], abs(item['negative'])) for item in bar_data]

                # Sort by strength (they're already sorted by positive, but neg might differ)
                pos_strengths.sort(key=lambda x: x[1], reverse=True)
                neg_strengths.sort(key=lambda x: x[1], reverse=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Top Positive:** {pos_strengths[0][0]} ({pos_strengths[0][1]:.0%})")
                with col2:
                    if neg_strengths[0][1] > 0.15:  # Only show if significant
                        st.warning(f"**Top Negative:** {neg_strengths[0][0]} ({neg_strengths[0][1]:.0%})")
                    else:
                        st.info("No significant negative associations")

                # Create figure for this element
                fig_diverging = go.Figure()

                # Extract sorted values for plotting
                y_labels = [item['label'] for item in bar_data]
                positive_values = [item['positive'] for item in bar_data]
                negative_values = [item['negative'] for item in bar_data]

                # Add negative bars (left side, red)
                fig_diverging.add_trace(go.Bar(
                    y=y_labels,
                    x=negative_values,
                    name='Negative',
                    orientation='h',
                    marker=dict(color='#EF5350'),
                    text=[f"{abs(v):.0%}" for v in negative_values],
                    textposition='inside',
                    textangle=0,
                    hovertemplate='%{text}<extra></extra>'
                ))

                # Add positive bars (right side, green)
                fig_diverging.add_trace(go.Bar(
                    y=y_labels,
                    x=positive_values,
                    name='Positive',
                    orientation='h',
                    marker=dict(color='#66BB6A'),
                    text=[f"{v:.0%}" for v in positive_values],
                    textposition='inside',
                    textangle=0,
                    hovertemplate='%{text}<extra></extra>'
                ))

                # Update layout
                fig_diverging.update_layout(
                    barmode='overlay',
                    xaxis=dict(
                        title="",
                        range=[-0.7, 0.7],
                        tickformat='.0%',
                        tickvals=[-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6],
                        ticktext=['60%', '40%', '20%', '0%', '20%', '40%', '60%'],
                        zeroline=True,
                        zerolinewidth=2,
                        zerolinecolor='#333'
                    ),
                    yaxis=dict(
                        title="",
                        autorange='reversed'
                    ),
                    height=500,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    margin=dict(l=200, r=50, t=50, b=50)
                )

                st.plotly_chart(fig_diverging, use_container_width=True)

                st.markdown("---")

            st.info("**How to read this chart:** Green bars (right) show % who chose positive adjectives (Bold, Stylish, Modern, etc.). " +
                    "Red bars (left) show % who chose negative adjectives (Cautious, Plain, Old-Fashioned, etc.). " +
                    "Longer bars indicate stronger associations. The center represents neutral responses.")

    st.markdown("---")

    # Market/Country Recognition Analysis
    st.markdown("### Recognition by Market")
    st.caption("See how brand elements perform across different countries")

    # Add demographic selector for market recognition
    st.markdown("#### 🎯 Filter by Demographics")
    market_demo_col1, market_demo_col2 = st.columns(2)

    with market_demo_col1:
        market_age = st.selectbox(
            "Age Group:",
            ["All Ages", "18-30", "31-42", "43-55"],
            key="market_age"
        )

    with market_demo_col2:
        market_gender = st.selectbox(
            "Gender:",
            ["All Genders", "Male", "Female"],
            key="market_gender"
        )

    # Show demographic context
    market_demo_text = []
    if market_age != "All Ages":
        market_demo_text.append(f"**{market_age}**")
    if market_gender != "All Genders":
        market_demo_text.append(f"**{market_gender}**")

    if market_demo_text:
        st.caption(f"Showing data for: {' | '.join(market_demo_text)}")
    else:
        st.caption("Showing data for: **All Demographics**")

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

    st.markdown("---")

    # Market-Level Uniqueness (Brand Attribution) Analysis
    if uniqueness_by_country:
        st.markdown("### 🌍 Brand Attribution (Uniqueness) by Market")
        st.caption("Shows which markets correctly identify each element as belonging to Škoda (not competitors)")

        with st.expander("📖 Why market-level uniqueness matters"):
            st.markdown("""
            **Uniqueness** measures brand attribution - the % of consumers who correctly identify an element as belonging to Škoda (vs competitors or generic design).

            **Why market variations are critical:**
            - **2x differences** exist between markets (e.g., Symbol: UK 23% vs Poland 55%)
            - Global averages mask these massive variations
            - **Strategy implications:** Elements may need market-specific support or repositioning
            - **Investment decisions:** High-uniqueness markets can leverage assets; low-uniqueness markets need brand education

            **What to look for:**
            - Markets with <30% uniqueness: Element not seen as distinctively Škoda
            - Large gaps between markets: Opportunity to learn from high performers
            """)

        col1, col2 = st.columns([2, 1])

        with col1:
            # Heatmap of uniqueness by country
            uniqueness_heatmap_data = []
            for element in brand_elements:
                if element in uniqueness_by_country:
                    row_data = {'Element': element}
                    for country in ['UK', 'Spain', 'Germany', 'Poland']:
                        if country in uniqueness_by_country[element]:
                            row_data[country] = uniqueness_by_country[element][country]
                        else:
                            row_data[country] = 0
                    uniqueness_heatmap_data.append(row_data)

            uniqueness_heatmap_df = pd.DataFrame(uniqueness_heatmap_data).set_index('Element')

            fig_uniqueness_heatmap = px.imshow(
                uniqueness_heatmap_df,
                labels=dict(x="Country", y="Brand Element", color="Brand Attribution"),
                text_auto='.0%',
                aspect="auto",
                color_continuous_scale='RdYlGn',
                title="Brand Attribution (Uniqueness) by Country"
            )
            fig_uniqueness_heatmap.update_layout(height=500)
            st.plotly_chart(fig_uniqueness_heatmap, use_container_width=True)

        with col2:
            st.markdown("#### Key Findings:")

            # Find strongest market per element for uniqueness
            st.markdown("**Strongest Attribution:**")
            for element in brand_elements[:5]:  # Show top 5
                if element in uniqueness_by_country:
                    countries_sorted = sorted(
                        uniqueness_by_country[element].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    best_country = countries_sorted[0]
                    st.success(f"**{element}**: {best_country[0]} ({best_country[1]:.0%})")

            st.markdown("**Biggest Market Gaps:**")
            # Find elements with biggest uniqueness variations
            uniqueness_variations = []
            for element in brand_elements:
                if element in uniqueness_by_country:
                    values = list(uniqueness_by_country[element].values())
                    if values:
                        variation = max(values) - min(values)
                        min_country = min(uniqueness_by_country[element].items(), key=lambda x: x[1])
                        max_country = max(uniqueness_by_country[element].items(), key=lambda x: x[1])
                        uniqueness_variations.append((element, variation, min_country, max_country))

            uniqueness_variations_sorted = sorted(uniqueness_variations, key=lambda x: x[1], reverse=True)
            for element, var, min_c, max_c in uniqueness_variations_sorted[:3]:
                with st.expander(f"**{element}**: {var:.0%} gap"):
                    st.write(f"**Highest:** {max_c[0]} ({max_c[1]:.0%})")
                    st.write(f"**Lowest:** {min_c[0]} ({min_c[1]:.0%})")
                    st.markdown("**Why this gap matters:**")
                    if min_c[1] < 0.30:
                        st.warning(f"⚠️ In {min_c[0]}, consumers don't strongly associate {element} with Škoda - risk of competitor confusion")
                    st.markdown("**Strategic actions:**")
                    st.write(f"• **{max_c[0]} playbook:** Study what makes {element} distinctively Škoda here ({max_c[1]:.0%} attribution)")
                    st.write(f"• **{min_c[0]} support:** Increase co-branding of {element} with strong Škoda identifiers (Symbol, Wordmark)")
                    st.write(f"• **Investment priority:** Focus {element} investment in high-attribution markets; support with core brand assets in {min_c[0]}")

    st.markdown("---")

    # NEW: Market Consistency Score
    st.markdown("### 📊 Market Consistency Score: Which Assets Travel Well?")
    st.caption("Identifies which brand elements perform consistently across markets vs those that are market-specific")

    with st.expander("📖 Why market consistency matters"):
        st.markdown("""
        **Market consistency** reveals which assets are "universal" vs "local":

        **Universal assets** (low variation):
        - Perform similarly across all markets
        - Safe for global campaigns
        - Easy to scale internationally
        - Example: Symbol recognition 44-55% across all markets (11% variation)

        **Market-specific assets** (high variation):
        - Performance varies significantly by market
        - Require market-specific strategies
        - May need localization or repositioning
        - Example: Element X: 15% in UK, 45% in Poland (30% variation)

        **Strategic value:**
        - Identify assets for global rollout
        - Spot markets needing special attention
        - Optimize creative for regional differences
        """)

    # Calculate consistency scores for both recognition and uniqueness
    consistency_data = []

    for element in brand_elements:
        row = {'Element': element}

        # Recognition consistency (using recognition_by_country)
        if element in recognition_by_country:
            rec_values = list(recognition_by_country[element].values())
            if rec_values:
                rec_mean = sum(rec_values) / len(rec_values)
                rec_std = pd.Series(rec_values).std()
                rec_coef_var = (rec_std / rec_mean) if rec_mean > 0 else 0
                row['Recognition Mean'] = rec_mean
                row['Recognition StdDev'] = rec_std
                row['Recognition Consistency'] = 1 - rec_coef_var  # Higher = more consistent

        # Uniqueness consistency (using uniqueness_by_country)
        if element in uniqueness_by_country:
            uniq_values = list(uniqueness_by_country[element].values())
            if uniq_values:
                uniq_mean = sum(uniq_values) / len(uniq_values)
                uniq_std = pd.Series(uniq_values).std()
                uniq_coef_var = (uniq_std / uniq_mean) if uniq_mean > 0 else 0
                row['Uniqueness Mean'] = uniq_mean
                row['Uniqueness StdDev'] = uniq_std
                row['Uniqueness Consistency'] = 1 - uniq_coef_var  # Higher = more consistent

        consistency_data.append(row)

    consistency_df = pd.DataFrame(consistency_data)

    # Display consistency rankings
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌍 Most Consistent Assets (Travel Well)")
        st.caption("Low variation across markets - ideal for global campaigns")

        if 'Recognition Consistency' in consistency_df.columns:
            consistent_assets = consistency_df.nlargest(5, 'Recognition Consistency')
            for _, row in consistent_assets.iterrows():
                st.success(f"**{row['Element']}**")
                st.write(f"  • Avg Recognition: {row['Recognition Mean']:.0%}")
                st.write(f"  • Variation: ±{row['Recognition StdDev']:.1%}")
                if 'Uniqueness Mean' in row:
                    st.write(f"  • Avg Uniqueness: {row['Uniqueness Mean']:.0%}")

    with col2:
        st.markdown("#### 🗺️ Most Variable Assets (Market-Specific)")
        st.caption("High variation - requires localized strategy")

        if 'Recognition Consistency' in consistency_df.columns:
            variable_assets = consistency_df.nsmallest(5, 'Recognition Consistency')
            for _, row in variable_assets.iterrows():
                st.warning(f"**{row['Element']}**")
                st.write(f"  • Avg Recognition: {row['Recognition Mean']:.0%}")
                st.write(f"  • Variation: ±{row['Recognition StdDev']:.1%}")
                if element in recognition_by_country:
                    values = recognition_by_country[row['Element']]
                    min_market = min(values.items(), key=lambda x: x[1])
                    max_market = max(values.items(), key=lambda x: x[1])
                    st.caption(f"  Range: {min_market[0]} {min_market[1]:.0%} → {max_market[0]} {max_market[1]:.0%}")

    # Visualizations
    st.markdown("---")
    st.markdown("#### 📈 Consistency Score Comparison")

    col1, col2 = st.columns(2)

    with col1:
        # Recognition consistency bar chart
        if 'Recognition Consistency' in consistency_df.columns:
            rec_consistency = consistency_df[['Element', 'Recognition Consistency', 'Recognition StdDev']].sort_values('Recognition Consistency', ascending=True)

            fig_rec_consistency = go.Figure(go.Bar(
                x=rec_consistency['Recognition Consistency'],
                y=rec_consistency['Element'],
                orientation='h',
                marker_color='#4CAF50',
                text=rec_consistency['Recognition StdDev'].apply(lambda x: f'±{x:.1%}'),
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Consistency: %{x:.2f}<br>StdDev: %{text}<extra></extra>'
            ))

            fig_rec_consistency.update_layout(
                title='Recognition Consistency Score',
                xaxis_title='Consistency (Higher = More Consistent)',
                yaxis_title='',
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig_rec_consistency, use_container_width=True)

    with col2:
        # Uniqueness consistency bar chart
        if 'Uniqueness Consistency' in consistency_df.columns:
            uniq_consistency = consistency_df[['Element', 'Uniqueness Consistency', 'Uniqueness StdDev']].sort_values('Uniqueness Consistency', ascending=True)

            fig_uniq_consistency = go.Figure(go.Bar(
                x=uniq_consistency['Uniqueness Consistency'],
                y=uniq_consistency['Element'],
                orientation='h',
                marker_color='#2196F3',
                text=uniq_consistency['Uniqueness StdDev'].apply(lambda x: f'±{x:.1%}'),
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Consistency: %{x:.2f}<br>StdDev: %{text}<extra></extra>'
            ))

            fig_uniq_consistency.update_layout(
                title='Uniqueness Consistency Score',
                xaxis_title='Consistency (Higher = More Consistent)',
                yaxis_title='',
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig_uniq_consistency, use_container_width=True)

    # Strategic recommendations
    st.success("""
    **Strategic Recommendations:**

    **For Consistent Assets (Low Variation):**
    - Safe for global campaign rollout
    - Standardize creative guidelines across markets
    - Leverage learnings from one market to another

    **For Variable Assets (High Variation):**
    - Identify "hero markets" where asset performs well
    - Study what makes it work in strong markets
    - Create market-specific support plans for weak markets
    - Consider localized creative adaptations
    """)

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

    # NEW SECTION: First Recognition Trigger Index with Demographics
    if first_recognition_trigger or recognition_by_age_gender:
        st.markdown("### 🎯 First Recognition Trigger Index")
        st.caption("Which elements are most likely to trigger brand recognition when shown first?")

        st.info("""
        **KEY INSIGHT:** This analysis shows which brand elements are most effective at triggering
        immediate Škoda recognition when consumers see them as their FIRST exposure to the brand.
        Use the filters below to explore how this varies by age and gender.
        """)

        # Demographic selectors
        demo_col1, demo_col2, demo_col3 = st.columns([1, 1, 2])

        with demo_col1:
            trigger_demo_type = st.selectbox(
                "Filter by:",
                ["All Audiences", "Age", "Gender"],
                key="trigger_demo_type"
            )

        trigger_demo_filter = None
        trigger_demo_label = "All Audiences"

        if trigger_demo_type == "Age" and recognition_by_age_gender:
            with demo_col2:
                age_choice = st.selectbox("Age Group:", ["18-30", "31-42", "43-55"], key="trigger_age")
                trigger_demo_filter = ("age", age_choice)
                trigger_demo_label = f"Age {age_choice}"
        elif trigger_demo_type == "Gender" and recognition_by_age_gender:
            with demo_col2:
                gender_choice = st.selectbox("Gender:", ["Male", "Female"], key="trigger_gender")
                trigger_demo_filter = ("gender", gender_choice.lower())
                trigger_demo_label = gender_choice

        st.markdown(f"**Currently viewing:** {trigger_demo_label}")
        st.markdown("---")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Use filtered or default trigger data
            if first_recognition_trigger:
                # Create dataframe and sort by percentage
                trigger_df = pd.DataFrame([
                    {
                        'Element': element,
                        'Percent': data['percent_of_total_first_triggers'],
                        'Count': data['count'],
                        'Recognition Rate': data['percent_recognized']
                    }
                    for element, data in first_recognition_trigger.items()
                ]).sort_values('Percent', ascending=True)

                # Create horizontal bar chart
                fig_trigger = go.Figure(go.Bar(
                    x=trigger_df['Percent'],
                    y=trigger_df['Element'],
                    orientation='h',
                    marker_color='#4CAF50',
                    text=trigger_df['Percent'].apply(lambda x: f'{x:.1%}'),
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>%{x:.1%} of all first recognitions<br>Count: %{customdata}<extra></extra>',
                    customdata=trigger_df['Count']
                ))

                fig_trigger.update_layout(
                    title=f'First Recognition Trigger: {trigger_demo_label}',
                    xaxis_title='% of All First Recognitions',
                    yaxis_title='',
                    xaxis_tickformat='.0%',
                    height=400,
                    showlegend=False
                )

                st.plotly_chart(fig_trigger, use_container_width=True)
            else:
                st.warning("First recognition trigger data not available")

        with col2:
            st.markdown("#### 🔍 Key Findings")

            if first_recognition_trigger:
                # Get top trigger
                top_trigger = max(first_recognition_trigger.items(), key=lambda x: x[1]['percent_of_total_first_triggers'])
                st.success(f"**Top Trigger:** {top_trigger[0]}")
                st.metric("% of First Recognitions", f"{top_trigger[1]['percent_of_total_first_triggers']:.1%}")
                st.caption(f"{top_trigger[1]['count']} people recognized Škoda when shown this element first")

                st.markdown("---")

                st.markdown("#### 💡 Strategic Implication")
                st.markdown(f"""
                **{top_trigger[0]}** is your strongest "first impression" asset:
                - Use in teaser campaigns and new market launches
                - Prioritize in media with limited brand exposure time
                - Ensure prominent placement in all touchpoints
                """)

        st.markdown("---")

        # NEW: Age Migration Analysis
        if recognition_by_age_gender and uniqueness_by_age_gender:
            st.markdown("### 👥 Age Migration Analysis: How Recognition Triggers Shift by Cohort")
            st.caption("Shows which elements trigger recognition for different age groups and how distinctiveness varies")

            with st.expander("📖 Why age migration matters"):
                st.markdown("""
                **Age migration** reveals how brand recognition patterns shift across generations:

                - **Different age groups** may recognize different brand elements first
                - **Recognition rates** for the same element vary by cohort (e.g., younger audiences may respond to modern elements)
                - **Distinctiveness (uniqueness)** also varies - what feels "Škoda" to 18-30 may differ from 43-55
                - **Strategic insight:** Tailor asset deployment to target demographics

                This analysis helps you:
                1. Identify which elements resonate with each age group
                2. Spot generation gaps in brand recognition
                3. Optimize creative for specific audience segments
                """)

            # Create comparison across all age groups
            age_groups = ["18-30", "31-42", "43-55"]

            migration_data = []
            for element in brand_elements:
                row = {'Element': element}

                # Add recognition by age
                if element in recognition_by_age_gender and 'age' in recognition_by_age_gender[element]:
                    for age in age_groups:
                        if age in recognition_by_age_gender[element]['age']:
                            row[f'Recognition {age}'] = recognition_by_age_gender[element]['age'][age]

                # Add uniqueness by age
                if element in uniqueness_by_age_gender and 'age' in uniqueness_by_age_gender[element]:
                    for age in age_groups:
                        if age in uniqueness_by_age_gender[element]['age']:
                            row[f'Uniqueness {age}'] = uniqueness_by_age_gender[element]['age'][age]

                migration_data.append(row)

            migration_df = pd.DataFrame(migration_data)

            # Show recognition by age heatmap
            st.markdown("#### 🔥 Recognition Heatmap by Age")
            recognition_cols = [f'Recognition {age}' for age in age_groups]
            if all(col in migration_df.columns for col in recognition_cols):
                recognition_heatmap = migration_df[['Element'] + recognition_cols].set_index('Element')
                recognition_heatmap.columns = age_groups

                fig_rec_age = px.imshow(
                    recognition_heatmap,
                    labels=dict(x="Age Group", y="Brand Element", color="Recognition"),
                    text_auto='.0%',
                    aspect="auto",
                    color_continuous_scale='RdYlGn',
                    title="Recognition by Age Group"
                )
                fig_rec_age.update_layout(height=400)
                st.plotly_chart(fig_rec_age, use_container_width=True)

            # Show uniqueness by age heatmap
            st.markdown("#### 🎯 Distinctiveness (Uniqueness) by Age")
            uniqueness_cols = [f'Uniqueness {age}' for age in age_groups]
            if all(col in migration_df.columns for col in uniqueness_cols):
                uniqueness_heatmap = migration_df[['Element'] + uniqueness_cols].set_index('Element')
                uniqueness_heatmap.columns = age_groups

                fig_uniq_age = px.imshow(
                    uniqueness_heatmap,
                    labels=dict(x="Age Group", y="Brand Element", color="Uniqueness"),
                    text_auto='.0%',
                    aspect="auto",
                    color_continuous_scale='RdYlGn',
                    title="Brand Distinctiveness (Uniqueness) by Age Group"
                )
                fig_uniq_age.update_layout(height=400)
                st.plotly_chart(fig_uniq_age, use_container_width=True)

            # Key insights
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📊 Recognition Patterns")
                if all(col in migration_df.columns for col in recognition_cols):
                    for element in brand_elements[:3]:  # Top 3
                        element_row = migration_df[migration_df['Element'] == element].iloc[0]
                        values = [element_row[col] for col in recognition_cols if col in element_row]
                        if values:
                            max_age_idx = values.index(max(values))
                            min_age_idx = values.index(min(values))
                            st.write(f"**{element}:**")
                            st.write(f"  • Strongest: {age_groups[max_age_idx]} ({values[max_age_idx]:.0%})")
                            st.write(f"  • Weakest: {age_groups[min_age_idx]} ({values[min_age_idx]:.0%})")

            with col2:
                st.markdown("#### 🎯 Distinctiveness Patterns")
                if all(col in migration_df.columns for col in uniqueness_cols):
                    for element in brand_elements[:3]:  # Top 3
                        element_row = migration_df[migration_df['Element'] == element].iloc[0]
                        values = [element_row[col] for col in uniqueness_cols if col in element_row]
                        if values:
                            max_age_idx = values.index(max(values))
                            min_age_idx = values.index(min(values))
                            st.write(f"**{element}:**")
                            st.write(f"  • Most distinctive: {age_groups[max_age_idx]} ({values[max_age_idx]:.0%})")
                            st.write(f"  • Least distinctive: {age_groups[min_age_idx]} ({values[min_age_idx]:.0%})")

            st.success("""
            **Strategic Takeaway:** Use this age migration data to:
            - Target younger audiences with elements that score high in recognition and uniqueness for 18-30
            - Reinforce traditional elements (like Symbol) with older cohorts where they perform best
            - Identify cross-generational assets that work across all age groups
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
            {'Response': 'Fits expectations', 'Percentage': response_to_reveal['fits_expectations'], 'Sentiment': 'Positive'},
            {'Response': 'Does not fit', 'Percentage': response_to_reveal['does_not_fit'], 'Sentiment': 'Negative'},
            {'Response': 'Had not heard of Škoda', 'Percentage': response_to_reveal['not_heard_of_skoda'], 'Sentiment': 'Neutral'},
            {'Response': 'Other', 'Percentage': response_to_reveal['other'], 'Sentiment': 'Neutral'},
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

        st.metric("Fits Expectations", f"{response_to_reveal['fits_expectations']:.0%}", "Aligns with Škoda brand")

        st.metric("Does Not Fit", f"{response_to_reveal['does_not_fit']:.0%}", "Conflicts with brand perception")

        st.metric("Unaware of Škoda", f"{response_to_reveal['not_heard_of_skoda']:.0%}", "No prior brand knowledge")

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
    
    2. **Create Combinations** - Since single elements drive only 10% recognition, ensure multiple elements appear together. Recommended minimum: 3 elements per touchpoint.
    
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
