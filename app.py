import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import json
import os

# =====================================================================
# 🎨 ŠKODA BRAND INTELLIGENCE DASHBOARD
# Data-Driven Brand Asset Analysis Platform
# =====================================================================

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Škoda Brand Intelligence Dashboard",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# =====================================================================
# SESSION STATE INITIALIZATION
# =====================================================================

# Initialize session state for global filters
if 'global_filters_enabled' not in st.session_state:
    st.session_state.global_filters_enabled = False

if 'global_country' not in st.session_state:
    st.session_state.global_country = "All Countries"

if 'global_age' not in st.session_state:
    st.session_state.global_age = "All Ages"

if 'global_gender' not in st.session_state:
    st.session_state.global_gender = "All Genders"

if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = False

if 'selected_elements' not in st.session_state:
    st.session_state.selected_elements = []

if 'show_raw_data' not in st.session_state:
    st.session_state.show_raw_data = False

# =====================================================================
# DATA LOADING
# =====================================================================

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

# Load Q05 competitor detail data
try:
    if os.path.exists('q05_competitor_detail_CLEANED.json'):
        with open('q05_competitor_detail_CLEANED.json', 'r') as f:
            competitor_detail = json.load(f)
    else:
        competitor_detail = {}
except Exception as e:
    competitor_detail = {}

# Load Q29 MaxDiff Asset Power Rankings
try:
    if os.path.exists('q29_rankings_first.json'):
        with open('q29_rankings_first.json', 'r', encoding='utf-8') as f:
            q29_rankings_first = json.load(f)
    else:
        q29_rankings_first = {}

    if os.path.exists('q29_rankings_top3.json'):
        with open('q29_rankings_top3.json', 'r', encoding='utf-8') as f:
            q29_rankings_top3 = json.load(f)
    else:
        q29_rankings_top3 = {}
except Exception as e:
    st.error(f"Error loading Q29 data: {e}")
    q29_rankings_first = {}
    q29_rankings_top3 = {}

# Load Q30 Top-of-Mind Word Associations
try:
    if os.path.exists('q30_word_associations.json'):
        with open('q30_word_associations.json', 'r', encoding='utf-8') as f:
            q30_word_associations = json.load(f)
    else:
        q30_word_associations = []
except Exception as e:
    st.error(f"Error loading Q30 data: {e}")
    q30_word_associations = []

# Load Q28 Emotional Response to Reveal
try:
    if os.path.exists('q28_emotional_response.json'):
        with open('q28_emotional_response.json', 'r', encoding='utf-8') as f:
            q28_emotional_response = json.load(f)
    else:
        q28_emotional_response = []
except Exception as e:
    st.error(f"Error loading Q28 data: {e}")
    q28_emotional_response = []

from comms_data import comms_audit_data

# --- Brand Elements ---
brand_elements = [
    "Electric Green", "Emerald Green", "Type", "Tagline", "Symbol",
    "Hacek", "Wordmark", "Facets", "Sonic"
]

# Survey Base
SURVEY_BASE = 2011  # Total respondents across UK, Spain, Germany, Poland

# --- VERIFIED Research Data from P045556 Study ---
research_data = {
    'Electric Green': {
        'recognition': 0.376, 'uniqueness': 0.174,
        'bold': 0.490, 'stylish': 0.463, 'modern': 0.499, 'simple': 0.502,
        'human': 0.452, 'exciting': 0.450, 'playful': 0.443,
        'positive_sentiment': 0.471, 'negative_sentiment': 0.529,
        'net_sentiment': -0.057
    },
    'Emerald Green': {
        'recognition': 0.393, 'uniqueness': 0.195,
        'bold': 0.493, 'stylish': 0.473, 'modern': 0.503, 'simple': 0.513,
        'human': 0.453, 'exciting': 0.460, 'playful': 0.453,
        'positive_sentiment': 0.478, 'negative_sentiment': 0.522,
        'net_sentiment': -0.044
    },
    'Type': {
        'recognition': 0.365, 'uniqueness': 0.169,
        'bold': 0.514, 'stylish': 0.476, 'modern': 0.516, 'simple': 0.533,
        'human': 0.469, 'exciting': 0.482, 'playful': 0.475,
        'positive_sentiment': 0.495, 'negative_sentiment': 0.505,
        'net_sentiment': -0.010
    },
    'Tagline': {
        'recognition': 0.383, 'uniqueness': 0.175,
        'bold': 0.498, 'stylish': 0.478, 'modern': 0.507, 'simple': 0.527,
        'human': 0.494, 'exciting': 0.478, 'playful': 0.483,
        'positive_sentiment': 0.495, 'negative_sentiment': 0.505,
        'net_sentiment': -0.010
    },
    'Symbol': {
        'recognition': 0.643, 'uniqueness': 0.385,
        'bold': 0.488, 'stylish': 0.492, 'modern': 0.507, 'simple': 0.558,
        'human': 0.481, 'exciting': 0.476, 'playful': 0.489,
        'positive_sentiment': 0.499, 'negative_sentiment': 0.501,
        'net_sentiment': -0.002
    },
    'Hacek': {
        'recognition': 0.362, 'uniqueness': 0.186,
        'bold': 0.499, 'stylish': 0.476, 'modern': 0.506, 'simple': 0.525,
        'human': 0.480, 'exciting': 0.480, 'playful': 0.483,
        'positive_sentiment': 0.493, 'negative_sentiment': 0.507,
        'net_sentiment': -0.014
    },
    'Wordmark': {
        'recognition': 0.456, 'uniqueness': 0.279,
        'bold': 0.507, 'stylish': 0.472, 'modern': 0.504, 'simple': 0.547,
        'human': 0.475, 'exciting': 0.472, 'playful': 0.478,
        'positive_sentiment': 0.493, 'negative_sentiment': 0.507,
        'net_sentiment': -0.013
    },
    'Facets': {
        'recognition': 0.414, 'uniqueness': 0.158,
        'bold': 0.507, 'stylish': 0.493, 'modern': 0.516, 'simple': 0.526,
        'human': 0.471, 'exciting': 0.488, 'playful': 0.483,
        'positive_sentiment': 0.498, 'negative_sentiment': 0.502,
        'net_sentiment': -0.004
    },
    'Sonic': {
        'recognition': 0.513, 'uniqueness': 0.227,
        'bold': 0.490, 'stylish': 0.472, 'modern': 0.497, 'simple': 0.533,
        'human': 0.490, 'exciting': 0.477, 'playful': 0.480,
        'positive_sentiment': 0.491, 'negative_sentiment': 0.509,
        'net_sentiment': -0.018
    }
}

# Recognition by country data
recognition_by_country = {
    'Electric Green': {'UK': 0.41, 'Spain': 0.377, 'Germany': 0.294, 'Poland': 0.436},
    'Emerald Green': {'UK': 0.376, 'Spain': 0.383, 'Germany': 0.372, 'Poland': 0.413},
    'Type': {'UK': 0.452, 'Spain': 0.368, 'Germany': 0.301, 'Poland': 0.373},
    'Tagline': {'UK': 0.353, 'Spain': 0.394, 'Germany': 0.323, 'Poland': 0.364},
    'Symbol': {'UK': 0.535, 'Spain': 0.661, 'Germany': 0.610, 'Poland': 0.765},
    'Hacek': {'UK': 0.353, 'Spain': 0.379, 'Germany': 0.354, 'Poland': 0.410},
    'Wordmark': {'UK': 0.453, 'Spain': 0.462, 'Germany': 0.386, 'Poland': 0.485},
    'Facets': {'UK': 0.413, 'Spain': 0.395, 'Germany': 0.369, 'Poland': 0.355},
    'Sonic': {'UK': 0.391, 'Spain': 0.420, 'Germany': 0.386, 'Poland': 0.394},
}

# Recognition Journey data
recognition_journey = {
    'after_1_element': 0.102,
    'after_2_elements': 0.109,
    'after_3_elements': 0.243,
    'after_4_elements': 0.403,
    'after_5_elements': 0.427,
    'after_all_6_elements': 0.438,
    'never_recognized': 0.562
}

# Škoda Familiarity data
skoda_familiarity = {
    'very_familiar': 0.214,
    'quite_familiar': 0.386,
    'heard_of_not_much': 0.321,
    'never_heard': 0.045,
    'not_sure': 0.034
}

# Response to reveal data
response_to_reveal = {
    'fits_expectations': 0.560,
    'does_not_fit': 0.222,
    'not_heard_of_skoda': 0.078,
    'other': 0.007,
    'dont_know': 0.133
}

# Demographics data
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
        'male': 0.490,
        'female': 0.507
    },
    'skoda_awareness': {
        'heard_of_skoda': 0.92,
        'unaware': 0.08
    }
}

# Adjective data for semantic differential
adjective_data = {
    'Electric Green': {
        'bold': {'positive_net': 0.490, 'negative_net': 0.218, 'neutral': 0.293, 'negative_adjective': 'Cautious'},
        'stylish': {'positive_net': 0.463, 'negative_net': 0.144, 'neutral': 0.301, 'negative_adjective': 'Plain'},
        'modern': {'positive_net': 0.499, 'negative_net': 0.216, 'neutral': 0.286, 'negative_adjective': 'Old-Fashioned'},
        'playful': {'positive_net': 0.443, 'negative_net': 0.264, 'neutral': 0.293, 'negative_adjective': 'Serious'},
        'exciting': {'positive_net': 0.450, 'negative_net': 0.264, 'neutral': 0.287, 'negative_adjective': 'Boring'},
        'human': {'positive_net': 0.452, 'negative_net': 0.243, 'neutral': 0.305, 'negative_adjective': 'Cold'},
        'simple': {'positive_net': 0.502, 'negative_net': 0.218, 'neutral': 0.280, 'negative_adjective': 'Complicated'},
    },
    'Facets': {
        'bold': {'positive_net': 0.502, 'negative_net': 0.216, 'neutral': 0.282, 'negative_adjective': 'Cautious'},
        'stylish': {'positive_net': 0.484, 'negative_net': 0.255, 'neutral': 0.262, 'negative_adjective': 'Plain'},
        'modern': {'positive_net': 0.514, 'negative_net': 0.204, 'neutral': 0.282, 'negative_adjective': 'Old-Fashioned'},
        'playful': {'positive_net': 0.461, 'negative_net': 0.254, 'neutral': 0.285, 'negative_adjective': 'Serious'},
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
        'stylish': {'positive_net': 0.490, 'negative_net': 0.202, 'neutral': 0.308, 'negative_adjective': 'Plain'},
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
        'exciting': {'positive_net': 0.509, 'negative_net': 0.227, 'neutral': 0.264, 'negative_adjective': 'Boring'},
        'human': {'positive_net': 0.464, 'negative_net': 0.232, 'neutral': 0.304, 'negative_adjective': 'Cold'},
        'simple': {'positive_net': 0.495, 'negative_net': 0.199, 'neutral': 0.306, 'negative_adjective': 'Complicated'},
    },
}

# =====================================================================
# DESIGN CONSTANTS FOR CONSISTENCY
# =====================================================================

# Color Palette
COLORS = {
    'primary': '#4A90E2',      # Škoda blue
    'success': '#4CAF50',      # Green
    'warning': '#FFC107',      # Yellow
    'error': '#F44336',        # Red
    'info': '#2196F3',         # Light blue
    'neutral': '#757575',      # Grey
    'background': '#F5F5F5',   # Light grey
    'accent': '#667eea',       # Purple
}

# Button Styling
BUTTON_STYLE = """
<style>
.stButton>button {
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 500;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.stDownloadButton>button {
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 500;
}
</style>
"""

# Mobile Responsive Styling
MOBILE_STYLE = """
<style>
@media (max-width: 768px) {
    .stColumns {
        flex-direction: column;
    }
    .row-widget.stRadio > div {
        flex-direction: column;
    }
    .stMetric {
        font-size: 0.9em;
    }
}
</style>
"""

# Glossary of Terms
GLOSSARY = {
    "Recognition": "How many people recognize this element - shows brand visibility and awareness",
    "Uniqueness": "How many people know this belongs to Škoda (not competitors) - measures brand ownership strength",
    "Brand Equity": "Recognition × Uniqueness - shows if an element is both famous AND identified as Škoda's",
    "Brand Linkage": "How strongly people connect this element to Škoda - shows perceived brand ownership",
    "Top-of-Mind": "Words people think of first when they hear 'Škoda' - reveals unprompted brand associations",
    "ROI per €1M": "Recognition points gained per €1M spent - higher scores mean better value for money",
    "Net Sentiment": "Positive associations minus negative - shows if people feel good or bad about this element",
    "Usage": "Percentage of advertising campaigns that include this brand element",
    "Most Positive (Most Positive)": "Percentage who gave one of the 2 most positive ratings (strongly agree or agree)",
    "Most Negative (Most Negative)": "Percentage who gave one of the 2 most negative ratings (strongly disagree or disagree)",
    "Market Consistency": "Low variation = works everywhere. High variation = tailor strategy by market",
    "First Recognition Trigger": "Which elements make people think 'Škoda' first - best for grabbing attention quickly",
}

# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

# Apply styling
st.markdown(BUTTON_STYLE, unsafe_allow_html=True)
st.markdown(MOBILE_STYLE, unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def to_excel(df):
    """Convert DataFrame to Excel bytes for download"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    return output.getvalue()

def help_icon(term):
    """Render a help icon with tooltip for complex terms"""
    if term in GLOSSARY:
        return f" [ℹ️]({GLOSSARY[term]} '{GLOSSARY[term]}')"
    return ""

def render_empty_state(title="No Data Found", suggestions=None):
    """Render a helpful empty state when filters return no results"""
    st.warning(f"### {title}")
    st.markdown("""
    **No data matches your current filters.**

    Try:
    - Adjusting your filter criteria
    - Resetting filters to defaults
    - Selecting different elements or markets
    """)

    if suggestions:
        st.info("**💡 Suggestions:**")
        for suggestion in suggestions:
            st.markdown(f"• {suggestion}")

    if st.button("🔄 Reset All Filters", key=f"reset_{title}"):
        st.session_state.global_country = "All Countries"
        st.session_state.global_age = "All Ages"
        st.session_state.global_gender = "All Genders"
        st.toast("✅ Filters reset!", icon="✅")
        st.rerun()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def calculate_metrics():
    """Calculate master metrics DataFrame - cached for performance"""
    audit_df = pd.DataFrame(comms_audit_data)

    # Calculate overall usage
    usage_data = []
    for element in brand_elements:
        if element in audit_df.columns:
            overall_usage = audit_df[element].mean()
        else:
            overall_usage = 0.0
        usage_data.append(overall_usage)

    # Create master DataFrame
    master_df = pd.DataFrame({
        'Element': brand_elements,
        'Recognition': [research_data[e]['recognition'] for e in brand_elements],
        'Uniqueness': [research_data[e]['uniqueness'] for e in brand_elements],
        'Overall Usage': usage_data,
        'Positive Sentiment': [research_data[e]['positive_sentiment'] for e in brand_elements],
        'Negative Sentiment': [research_data[e]['negative_sentiment'] for e in brand_elements],
        'Net Sentiment': [research_data[e]['net_sentiment'] for e in brand_elements]
    })

    # Calculate investment per element
    for element in brand_elements:
        if element in audit_df.columns:
            ads_with_element = audit_df[audit_df[element] == True]
            total_investment = ads_with_element['Spend'].sum() if len(ads_with_element) > 0 else 0
        else:
            total_investment = 0
        master_df.loc[master_df['Element'] == element, 'Total Investment'] = total_investment

    # Calculate ROI metrics
    master_df['Recognition ROI'] = master_df.apply(
        lambda row: (row['Recognition'] / (row['Total Investment'] / 1_000_000)) if row['Total Investment'] > 0 else 0,
        axis=1
    )

    return master_df, audit_df

# =====================================================================
# UI COMPONENT FUNCTIONS (REUSABLE)
# =====================================================================

def render_tldr_box(title, bullets):
    """
    Render a TL;DR summary box with gradient background

    Args:
        title: Section title
        bullets: List of bullet points (max 3 recommended)
    """
    bullets_html = "".join([f"<li style='margin: 8px 0;'>{bullet}</li>" for bullet in bullets])

    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 25px; border-radius: 12px; color: white; margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <h3 style='margin: 0 0 15px 0; font-size: 1.4em;'>⚡ {title}</h3>
        <p style='margin: 0 0 10px 0; opacity: 0.9; font-size: 0.95em;'>30-second read</p>
        <ul style='margin: 0; padding-left: 20px; font-size: 1.05em; line-height: 1.6;'>
            {bullets_html}
        </ul>
    </div>
    """, unsafe_allow_html=True)

def render_action_box(do_items, stop_items=None):
    """
    Render 'What This Means' action box with do/stop recommendations

    Args:
        do_items: List of things to DO (shown in green)
        stop_items: Optional list of things to STOP (shown in red)
    """
    cols = st.columns(2) if stop_items else [st.container()]

    with cols[0]:
        st.markdown("""
        <div style='background-color: #d4edda; border-left: 5px solid #28a745;
                    padding: 20px; border-radius: 8px; margin: 10px 0;'>
            <h4 style='color: #155724; margin: 0 0 15px 0;'>✅ DO THIS</h4>
        """, unsafe_allow_html=True)
        for item in do_items:
            st.markdown(f"• {item}")
        st.markdown("</div>", unsafe_allow_html=True)

    if stop_items and len(cols) > 1:
        with cols[1]:
            st.markdown("""
            <div style='background-color: #f8d7da; border-left: 5px solid #dc3545;
                        padding: 20px; border-radius: 8px; margin: 10px 0;'>
                <h4 style='color: #721c24; margin: 0 0 15px 0;'>🚫 STOP THIS</h4>
            """, unsafe_allow_html=True)
            for item in stop_items:
                st.markdown(f"• {item}")
            st.markdown("</div>", unsafe_allow_html=True)

def render_section_header(title, subtitle="", color="#667eea"):
    """
    Render a color-coded section header with divider

    Args:
        title: Section title
        subtitle: Optional subtitle
        color: Hex color for the accent
    """
    subtitle_html = f"<p style='color: #666; margin: 5px 0 0 0; font-size: 0.95em;'>{subtitle}</p>" if subtitle else ""

    st.markdown(f"""
    <div style='border-left: 5px solid {color}; padding-left: 15px; margin: 30px 0 20px 0;'>
        <h3 style='margin: 0; color: {color}; font-size: 1.5em;'>{title}</h3>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)

def render_metric_card_enhanced(label, value, delta=None, help_text="", icon=""):
    """
    Enhanced metric card with optional icon and delta

    Args:
        label: Metric label
        value: Metric value (string or number)
        delta: Optional delta value for comparison
        help_text: Tooltip text
        icon: Optional emoji icon
    """
    st.metric(
        label=f"{icon} {label}" if icon else label,
        value=value,
        delta=delta,
        help=help_text
    )

def get_standard_chart_config():
    """
    Returns standard configuration for all Plotly charts
    """
    return {
        'displayModeBar': True,
        'modeBarButtonsToAdd': ['downloadSvg'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'skoda_chart',
            'height': 800,
            'width': 1200,
            'scale': 2
        }
    }

def apply_standard_chart_styling(fig, title=""):
    """
    Apply consistent styling to Plotly charts

    Args:
        fig: Plotly figure object
        title: Chart title
    """
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#333'}
        },
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Arial, sans-serif', 'color': '#333'},
        margin=dict(t=60, b=60, l=60, r=60),
        hovermode='closest'
    )

    # Grid styling
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)'
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)'
    )

    return fig

# =====================================================================
# FILTER LOGIC
# =====================================================================

def render_demographic_filters(prefix="", use_global=False):
    """
    Render demographic filters - can use global or local

    Args:
        prefix: Unique prefix for widget keys
        use_global: If True, use global session state filters

    Returns:
        dict with country, age, gender, and context_text
    """
    if use_global and st.session_state.global_filters_enabled:
        # Use global filters
        return {
            'country': st.session_state.global_country,
            'age': st.session_state.global_age,
            'gender': st.session_state.global_gender,
            'context_text': f"{st.session_state.global_country} | {st.session_state.global_age} | {st.session_state.global_gender}"
        }
    else:
        # Render local filters
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

        return {
            'country': country,
            'age': age,
            'gender': gender,
            'context_text': context_text
        }

def apply_demographic_filters(df, filters, elements):
    """
    Apply demographic filters to update Recognition and Uniqueness

    Args:
        df: DataFrame to update (will be copied)
        filters: dict from render_demographic_filters()
        elements: list of element names

    Returns:
        filtered DataFrame
    """
    df = df.copy()

    # Apply recognition filters (age/gender)
    if filters['age'] != "All Ages" or filters['gender'] != "All Genders":
        for element in elements:
            if element in recognition_by_age_gender:
                age_key = filters['age'] if filters['age'] != "All Ages" else None
                gender_key = filters['gender'] if filters['gender'] != "All Genders" else None

                if age_key and gender_key:
                    key = f"{age_key}_{gender_key}"
                    if key in recognition_by_age_gender[element]:
                        df.loc[df['Element'] == element, 'Recognition'] = recognition_by_age_gender[element][key]
                elif age_key:
                    for key, val in recognition_by_age_gender[element].items():
                        if age_key in key:
                            df.loc[df['Element'] == element, 'Recognition'] = val
                            break
                elif gender_key:
                    for key, val in recognition_by_age_gender[element].items():
                        if gender_key in key:
                            df.loc[df['Element'] == element, 'Recognition'] = val
                            break

    # Apply uniqueness filters (country)
    if filters['country'] != "All Countries":
        for element in elements:
            if element in uniqueness_by_country:
                country_key = filters['country']
                if country_key in uniqueness_by_country[element]:
                    df.loc[df['Element'] == element, 'Uniqueness'] = uniqueness_by_country[element][country_key]

    return df

# =====================================================================
# GLOBAL SIDEBAR
# =====================================================================

with st.sidebar:
    st.markdown("# 🎛️ Control Panel")
    st.markdown("---")

    # Global Filters Section
    st.markdown("### 🌍 Global Filters")
    st.caption("Apply filters across all tabs")

    enable_global = st.toggle(
        "Enable Global Filters",
        value=st.session_state.global_filters_enabled,
        help="When enabled, filters apply to all tabs automatically"
    )
    st.session_state.global_filters_enabled = enable_global

    if enable_global:
        st.session_state.global_country = st.selectbox(
            "Country",
            ["All Countries", "UK", "Spain", "Germany", "Poland"],
            key="sidebar_country"
        )

        st.session_state.global_age = st.selectbox(
            "Age Group",
            ["All Ages", "18-30", "31-42", "43-55"],
            key="sidebar_age"
        )

        st.session_state.global_gender = st.selectbox(
            "Gender",
            ["All Genders", "Male", "Female"],
            key="sidebar_gender"
        )

        # Show active filters
        st.success(f"""
        **Active Filters:**
        - {st.session_state.global_country}
        - {st.session_state.global_age}
        - {st.session_state.global_gender}
        """)

        if st.button("🔄 Reset All Filters"):
            st.session_state.global_country = "All Countries"
            st.session_state.global_age = "All Ages"
            st.session_state.global_gender = "All Genders"
            st.toast("✅ All filters reset successfully!", icon="✅")
            st.rerun()
    else:
        st.info("Global filters disabled. Use local filters in each tab.")

    st.markdown("---")

    # Comparison Mode Section
    st.markdown("### 🔀 Comparison Mode")
    comparison_enabled = st.toggle(
        "Enable Comparison",
        value=st.session_state.comparison_mode,
        help="Compare multiple elements side-by-side"
    )
    st.session_state.comparison_mode = comparison_enabled

    if comparison_enabled:
        st.session_state.selected_elements = st.multiselect(
            "Select elements to compare (2-4):",
            brand_elements,
            default=st.session_state.selected_elements[:4] if st.session_state.selected_elements else []
        )

        if len(st.session_state.selected_elements) >= 2:
            st.success(f"Comparing {len(st.session_state.selected_elements)} elements")
        else:
            st.warning("Select at least 2 elements")

    st.markdown("---")

    # Quick Actions
    st.markdown("### ⚡ Quick Actions")

    if st.button("📊 Export All Data (Excel)"):
        with st.spinner("Preparing Excel export..."):
            master_df, audit_df = calculate_metrics()
            excel_data = to_excel(master_df)
        st.download_button(
            label="Download Excel",
            data=excel_data,
            file_name="skoda_complete_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.toast("📊 Excel file ready for download!", icon="📊")

    if st.button("🔄 Refresh Dashboard"):
        st.toast("🔄 Refreshing dashboard...", icon="🔄")
        st.rerun()

    st.markdown("---")

    # Raw Data Access
    st.markdown("### 📄 Data Access")
    if st.button("🔍 View Raw Data", use_container_width=True):
        st.session_state.show_raw_data = True
        st.toast("📄 Opening Data Explorer...", icon="📄")
        st.rerun()

    st.markdown("---")

    # Glossary Section
    st.markdown("### 📖 Glossary")
    with st.expander("View Terminology Guide"):
        st.markdown("**Key Terms Explained:**")
        for term, definition in GLOSSARY.items():
            st.markdown(f"**{term}**")
            st.caption(definition)
            st.markdown("")

    st.markdown("---")
    st.caption("Škoda Brand Intelligence Dashboard")
    st.caption("© 2025 Saffron Brand Consultants")

# =====================================================================
# PERSISTENT HEADER BAR
# =====================================================================

# Show active filters in header if global filters enabled
if st.session_state.global_filters_enabled:
    filter_text = f"🎯 Active: {st.session_state.global_country} | {st.session_state.global_age} | {st.session_state.global_gender}"
    st.markdown(f"""
    <div style='background-color: #e3f2fd; padding: 10px 20px; border-radius: 8px;
                margin-bottom: 20px; text-align: center; border: 2px solid #2196F3;'>
        <strong>{filter_text}</strong>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# APP HEADER
# =====================================================================

st.title("📊 Škoda Brand Intelligence Dashboard")
st.markdown("---")

# Calculate master metrics once for use across all tabs
master_df, audit_df = calculate_metrics()

# =====================================================================
# RAW DATA EXPLORER (CONDITIONAL DISPLAY)
# =====================================================================

if st.session_state.show_raw_data:
    st.markdown("---")
    st.markdown("---")
    st.header("📄 Data Explorer")
    st.caption("Raw data access and detailed views")

    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("✖️ Close", use_container_width=True):
            st.session_state.show_raw_data = False
            st.rerun()

    st.markdown("---")

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
        st.markdown("### Research Data (9 Elements)")
        st.dataframe(master_df, use_container_width=True)

        csv = master_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Research CSV",
            data=csv,
            file_name="skoda_research_data.csv",
            mime="text/csv"
        )

    with tab_c:
        st.markdown("### Combined Metrics View")

        # Show key metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Ads Analyzed", len(audit_df))

        with col2:
            st.metric("Brand Elements", len(master_df))

        with col3:
            total_spend = audit_df['Spend (£)'].sum() if 'Spend (£)' in audit_df.columns else 0
            st.metric("Total Ad Spend", f"£{total_spend:,.0f}")

        st.markdown("---")

        # Combine relevant data
        st.markdown("#### Recognition vs. Ad Spend")
        st.caption("How brand element recognition relates to advertising investment")

        if 'Element' in master_df.columns and 'Recognition_Total' in master_df.columns:
            combined = master_df[['Element', 'Recognition_Total', 'Uniqueness_Total', 'Brand_Equity']].copy()
            st.dataframe(combined, use_container_width=True)

    with tab_d:
        st.markdown("### Survey Demographics")
        st.caption("Sample composition across markets")

        # Sample sizes per country
        sample_data = {
            'Market': ['UK', 'Spain', 'Germany', 'Poland', 'Total'],
            'Sample Size': [450, 475, 440, 490, 1855],
            'Aware of Škoda': [324, 380, 352, 441, 1497],
            'Aware %': ['72%', '80%', '80%', '90%', '81%']
        }

        import pandas as pd
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)

        st.markdown("---")
        st.caption("Data collected: October 2025")
        st.caption("Research method: Online quantitative survey")

    st.stop()

# =====================================================================
# TAB NAVIGATION
# =====================================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview",
    "💚 Brand Perception",
    "📈 Portfolio Strategy",
    "🎯 Performance Tiers",
    "🔮 Growth Opportunities",
    "🔍 Market Analysis",
    "🧭 Consumer Journey"
])

# =====================================================================
# TAB 1: EXECUTIVE SUMMARY
# =====================================================================
with tab1:
    st.markdown("## Executive Summary")
    st.caption("Performance snapshot: which elements are strongest, and where to invest")

    # TL;DR Box
    most_recognized = master_df.nlargest(1, 'Recognition').iloc[0]
    lowest_roi = master_df.nsmallest(1, 'Recognition ROI').iloc[0]
    total_investment = master_df['Total Investment'].sum()

    best_roi = master_df.nlargest(1, 'Recognition ROI').iloc[0]
    worst_roi = master_df.nsmallest(1, 'Recognition ROI').iloc[0]

    render_tldr_box(
        "Key Insights at a Glance",
        [
            f"<b>{most_recognized['Element']}</b> demonstrates the highest performance: {most_recognized['Recognition']:.0%} recognition and {most_recognized['Uniqueness']:.0%} uniqueness",
            f"<b>€{total_investment:,.0f}</b> total investment across 9 brand elements with efficiency ranging from {worst_roi['Recognition ROI']:.2f} to {best_roi['Recognition ROI']:.2f} ROI per €1M",
            f"<b>Recognition range:</b> {lowest_roi['Recognition']:.0%} ({lowest_roi['Element']}) to {most_recognized['Recognition']:.0%} ({most_recognized['Element']}) showing varied brand awareness levels"
        ]
    )

    # Quick Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric_card_enhanced(
            "Most Recognized",
            most_recognized['Element'],
            delta=f"{most_recognized['Recognition']:.0%}",
            help_text="% of people who have seen/heard this element",
            icon="⭐"
        )

    with col2:
        most_unique = master_df.nlargest(1, 'Uniqueness').iloc[0]
        render_metric_card_enhanced(
            "Most Unique",
            most_unique['Element'],
            delta=f"{most_unique['Uniqueness']:.0%}",
            help_text="% who correctly identify as Škoda",
            icon="💎"
        )

    with col3:
        render_metric_card_enhanced(
            "Total Investment",
            f"€{total_investment/1_000_000:.1f}M",
            help_text="Across all brand elements",
            icon="💰"
        )

    with col4:
        best_roi = master_df.nlargest(1, 'Recognition ROI').iloc[0]
        render_metric_card_enhanced(
            "Best ROI",
            best_roi['Element'],
            delta=f"{best_roi['Recognition ROI']:.2f}x",
            help_text="Recognition points per €1M spent",
            icon="📈"
        )

    # Enhanced metric cards with detailed expanders
    col1, col2, col3, col4 = st.columns(4)

    highest_investment = master_df.nlargest(1, 'Total Investment').iloc[0]

    with col1:
        with st.expander(f"📊 Why is **{most_recognized['Element']}** most recognized?"):
            median_usage = master_df['Overall Usage'].median()
            median_investment = master_df['Total Investment'].median()

            factors = []
            if most_recognized['Total Investment'] >= median_investment:
                inv_vs_median = ((most_recognized['Total Investment'] / median_investment) - 1) * 100
                factors.append(f"**Substantial Investment:** €{most_recognized['Total Investment']:,.0f} invested ({inv_vs_median:.0f}% above median)")

            if most_recognized['Overall Usage'] >= median_usage:
                factors.append(f"**High Campaign Frequency:** Used in {most_recognized['Overall Usage']:.0%} of campaigns")

            if most_recognized['Recognition ROI'] >= master_df['Recognition ROI'].median():
                factors.append(f"**Strong ROI:** {most_recognized['Recognition ROI']:.2f} recognition points per €1M")

            if most_recognized['Uniqueness'] >= master_df['Uniqueness'].median():
                factors.append(f"**Distinctive Design:** {most_recognized['Uniqueness']:.0%} uniqueness reinforces attribution")

            st.markdown("**Observed Patterns:**")
            for i, factor in enumerate(factors, 1):
                st.markdown(f"{i}. {factor}")

            st.info(f"**Context:** {most_recognized['Recognition']:.0%} recognition places this element {((most_recognized['Recognition'] / master_df['Recognition'].mean()) - 1) * 100:.0f}% above portfolio average.")

    with col2:
        with st.expander(f"🎯 Why is **{most_unique['Element']}** most unique?"):
            factors = []
            if most_unique['Recognition'] >= master_df['Recognition'].median():
                factors.append(f"**Strong Recognition:** {most_unique['Recognition']:.0%} of consumers have seen it")

            uniqueness_gap = most_unique['Uniqueness'] - master_df['Uniqueness'].median()
            if uniqueness_gap >= 0.15:
                factors.append(f"**Exceptional Distinctiveness:** {uniqueness_gap:.0%} points above median")

            if most_unique['Recognition ROI'] >= master_df['Recognition ROI'].median():
                factors.append(f"**Efficient Performance:** {most_unique['Recognition ROI']:.2f} ROI per €1M")

            st.markdown("**Distinctiveness Factors:**")
            for i, factor in enumerate(factors, 1):
                st.markdown(f"{i}. {factor}")

            st.info(f"**Context:** {most_unique['Uniqueness']:.0%} uniqueness is {((most_unique['Uniqueness'] / master_df['Uniqueness'].mean()) - 1) * 100:.0f}% above portfolio average of {master_df['Uniqueness'].mean():.0%}.")

    with col3:
        with st.expander(f"💰 Why does **{highest_investment['Element']}** have highest investment?"):
            inv_vs_median = ((highest_investment['Total Investment'] / master_df['Total Investment'].median()) - 1) * 100
            num_ads = int(highest_investment['Overall Usage'] * len(audit_df))

            st.markdown(f"**Investment:** €{highest_investment['Total Investment']:,.0f} ({inv_vs_median:.0f}% above median)")
            st.markdown(f"**Usage:** {highest_investment['Overall Usage']:.0%} of campaigns ({num_ads} ads)")
            st.markdown(f"**Recognition Achieved:** {highest_investment['Recognition']:.0%}")
            st.markdown(f"**ROI:** {highest_investment['Recognition ROI']:.2f} per €1M")

            roi_ratio = highest_investment['Recognition ROI'] / best_roi['Recognition ROI'] if best_roi['Recognition ROI'] > 0 else 0
            st.info(f"**Relative Efficiency:** This element's ROI of {highest_investment['Recognition ROI']:.2f} represents {roi_ratio:.0%} of the portfolio's best performer ({best_roi['Element']} at {best_roi['Recognition ROI']:.2f}).")

    with col4:
        with st.expander(f"⚡ Why is **{best_roi['Element']}** most efficient?"):
            roi_vs_median = ((best_roi['Recognition ROI'] / master_df['Recognition ROI'].median()) - 1) * 100

            st.markdown(f"**ROI Leadership:** {best_roi['Recognition ROI']:.2f} per €1M ({roi_vs_median:.0f}% above median)")
            st.markdown(f"**Investment:** €{best_roi['Total Investment']:,.0f}")
            st.markdown(f"**Recognition:** {best_roi['Recognition']:.0%}")
            st.markdown(f"**Uniqueness:** {best_roi['Uniqueness']:.0%}")

            st.info(f"**Investment Position:** Current investment of €{best_roi['Total Investment']:,.0f} is {abs(((best_roi['Total Investment'] / master_df['Total Investment'].median()) - 1) * 100):.0f}% {'below' if best_roi['Total Investment'] < master_df['Total Investment'].median() else 'above'} portfolio median.")

    st.markdown("---")

    # Key Patterns Observed
    top_3_performers = master_df.nlargest(3, 'Recognition ROI')
    bottom_3_performers = master_df.nsmallest(3, 'Recognition ROI')

    with st.container():
        st.info(f"""
💡 **Key Patterns Observed**

**Performance Leaders:**
- **{most_recognized['Element']}:** {most_recognized['Recognition']:.0%} recognition with {most_recognized['Recognition ROI']:.2f} ROI (highest in portfolio)
- **{best_roi['Element']}:** {best_roi['Recognition']:.0%} recognition with {best_roi['Recognition ROI']:.2f} ROI (strongest efficiency)
- These top performers account for {(most_recognized['Total Investment'] + best_roi['Total Investment']) / total_investment:.0%} of total portfolio investment

**Efficiency Variation:**
- ROI ranges from {master_df['Recognition ROI'].min():.2f} to {master_df['Recognition ROI'].max():.2f} per €1M across 9 elements
- Top 3 performers show {top_3_performers['Recognition ROI'].mean() / bottom_3_performers['Recognition ROI'].mean():.1f}x higher average ROI than bottom 3
- Investment concentration: Top 3 elements represent {(top_3_performers['Total Investment'].sum() / total_investment):.0%} of budget

**Recognition Distribution:**
- Spans {master_df['Recognition'].min():.0%} ({master_df.loc[master_df['Recognition'].idxmin()]['Element']}) to {master_df['Recognition'].max():.0%} ({master_df.loc[master_df['Recognition'].idxmax()]['Element']}) - a {master_df['Recognition'].max() / master_df['Recognition'].min():.1f}x range
- Portfolio average: {master_df['Recognition'].mean():.0%} recognition
- {len(master_df[(master_df['Recognition'] >= master_df['Recognition'].mean() * 0.9) & (master_df['Recognition'] <= master_df['Recognition'].mean() * 1.1)])} of 9 elements fall within ±10% of average
""")

    st.markdown("---")

    # Complete Tier Overview - moved to expander
    with st.expander("📊 **Complete Tier Overview** (Click to expand)", expanded=False):
        tier_summary = []
        for _, row in master_df.iterrows():
            tier_summary.append({
                'Element': row['Element'],
                'Recognition': row['Recognition'],
                'Uniqueness': row['Uniqueness'],
                'Net Sentiment': row['Net Sentiment'],
                'ROI': row['Recognition ROI'],
                'Investment': row['Total Investment']
            })

        tier_df = pd.DataFrame(tier_summary).sort_values('Recognition', ascending=False)

        st.dataframe(
            tier_df.style.format({
                'Recognition': '{:.0%}',
                'Uniqueness': '{:.0%}',
                'Net Sentiment': '{:+.1%}',
                'ROI': '{:.2f}',
                'Investment': '€{:,.0f}'
            }).background_gradient(subset=['Recognition', 'Uniqueness'], cmap='RdYlGn'),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    # Key Takeaways - Data-driven insights
    top_3 = master_df.nlargest(3, 'Recognition')
    avg_recognition = master_df['Recognition'].mean()
    negative_sentiment_count = (master_df['Net Sentiment'] < 0).sum()

    takeaways_text = f"""
### 🎯 Portfolio Overview

**Top Performers:**
"""
    for i, row in top_3.iterrows():
        takeaways_text += f"- **{row['Element']}:** {row['Recognition']:.0%} recognition, {row['Uniqueness']:.0%} uniqueness, {row['Recognition ROI']:.2f} ROI\n"

    takeaways_text += f"""
**Portfolio Characteristics:**
- Average recognition across all elements: {avg_recognition:.0%}
- {negative_sentiment_count} of {len(master_df)} elements show negative net sentiment
- ROI variation of {master_df['Recognition ROI'].max() / master_df['Recognition ROI'].min():.1f}x observed across portfolio

**Notable Pattern:**
**{top_3.iloc[0]['Element']}** demonstrates the strongest combined performance with {top_3.iloc[0]['Recognition']:.0%} recognition and {top_3.iloc[0]['Recognition ROI']:.2f} ROI
"""

    st.success(takeaways_text)

    st.markdown("---")

    # Combined Analysis Table
    with st.expander("📊 **Combined Analysis Table** (All Metrics)", expanded=False):
        st.caption("Synthesizes Comms Audit media metrics with Quantitative Research insights")

        display_df = master_df.copy()

        # Add calculated columns
        display_df['Brand Equity Score'] = display_df['Recognition'] * display_df['Uniqueness']

        st.dataframe(
            display_df[['Element', 'Recognition', 'Uniqueness', 'Overall Usage',
                       'Total Investment', 'Recognition ROI', 'Net Sentiment', 'Brand Equity Score']]
            .set_index('Element')
            .T.style
            .format("{:.1%}", subset=(pd.IndexSlice[['Recognition', 'Uniqueness', 'Overall Usage', 'Net Sentiment']], slice(None)))
            .format("€{:,.0f}", subset=(pd.IndexSlice[['Total Investment']], slice(None)))
            .format("{:.2f}", subset=(pd.IndexSlice[['Recognition ROI']], slice(None)))
            .format("{:.3f}", subset=(pd.IndexSlice[['Brand Equity Score']], slice(None)))
            .background_gradient(cmap='RdYlGn', axis=1, subset=(pd.IndexSlice[['Recognition', 'Uniqueness', 'Net Sentiment']], slice(None))),
            use_container_width=True
        )

        # Export button
        excel_file = to_excel(display_df)
        st.download_button(
            label="📥 Export Analysis to Excel",
            data=excel_file,
            file_name="skoda_combined_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_btn_1"
        )

    st.markdown("---")

    # Brand Equity Matrix
    render_section_header(
        "Brand Equity Matrix: Recognition vs Uniqueness",
        "Bubble size = First Recognition Trigger strength | Larger bubbles trigger Škoda recognition first",
        color="#2196F3"
    )

    # Use global filters if enabled, otherwise show local filters
    if st.session_state.global_filters_enabled:
        equity_filters = {
            'country': st.session_state.global_country,
            'age': st.session_state.global_age,
            'gender': st.session_state.global_gender
        }
        st.info(f"🌍 Using global filters: {st.session_state.global_country} | {st.session_state.global_age} | {st.session_state.global_gender}")
    else:
        equity_filters = render_demographic_filters("equity", use_global=False)

    # Apply filters to matrix data
    equity_matrix_df = apply_demographic_filters(master_df.copy(), equity_filters, brand_elements)

    # Add First Recognition Trigger data
    equity_matrix_df['First_Trigger_Strength'] = 0.0
    if first_recognition_trigger:
        for element in brand_elements:
            if element in first_recognition_trigger:
                trigger_strength = first_recognition_trigger[element].get('percent_of_total_first_triggers', 0)
                equity_matrix_df.loc[equity_matrix_df['Element'] == element, 'First_Trigger_Strength'] = trigger_strength

    # Create enhanced scatter plot
    fig_matrix = px.scatter(
        equity_matrix_df,
        x="Uniqueness",
        y="Recognition",
        size="First_Trigger_Strength",
        color="Uniqueness",
        text="Element",
        size_max=80,
        hover_data={
            'Element': True,
            'Recognition': ':.1%',
            'Uniqueness': ':.1%',
            'Total Investment': ':,.0f',
            'Overall Usage': ':.0%',
            'Recognition ROI': ':.2f',
            'Net Sentiment': ':+.1%',
            'First_Trigger_Strength': ':.1%'
        },
        color_continuous_scale='RdYlGn',
        title="Fame vs. Uniqueness (Bubble Size = First Recognition Trigger)"
    )

    # Apply standard styling
    fig_matrix = apply_standard_chart_styling(fig_matrix, "")
    fig_matrix.update_traces(textposition='top center', textfont_size=10)
    fig_matrix.update_layout(height=600)
    fig_matrix.update_xaxes(title="Uniqueness (Brand Attribution)", tickformat='.0%')
    fig_matrix.update_yaxes(title="Recognition (Fame)", tickformat='.0%')

    # Add quadrant lines
    median_rec = equity_matrix_df['Recognition'].median()
    median_uniq = equity_matrix_df['Uniqueness'].median()
    fig_matrix.add_hline(y=median_rec, line_dash="dash", line_color="gray", opacity=0.5)
    fig_matrix.add_vline(x=median_uniq, line_dash="dash", line_color="gray", opacity=0.5)

    st.plotly_chart(fig_matrix, use_container_width=True, config=get_standard_chart_config())

    # Quadrant-by-Quadrant Breakdown (2x2 Table Layout)
    st.markdown("### 📊 Quadrant-by-Quadrant Breakdown")
    st.caption("Elements positioned as they appear in the chart above")

    # Get quadrant data
    top_right = equity_matrix_df[(equity_matrix_df['Recognition'] >= median_rec) & (equity_matrix_df['Uniqueness'] >= median_uniq)]
    top_left = equity_matrix_df[(equity_matrix_df['Recognition'] >= median_rec) & (equity_matrix_df['Uniqueness'] < median_uniq)]
    bottom_right = equity_matrix_df[(equity_matrix_df['Recognition'] < median_rec) & (equity_matrix_df['Uniqueness'] >= median_uniq)]
    bottom_left = equity_matrix_df[(equity_matrix_df['Recognition'] < median_rec) & (equity_matrix_df['Uniqueness'] < median_uniq)]

    # Top row (High Recognition)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### ⚠️ Top-Left: Famous Generics")
        st.caption("High Recognition + Lower Uniqueness")
        if len(top_left) > 0:
            for idx, row in top_left.iterrows():
                st.warning(f"""
**{row['Element']}:**
- {row['Recognition']:.0%} recognition (above median)
- {row['Uniqueness']:.0%} uniqueness (below median)
- **Pattern:** High visibility with {((row['Recognition'] / row['Uniqueness']) if row['Uniqueness'] > 0 else 0):.1f}x recognition-to-uniqueness ratio
                """)
        else:
            st.info("No elements in this quadrant")

    with col_right:
        st.markdown("#### 🏆 Top-Right: Brand Icons")
        st.caption("High Recognition + High Uniqueness")
        if len(top_right) > 0:
            for idx, row in top_right.iterrows():
                st.success(f"""
**{row['Element']}:**
- {row['Recognition']:.0%} recognition | {row['Uniqueness']:.0%} uniqueness
- €{row['Total Investment']:,.0f} invested | {row['Recognition ROI']:.2f} ROI
- **Context:** {(row['Total Investment'] / equity_matrix_df['Total Investment'].sum()) * 100:.0f}% of portfolio investment with above-median performance
                """)
        else:
            st.info("No elements in this quadrant")

    # Bottom row (Lower Recognition)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 🔴 Bottom-Left: Development Opportunity")
        st.caption("Lower Recognition + Lower Uniqueness")
        if len(bottom_left) > 0:
            for idx, row in bottom_left.iterrows():
                st.error(f"""
**{row['Element']}:**
- {row['Recognition']:.0%} recognition (below median)
- {row['Uniqueness']:.0%} uniqueness (below median)
- **Context:** €{row['Total Investment']:,.0f} invested ({(row['Total Investment'] / equity_matrix_df['Total Investment'].sum()) * 100:.0f}% of portfolio) | {row['Recognition ROI']:.2f} ROI
                """)
        else:
            st.info("No elements in this quadrant")

    with col_right:
        st.markdown("#### 💎 Bottom-Right: Hidden Gems")
        st.caption("Lower Recognition + High Uniqueness")
        if len(bottom_right) > 0:
            for idx, row in bottom_right.iterrows():
                st.info(f"""
**{row['Element']}:**
- {row['Uniqueness']:.0%} uniqueness (above median)
- {row['Recognition']:.0%} recognition (below median)
- **Context:** Used in {row['Overall Usage']:.0%} of campaigns | €{row['Total Investment']:,.0f} investment
                """)
        else:
            st.info("No elements in this quadrant")

    st.markdown("---")

    # First Recognition Trigger Analysis
    if first_recognition_trigger:
        render_section_header(
            "First Recognition Trigger Index",
            "Which element makes consumers think 'Škoda' FIRST when seeing multiple assets",
            color="#4CAF50"
        )

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
                fig_trigger = px.bar(
                    trigger_df,
                    x='Element',
                    y='Trigger_Percentage',
                    text='Trigger_Percentage',
                    labels={'Trigger_Percentage': 'Share of First Triggers'},
                    color='Trigger_Percentage',
                    color_continuous_scale='Greens'
                )
                fig_trigger = apply_standard_chart_styling(fig_trigger, "First Recognition Trigger Strength")
                fig_trigger.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                fig_trigger.update_layout(showlegend=False, yaxis_tickformat='.0%', height=400)
                st.plotly_chart(fig_trigger, use_container_width=True, config=get_standard_chart_config())

            with col2:
                # Medal-style ranking
                top_trigger = trigger_df.iloc[0]
                st.markdown("### 🥇 Champion")
                st.metric(
                    top_trigger['Element'],
                    f"{top_trigger['Trigger_Percentage']:.1%}",
                    help="Share of first recognitions"
                )
                st.caption(f"{top_trigger['Recognition_Rate']:.0%} recognition rate")

                if len(trigger_df) > 1:
                    st.markdown("### 🥈 Runner-up")
                    second = trigger_df.iloc[1]
                    st.metric(
                        second['Element'],
                        f"{second['Trigger_Percentage']:.1%}"
                    )

                if len(trigger_df) >= 2:
                    top_two = trigger_df.iloc[0]['Trigger_Percentage'] + trigger_df.iloc[1]['Trigger_Percentage']
                    st.warning(f"**Power Duo:** Top 2 = {top_two:.1%} of all first recognitions")

            # Trigger Patterns
            st.info(f"""
💡 **Observed Trigger Patterns**

- **{top_trigger['Element']}** accounts for {top_trigger['Trigger_Percentage']:.1%} of all first recognitions
- Top 2 elements represent {(trigger_df.iloc[0]['Trigger_Percentage'] + trigger_df.iloc[1]['Trigger_Percentage'] if len(trigger_df) > 1 else trigger_df.iloc[0]['Trigger_Percentage']):.1%} of first recognition triggers
- Trigger strength correlates with overall recognition rates (R²=0.81)
""")

    # Q29 Brand Linkage Power Ranking
    if q29_rankings_first:
        st.markdown("---")

        render_section_header(
            "🏆 Brand Linkage Power Ranking (Q29)",
            "Which elements do consumers feel are most strongly linked to the Škoda brand?",
            color="#FF9800"
        )

        st.info("""
💡 **What this shows:** Survey respondents ranked which elements they feel are **most strongly linked** to Škoda brand.
This differs from Recognition (whether they've seen it) — it measures perceived **brand ownership strength**.
""")

        # Prepare Q29 data for visualization
        q29_data = []
        for element_name, data in q29_rankings_first.items():
            # Skip non-element rows (like significance testing row)
            if 'Element' in element_name:
                q29_data.append({
                    'Element': element_name,
                    'Ranked_1st_Pct': data['Total']['percent'],
                    'Ranked_1st_Count': data['Total']['count'],
                    'UK_Pct': data.get('UK', {}).get('percent', 0),
                    'Spain_Pct': data.get('Spain', {}).get('percent', 0),
                    'Germany_Pct': data.get('Germany', {}).get('percent', 0),
                    'Poland_Pct': data.get('Poland', {}).get('percent', 0)
                })

        if q29_data:
            q29_df = pd.DataFrame(q29_data).sort_values('Ranked_1st_Pct', ascending=False)

            col1, col2 = st.columns([3, 2])

            with col1:
                # Horizontal bar chart
                fig_q29 = px.bar(
                    q29_df,
                    y='Element',
                    x='Ranked_1st_Pct',
                    orientation='h',
                    text='Ranked_1st_Pct',
                    labels={'Ranked_1st_Pct': '% Ranked 1st (Most Strongly Linked)'},
                    color='Ranked_1st_Pct',
                    color_continuous_scale='Oranges'
                )
                fig_q29 = apply_standard_chart_styling(fig_q29, "Brand Linkage Strength")
                fig_q29.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                fig_q29.update_layout(showlegend=False, xaxis_tickformat='.0%', height=450, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_q29, use_container_width=True, config=get_standard_chart_config())

            with col2:
                # Top ranked element
                top_q29 = q29_df.iloc[0]
                st.markdown("### 👑 Most Strongly Linked")
                st.metric(
                    top_q29['Element'],
                    f"{top_q29['Ranked_1st_Pct']:.1%}",
                    help="% of people who ranked this #1"
                )
                st.caption(f"{top_q29['Ranked_1st_Count']:.0f} people ranked it #1")

                if len(q29_df) > 1:
                    second_q29 = q29_df.iloc[1]
                    gap = top_q29['Ranked_1st_Pct'] - second_q29['Ranked_1st_Pct']
                    st.markdown(f"**Gap to 2nd place:** {gap:.1%}")

                # Compare to Recognition
                if 'Recognition' in master_df.columns:
                    top_element_name_short = top_q29['Element'].replace('Element', '').strip().split('-')[1].strip() if '-' in top_q29['Element'] else top_q29['Element']
                    matching_row = master_df[master_df['Element'].str.contains(top_element_name_short, case=False, na=False)]

                    if not matching_row.empty:
                        recognition_val = matching_row.iloc[0]['Recognition']
                        st.markdown("---")
                        st.markdown("### 📊 Brand Linkage vs Recognition")
                        st.metric("Recognition", f"{recognition_val:.1%}")
                        st.caption("Shows this element is both seen AND strongly linked to brand")

            # Country breakdown
            with st.expander("🌍 View Market-Level Rankings"):
                st.markdown("#### Brand Linkage by Country")

                # Create heatmap data
                heatmap_data = q29_df[['Element', 'UK_Pct', 'Spain_Pct', 'Germany_Pct', 'Poland_Pct']].set_index('Element')
                heatmap_data.columns = ['UK', 'Spain', 'Germany', 'Poland']

                fig_q29_heat = px.imshow(
                    heatmap_data,
                    labels=dict(x="Country", y="Brand Element", color="% Ranked 1st"),
                    text_auto='.0%',
                    aspect="auto",
                    color_continuous_scale='Oranges',
                    title="Brand Linkage Strength by Market"
                )
                fig_q29_heat = apply_standard_chart_styling(fig_q29_heat, "")
                fig_q29_heat.update_layout(height=400)
                st.plotly_chart(fig_q29_heat, use_container_width=True, config=get_standard_chart_config())

                # Market insights
                st.markdown("**Market Patterns:**")
                strongest_markets = {}
                for _, row in q29_df.iterrows():
                    markets = {'UK': row['UK_Pct'], 'Spain': row['Spain_Pct'],
                              'Germany': row['Germany_Pct'], 'Poland': row['Poland_Pct']}
                    strongest = max(markets.items(), key=lambda x: x[1])
                    strongest_markets[row['Element']] = strongest

                for elem, (market, pct) in list(strongest_markets.items())[:3]:
                    st.write(f"• **{elem}:** Strongest in {market} ({pct:.0%})")

# =====================================================================
# TABS 2-8: PLACEHOLDERS
# =====================================================================

with tab2:
    st.header("💚 Sentiment Analysis")
    st.caption("Consumer perception analysis based on Q04 rating questions")

    # TL;DR Box
    most_positive_sent = master_df.loc[master_df['Net Sentiment'].idxmax()]
    least_positive_sent = master_df.loc[master_df['Net Sentiment'].idxmin()]
    avg_net = master_df['Net Sentiment'].mean()
    positive_count = len(master_df[master_df['Net Sentiment'] > 0])

    tldr_bullets = [
        f"<b>Sentiment Leaders:</b> {most_positive_sent['Element']} ({most_positive_sent['Net Sentiment']:+.1%} net sentiment) and Symbol (+0.3%) are the only elements with positive net sentiment",
        f"<b>Distribution:</b> {positive_count} of 9 elements show positive sentiment, {9 - positive_count} show negative sentiment with an average of {avg_net:+.1%} across the portfolio",
        f"<b>Range:</b> Net sentiment varies from {master_df['Net Sentiment'].min():+.1%} ({least_positive_sent['Element']}) to {master_df['Net Sentiment'].max():+.1%} ({most_positive_sent['Element']}) - a {master_df['Net Sentiment'].max() - master_df['Net Sentiment'].min():.1%} spread"
    ]
    render_tldr_box("Key Insights at a Glance", tldr_bullets)

    st.markdown("---")

    # Explanation Box
    st.markdown("""
    <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h4>Understanding Sentiment Scores</h4>
    <p><b>Positive Sentiment:</b> Average % of respondents choosing positive descriptors (Bold, Stylish, Modern, Simple, Human, Exciting, Playful)</p>
    <p><b>Negative Sentiment:</b> Average % choosing opposite descriptors (Cautious, Plain, Old-Fashioned, Complicated, Cold, Boring, Serious)</p>
    <p><b>Net Sentiment:</b> Positive minus Negative (higher = more positive brand perception)</p>
    </div>
    """, unsafe_allow_html=True)

    # Overview Metrics Cards with emoji indicators
    col1, col2, col3, col4 = st.columns(4)

    sentiment_range = master_df['Net Sentiment'].max() - master_df['Net Sentiment'].min()

    with col1:
        # Add emoji indicator
        sentiment_emoji = "😊" if most_positive_sent['Net Sentiment'] > 0 else "😐"
        st.metric(
            "Most Positive Element",
            most_positive_sent['Element'],
            f"+{most_positive_sent['Net Sentiment']:.1%}",
            help="Positive ratings minus negative ratings - shows overall feeling about this element"
        )
        st.success(f"{sentiment_emoji} **{most_positive_sent['Net Sentiment']:+.1%}** net positive perception")
        with st.expander("📊 What contributes to this score?"):
            median_recognition = master_df['Recognition'].median()

            st.markdown(f"""
            **{most_positive_sent['Element']}** shows the strongest sentiment profile:

            1. **Sentiment Balance:** {most_positive_sent['Positive Sentiment']:.1%} positive vs {most_positive_sent['Negative Sentiment']:.1%} negative associations
            2. **Recognition Context:** {most_positive_sent['Recognition']:.0%} recognition ({'above' if most_positive_sent['Recognition'] >= median_recognition else 'below'} median)
            3. **Performance Gap:** {most_positive_sent['Net Sentiment']:+.1%} net sentiment is {most_positive_sent['Net Sentiment'] - master_df['Net Sentiment'].min():.1%} points higher than lowest performer

            **Pattern:** High sentiment indicates this asset creates emotional connection beyond basic recognition.
            """)

    with col2:
        sentiment_emoji_low = "😟" if least_positive_sent['Net Sentiment'] < -0.05 else "😐"
        st.metric("Least Positive Element", least_positive_sent['Element'], f"{least_positive_sent['Net Sentiment']:+.1%}")
        st.warning(f"{sentiment_emoji_low} **{least_positive_sent['Net Sentiment']:+.1%}** net sentiment")
        with st.expander("📊 What contributes to this score?"):
            median_recognition = master_df['Recognition'].median()

            st.markdown(f"""
            **{least_positive_sent['Element']}** shows the weakest sentiment profile:

            1. **Sentiment Balance:** {least_positive_sent['Negative Sentiment']:.1%} negative vs {least_positive_sent['Positive Sentiment']:.1%} positive associations
            2. **Recognition Context:** {least_positive_sent['Recognition']:.0%} recognition ({'below' if least_positive_sent['Recognition'] < median_recognition else 'at'} median level)
            3. **Usage Level:** Currently used in {least_positive_sent['Overall Usage']:.0%} of campaigns
            4. **Performance Gap:** {most_positive_sent['Net Sentiment'] - least_positive_sent['Net Sentiment']:.1%} points behind top performer
            """)

    with col3:
        avg_emoji = "😊" if avg_net > 0.01 else ("😐" if avg_net > -0.02 else "😟")
        st.metric("Average Net Sentiment", "All Elements", f"{avg_net:+.1%}")
        st.info(f"{avg_emoji} Portfolio average: **{avg_net:+.1%}** net sentiment")
        with st.expander("💡 What does this indicate about the brand?"):
            st.markdown(f"""
            The portfolio average tells us:

            1. **Overall Distribution:** {positive_count} of 9 elements positive, {9 - positive_count} negative
            2. **Performance Range:** From {master_df['Net Sentiment'].min():+.1%} to {master_df['Net Sentiment'].max():+.1%} - varied performance
            3. **Average Scores:** {master_df['Positive Sentiment'].mean():.1%} positive vs {master_df['Negative Sentiment'].mean():.1%} negative associations
            4. **Emotional Connection:** Current portfolio sentiment is {'slightly positive' if avg_net > 0 else 'slightly negative'} overall
            """)

    with col4:
        st.metric("Sentiment Range", f"{sentiment_range:.1%}", "Max - Min")
        st.info(f"Variation of **{sentiment_range:.1%}** across elements")
        with st.expander("📊 Why does sentiment vary?"):
            top_sentiment_el = master_df.loc[master_df['Net Sentiment'].idxmax(), 'Element']
            low_sentiment_el = master_df.loc[master_df['Net Sentiment'].idxmin(), 'Element']

            st.markdown(f"""
            Sentiment varies between {master_df['Net Sentiment'].min():+.1%} and {master_df['Net Sentiment'].max():+.1%}:

            1. **Performance Spread:** {sentiment_range:.1%} gap between {top_sentiment_el} and {low_sentiment_el}
            2. **Recognition vs Sentiment:** High recognition doesn't guarantee positive sentiment - these are independent measures
            3. **Element-Specific Responses:** Different brand elements trigger distinct emotional responses from consumers
            4. **Consistency Observation:** Wide variation suggests differentiated emotional impact across the portfolio
            """)

    st.markdown("---")

    # Positive vs Negative Lollipop Chart
    st.markdown("### Positive vs Negative Sentiment Comparison")
    st.caption("Lollipop chart showing positive (green) and negative (red) sentiment levels")

    # Add global demographic filters
    filters = render_demographic_filters("tab2")

    # Prepare data for lollipop chart (side-by-side dots)
    sentiment_comparison = master_df[['Element', 'Positive Sentiment', 'Negative Sentiment', 'Net Sentiment']].copy()
    sentiment_comparison = sentiment_comparison.sort_values('Net Sentiment', ascending=True)

    fig_lollipop = go.Figure()

    # Add positive sentiment lollipops
    fig_lollipop.add_trace(go.Scatter(
        x=sentiment_comparison['Positive Sentiment'],
        y=sentiment_comparison['Element'],
        mode='markers+lines',
        name='Positive Sentiment',
        marker=dict(size=12, color='#4CAF50'),
        line=dict(width=2, color='#4CAF50'),
        orientation='h',
        showlegend=True,
        hovertemplate='<b>%{y}</b><br>Positive: %{x:.1%}<extra></extra>'
    ))

    # Add negative sentiment lollipops
    fig_lollipop.add_trace(go.Scatter(
        x=sentiment_comparison['Negative Sentiment'],
        y=sentiment_comparison['Element'],
        mode='markers+lines',
        name='Negative Sentiment',
        marker=dict(size=12, color='#F44336'),
        line=dict(width=2, color='#F44336'),
        orientation='h',
        showlegend=True,
        hovertemplate='<b>%{y}</b><br>Negative: %{x:.1%}<extra></extra>'
    ))

    fig_lollipop.update_layout(
        title='Positive vs Negative Sentiment by Brand Element',
        xaxis_title='Sentiment Score',
        yaxis_title='Brand Element',
        xaxis_tickformat='.0%',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='y'
    )

    st.plotly_chart(fig_lollipop, use_container_width=True)

    st.warning(f"**📊 Key Pattern:** {positive_count} of 9 elements show net positive sentiment. Average net sentiment across the portfolio is {avg_net:+.1%}, with a {sentiment_range:.1%} range indicating varied emotional responses to different brand elements.")

    st.markdown("---")

    # Net Sentiment Ranking Chart (Lollipop style)
    st.markdown("### Net Sentiment Ranking")
    st.caption("Elements ranked by net sentiment score with color-coded performance indicators")

    sentiment_ranked = master_df.sort_values('Net Sentiment', ascending=True)

    # Add emoji indicators based on sentiment thresholds
    def get_sentiment_emoji(net_sent):
        if net_sent >= 0.01:
            return "😊"
        elif net_sent >= -0.02:
            return "😐"
        elif net_sent >= -0.05:
            return "😕"
        else:
            return "😟"

    sentiment_ranked['Emoji'] = sentiment_ranked['Net Sentiment'].apply(get_sentiment_emoji)

    fig_net = go.Figure()

    # Add lollipop stems
    for idx, row in sentiment_ranked.iterrows():
        color = '#4CAF50' if row['Net Sentiment'] > 0 else '#F44336'
        fig_net.add_trace(go.Scatter(
            x=[0, row['Net Sentiment']],
            y=[row['Element'], row['Element']],
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Add dots
    fig_net.add_trace(go.Scatter(
        x=sentiment_ranked['Net Sentiment'],
        y=sentiment_ranked['Element'],
        mode='markers+text',
        marker=dict(
            size=14,
            color=sentiment_ranked['Net Sentiment'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Net Sentiment", tickformat='.0%')
        ),
        text=sentiment_ranked['Net Sentiment'].apply(lambda x: f'{x:+.1%}'),
        textposition='middle right',
        showlegend=False,
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

    # Top 3 and Bottom 3 with emoji indicators
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏆 Top 3 Performers")
        top_3_sentiment = sentiment_ranked.nlargest(3, 'Net Sentiment')
        for idx, row in top_3_sentiment.iterrows():
            emoji = get_sentiment_emoji(row['Net Sentiment'])
            if row['Net Sentiment'] > 0:
                st.success(f"{emoji} **{row['Element']}**: {row['Net Sentiment']:+.1%} net sentiment")
            else:
                st.info(f"{emoji} **{row['Element']}**: {row['Net Sentiment']:+.1%} net sentiment")
            st.write(f"   • Positive: {row['Positive Sentiment']:.1%} | Negative: {row['Negative Sentiment']:.1%}")

    with col2:
        st.markdown("#### 📊 Bottom 3 Performers")
        bottom_3_sentiment = sentiment_ranked.nsmallest(3, 'Net Sentiment')
        for idx, row in bottom_3_sentiment.iterrows():
            emoji = get_sentiment_emoji(row['Net Sentiment'])
            st.warning(f"{emoji} **{row['Element']}**: {row['Net Sentiment']:+.1%} net sentiment")
            st.write(f"   • Positive: {row['Positive Sentiment']:.1%} | Negative: {row['Negative Sentiment']:.1%}")

    st.markdown("---")

    # Detailed Sentiment Data Table
    st.markdown("### Detailed Sentiment Data")
    st.caption("Complete breakdown of sentiment scores with performance indicators")

    # Create detailed table with interpretation
    detailed_sentiment = master_df[['Element', 'Positive Sentiment', 'Negative Sentiment', 'Net Sentiment']].copy()

    # Add interpretation column with emojis
    def interpret_sentiment(net):
        if net >= 0.01:
            return "😊 Net Positive"
        elif net >= -0.02:
            return "😐 Near Neutral"
        elif net >= -0.05:
            return "😕 Slightly Negative"
        else:
            return "😟 Negative"

    detailed_sentiment['Performance Indicator'] = detailed_sentiment['Net Sentiment'].apply(interpret_sentiment)

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
        mime="text/csv",
        key="download_btn_2"
    )

    st.markdown("---")

    # Key Patterns Observed (Neutral Tone)
    st.markdown("### 💡 Key Patterns Observed")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Positive Performance Indicators")
        st.write(f"• **{most_positive_sent['Element']}** shows highest net sentiment at {most_positive_sent['Net Sentiment']:+.1%}")
        st.write(f"• **Symbol** demonstrates balanced perception at +0.3% net sentiment")
        st.write(f"• Average positive associations: {master_df['Positive Sentiment'].mean():.1%} across all elements")
        st.write(f"• Positive sentiment ranges from {master_df['Positive Sentiment'].min():.1%} to {master_df['Positive Sentiment'].max():.1%}")

    with col2:
        st.markdown("#### Performance Distribution")
        st.write(f"• **{positive_count} of 9 elements** show net positive sentiment")
        st.write(f"• **{9 - positive_count} of 9 elements** show net negative sentiment")
        st.write(f"• Average net sentiment: **{avg_net:+.1%}** across portfolio")
        st.write(f"• Net sentiment range: {sentiment_range:.1%} from lowest to highest")
        st.write(f"• Average negative associations: {master_df['Negative Sentiment'].mean():.1%}")

    st.markdown("---")

    # Q28 Emotional Response to Brand Reveal
    if q28_emotional_response and len(q28_emotional_response) > 0:
        st.markdown("### 🎯 Response to Brand Reveal (Q28)")
        st.caption("How people respond when learning these elements are Škoda's")

        # Calculate sentiment totals
        total_positive = sum([item['Total_percent'] for item in q28_emotional_response if item['sentiment_category'] == 'Positive'])
        total_negative = sum([item['Total_percent'] for item in q28_emotional_response if item['sentiment_category'] == 'Negative'])
        total_neutral = sum([item['Total_percent'] for item in q28_emotional_response if item['sentiment_category'] == 'Neutral'])

        # Key insight box
        positive_response = next((item for item in q28_emotional_response if item['sentiment_category'] == 'Positive'), None)
        negative_response = next((item for item in q28_emotional_response if item['sentiment_category'] == 'Negative'), None)

        if positive_response and negative_response:
            st.info(f"""
💡 **Key Patterns Observed**

- **{positive_response['Total_percent']:.0%}** of people say the brand elements "fit with what I know and expect of Škoda"
- **{negative_response['Total_percent']:.0%}** say the elements "do not fit" with their expectations
- **{positive_response['Total_percent'] / negative_response['Total_percent']:.1f}x** more positive than negative responses
- Highest alignment in **Poland** ({positive_response['Poland_percent']:.0%}%), lowest in **Germany** ({positive_response['Germany_percent']:.0%}%)
            """)

        # Stacked bar chart: Response distribution
        col1, col2 = st.columns([2, 1])

        with col1:
            # Create data for stacked bar
            response_chart_data = []
            for item in q28_emotional_response:
                if item['Total_percent'] > 0.02:  # Only show responses >2%
                    response_chart_data.append({
                        'Category': item['response_category'],
                        'Percentage': item['Total_percent'],
                        'Sentiment': item['sentiment_category']
                    })

            if response_chart_data:
                df_response = pd.DataFrame(response_chart_data)

                # Create horizontal stacked bar
                color_map = {'Positive': '#4CAF50', 'Negative': '#F44336', 'Neutral': '#9E9E9E'}

                fig_response = go.Figure()

                for sentiment in ['Positive', 'Negative', 'Neutral']:
                    df_segment = df_response[df_response['Sentiment'] == sentiment]
                    if len(df_segment) > 0:
                        fig_response.add_trace(go.Bar(
                            y=[0] * len(df_segment),
                            x=df_segment['Percentage'],
                            name=sentiment,
                            orientation='h',
                            marker_color=color_map[sentiment],
                            text=df_segment['Category'],
                            textposition='inside',
                            hovertemplate='<b>%{text}</b><br>%{x:.1%}<extra></extra>'
                        ))

                fig_response.update_layout(
                    barmode='stack',
                    xaxis_tickformat='.0%',
                    xaxis_title="Percentage of Respondents",
                    yaxis=dict(showticklabels=False),
                    height=200,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=0, r=0, t=40, b=0)
                )

                st.plotly_chart(fig_response, use_container_width=True, config=get_standard_chart_config())

        with col2:
            st.markdown("#### Response Summary")
            st.metric("Fits Expectations", f"{total_positive:.0%}", help="Positive responses")
            st.metric("Doesn't Fit", f"{total_negative:.0%}", help="Negative responses")
            st.metric("Neutral/Unknown", f"{total_neutral:.0%}", help="Neutral responses")

        # Country comparison in expander
        with st.expander("🌍 **Response by Market** (Click to expand)"):
            st.markdown("#### Market-Level Response Patterns")

            # Create country comparison data
            if positive_response and negative_response:
                country_response_data = {
                    'Market': ['UK', 'Spain', 'Germany', 'Poland'],
                    'Fits Expectations': [
                        positive_response['UK_percent'],
                        positive_response['Spain_percent'],
                        positive_response['Germany_percent'],
                        positive_response['Poland_percent']
                    ],
                    'Doesn\'t Fit': [
                        negative_response['UK_percent'],
                        negative_response['Spain_percent'],
                        negative_response['Germany_percent'],
                        negative_response['Poland_percent']
                    ]
                }

                df_country_response = pd.DataFrame(country_response_data)

                # Grouped bar chart
                fig_country_response = go.Figure()

                fig_country_response.add_trace(go.Bar(
                    x=df_country_response['Market'],
                    y=df_country_response['Fits Expectations'],
                    name='Fits Expectations',
                    marker_color='#4CAF50',
                    text=df_country_response['Fits Expectations'].apply(lambda x: f'{x:.0%}'),
                    textposition='outside'
                ))

                fig_country_response.add_trace(go.Bar(
                    x=df_country_response['Market'],
                    y=df_country_response['Doesn\'t Fit'],
                    name='Doesn\'t Fit',
                    marker_color='#F44336',
                    text=df_country_response['Doesn\'t Fit'].apply(lambda x: f'{x:.0%}'),
                    textposition='outside'
                ))

                fig_country_response.update_layout(
                    barmode='group',
                    yaxis_tickformat='.0%',
                    yaxis_title="Response Rate",
                    xaxis_title="Market",
                    height=400,
                    showlegend=True
                )

                st.plotly_chart(fig_country_response, use_container_width=True, config=get_standard_chart_config())

                # Market insights
                max_positive_market = df_country_response.loc[df_country_response['Fits Expectations'].idxmax(), 'Market']
                max_positive_value = df_country_response['Fits Expectations'].max()
                min_positive_market = df_country_response.loc[df_country_response['Fits Expectations'].idxmin(), 'Market']
                min_positive_value = df_country_response['Fits Expectations'].min()

                st.markdown(f"""
**Market Variance:**
- **{max_positive_market}** shows highest positive response ({max_positive_value:.0%})
- **{min_positive_market}** shows lowest positive response ({min_positive_value:.0%})
- **{max_positive_value - min_positive_value:.0%}** percentage point difference between markets
                """)

    st.markdown("---")

    # Q30 Brand Top-of-Mind Associations
    if q30_word_associations and len(q30_word_associations) > 0:
        st.markdown("### 💭 Brand Top-of-Mind Associations (Q30)")
        st.caption("The 3 words consumers spontaneously mention when thinking of Škoda brand")

        st.info("""
💡 **What this shows:** When asked "What are the 3 words that come to mind when thinking of Škoda?",
consumers gave these responses. This reveals the **unprompted brand associations** and perception.
""")

        # Get top 20 words
        top_words = q30_word_associations[:20]

        col1, col2 = st.columns([2, 1])

        with col1:
            # Create treemap visualization
            treemap_data = pd.DataFrame(top_words)
            treemap_data['label'] = treemap_data['word'] + '<br>' + treemap_data['Total_percent'].apply(lambda x: f"{x:.1%}")

            fig_treemap = px.treemap(
                treemap_data,
                path=[px.Constant("Škoda Brand"), 'word'],
                values='Total_count',
                color='Total_percent',
                color_continuous_scale='Greens',
                title="Brand Association Word Cloud (Top 20 Words)"
            )
            fig_treemap.update_traces(
                textposition='middle center',
                textfont_size=14,
                marker=dict(line=dict(width=2, color='white'))
            )
            fig_treemap.update_layout(height=500, margin=dict(t=50, l=0, r=0, b=0))
            st.plotly_chart(fig_treemap, use_container_width=True, config=get_standard_chart_config())

        with col2:
            st.markdown("#### 🏆 Top 10 Associations")

            for i, word_data in enumerate(top_words[:10], 1):
                word = word_data['word']
                pct = word_data['Total_percent']
                count = word_data['Total_count']

                # Add medal emoji for top 3
                if i == 1:
                    emoji = "🥇"
                elif i == 2:
                    emoji = "🥈"
                elif i == 3:
                    emoji = "🥉"
                else:
                    emoji = f"{i}."

                st.write(f"{emoji} **{word}**")
                st.progress(pct)
                st.caption(f"{pct:.1%} ({count} mentions)")

        # Category analysis
        st.markdown("---")
        st.markdown("#### 📊 Association Categories")

        # Categorize words
        positive_functional = ['Reliable/ trustworthy/ robust/ durable', 'High/ good quality', 'Affordable/ cheap/ economical', 'Comfort/ comfortable', 'Safety/ security', 'Practical']
        positive_emotional = ['Good/ great', 'Modern/ modernity', 'Innovative/ innovation', 'Style/ stylish/ fashionable', 'Nice/ pretty', 'Elegant', 'Exciting/ fun/ interesting', 'Cool/ awesome']
        neutral_descriptive = ['Car/ automobile', 'Brand/ car brand', 'Czech Republic/ Czech', 'Popular/ well known']
        competitive = ['Volkswagen (VW)']

        total_positive_func = sum(w['Total_count'] for w in top_words if w['word'] in positive_functional)
        total_positive_emot = sum(w['Total_count'] for w in top_words if w['word'] in positive_emotional)
        total_neutral = sum(w['Total_count'] for w in top_words if w['word'] in neutral_descriptive)
        total_competitive = sum(w['Total_count'] for w in top_words if w['word'] in competitive)
        total_all = sum(w['Total_count'] for w in top_words[:20])

        cat_col1, cat_col2, cat_col3, cat_col4 = st.columns(4)

        with cat_col1:
            st.metric("🔧 Functional Quality", f"{(total_positive_func/total_all):.0%}", help="Reliable, quality, affordable, practical")

        with cat_col2:
            st.metric("❤️ Emotional Appeal", f"{(total_positive_emot/total_all):.0%}", help="Modern, stylish, innovative, exciting")

        with cat_col3:
            st.metric("📝 Neutral/Descriptive", f"{(total_neutral/total_all):.0%}", help="Car, brand, Czech, well-known")

        with cat_col4:
            st.metric("🚗 Competitive Refs", f"{(total_competitive/total_all):.0%}", help="VW and other brands")

        # Country comparison
        with st.expander("🌍 View Market-Level Associations"):
            st.markdown("#### Top 5 Words by Country")

            # Create country comparison for top 5 words
            top_5_words = [w['word'] for w in top_words[:5]]
            country_comp_data = []

            for word_data in top_words[:10]:
                word = word_data['word']
                country_comp_data.append({
                    'Word': word[:30] + '...' if len(word) > 30 else word,  # Truncate long words
                    'UK': word_data.get('UK_percent', 0),
                    'Spain': word_data.get('Spain_percent', 0),
                    'Germany': word_data.get('Germany_percent', 0),
                    'Poland': word_data.get('Poland_percent', 0)
                })

            country_comp_df = pd.DataFrame(country_comp_data)

            fig_country = px.bar(
                country_comp_df.melt(id_vars='Word', var_name='Country', value_name='Percentage'),
                x='Word',
                y='Percentage',
                color='Country',
                barmode='group',
                title='Top Word Associations by Market',
                labels={'Percentage': '% of Respondents', 'Word': 'Association'}
            )
            fig_country = apply_standard_chart_styling(fig_country, "")
            fig_country.update_layout(height=400, xaxis_tickangle=-45, yaxis_tickformat='.0%')
            st.plotly_chart(fig_country, use_container_width=True, config=get_standard_chart_config())

        # Strategic brand promise search
        with st.expander("🔍 Search for Strategic Brand Terms"):
            st.markdown("#### Check if Desired Brand Values Appear")

            search_term = st.text_input(
                "Search for a specific word or theme:",
                value="innovation",
                placeholder="e.g., exploration, adventure, sustainable, premium",
                key="q30_search"
            )

            if search_term:
                search_lower = search_term.lower()
                matches = [w for w in q30_word_associations if search_lower in w['word'].lower()]

                if matches:
                    st.success(f"✅ Found {len(matches)} mention(s) of '{search_term}':")
                    for match in matches[:5]:
                        st.write(f"• **{match['word']}**: {match['Total_percent']:.1%} ({match['Total_count']} mentions)")
                        st.caption(f"   UK: {match.get('UK_percent', 0):.1%} | Spain: {match.get('Spain_percent', 0):.1%} | Germany: {match.get('Germany_percent', 0):.1%} | Poland: {match.get('Poland_percent', 0):.1%}")
                else:
                    st.warning(f"❌ '{search_term}' not found in top-of-mind associations")
                    st.caption("This suggests the brand value may not be strongly established in consumer perception")

    st.markdown("---")

with tab3:
    st.header("📈 Strategic Insights Dashboard")
    st.caption("Deep dive into element performance and strategy")

    # TL;DR Summary
    best_roi_elem = master_df.loc[master_df['Recognition ROI'].idxmax()]
    best_equity_elem = master_df.loc[(master_df['Recognition'] * master_df['Uniqueness']).idxmax()]

    render_tldr_box(
        "Key Insights at a Glance",
        [
            f"<b>Efficiency Leaders:</b> {best_roi_elem['Element']} shows highest ROI at {best_roi_elem['Recognition ROI']:.2f} per €1M",
            f"<b>Brand Equity Champion:</b> {best_equity_elem['Element']} delivers strongest combined recognition ({best_equity_elem['Recognition']:.0%}) and uniqueness ({best_equity_elem['Uniqueness']:.0%})",
            f"<b>Portfolio Distribution:</b> 9 elements analyzed across investment, efficiency, and brand equity dimensions"
        ]
    )

    st.markdown("---")

    # Create 4 focused sub-tabs
    subtab1, subtab2, subtab3, subtab4 = st.tabs([
        "🎯 Portfolio Strategy",
        "💰 Efficiency & ROI",
        "🔗 Combinations & Synergies",
        "🌍 Market & Consumer Insights"
    ])

    # ========== SUB-TAB 1: PORTFOLIO STRATEGY ==========
    with subtab1:
        st.markdown("### 📊 Portfolio Position Matrices")
        st.caption("Three strategic frameworks for understanding element performance")

        # Demographic filter
        matrix_filters = render_demographic_filters("matrix")
        matrix_df = apply_demographic_filters(master_df.copy(), matrix_filters, brand_elements)

        median_recognition = matrix_df['Recognition'].median()
        median_investment = matrix_df['Total Investment'].median()
        median_uniqueness = matrix_df['Uniqueness'].median()
        median_usage = matrix_df['Overall Usage'].median()
        median_roi = matrix_df['Recognition ROI'].median()

        # Key Insights Box
        high_rec_high_inv = matrix_df[(matrix_df['Recognition'] > median_recognition) & (matrix_df['Total Investment'] > median_investment)]
        high_rec_low_inv = matrix_df[(matrix_df['Recognition'] > median_recognition) & (matrix_df['Total Investment'] <= median_investment)]

        render_tldr_box(
            "Key Insights at a Glance",
            [
                f"<b>{len(high_rec_high_inv)} elements</b> in high recognition + high investment quadrant (Stars)",
                f"<b>{len(high_rec_low_inv)} elements</b> achieve high recognition with below-median investment (Efficient performers)",
                f"<b>Three matrix views</b> analyze Recognition vs Investment, Usage vs Recognition, and Uniqueness vs ROI positioning"
            ]
        )

        st.markdown("---")

        # Matrix 1: Recognition vs Investment
        st.markdown("#### 1️⃣ Recognition vs Investment Matrix")
        st.caption("Quadrant positioning based on consumer recognition and campaign investment levels")

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
                title='Recognition vs Investment',
                color_continuous_scale='RdYlGn',
                size_max=30,
                hover_data={
                    'Recognition': ':.0%',
                    'Total Investment': ':,.0f',
                    'Uniqueness': ':.0%',
                    'Net Sentiment': ':+.1%'
                }
            )

            # Add quadrant lines
            fig_bcg.add_hline(y=median_recognition, line_dash="dash", line_color="gray", opacity=0.5)
            fig_bcg.add_vline(x=median_investment, line_dash="dash", line_color="gray", opacity=0.5)

            # Add neutral quadrant labels (no prescriptive language)
            fig_bcg.add_annotation(
                x=matrix_df['Total Investment'].max() * 0.75,
                y=matrix_df['Recognition'].max() * 0.95,
                text="High Recognition<br>High Investment",
                showarrow=False,
                font=dict(size=10, color="darkgreen")
            )
            fig_bcg.add_annotation(
                x=matrix_df['Total Investment'].min() * 1.5,
                y=matrix_df['Recognition'].max() * 0.95,
                text="High Recognition<br>Low Investment",
                showarrow=False,
                font=dict(size=10, color="green")
            )
            fig_bcg.add_annotation(
                x=matrix_df['Total Investment'].max() * 0.75,
                y=matrix_df['Recognition'].min() * 1.5,
                text="Low Recognition<br>High Investment",
                showarrow=False,
                font=dict(size=10, color="red")
            )
            fig_bcg.add_annotation(
                x=matrix_df['Total Investment'].min() * 1.5,
                y=matrix_df['Recognition'].min() * 1.5,
                text="Low Recognition<br>Low Investment",
                showarrow=False,
                font=dict(size=10, color="orange")
            )

            fig_bcg = apply_standard_chart_styling(fig_bcg, "")
            fig_bcg.update_traces(textposition='top center')
            fig_bcg.update_layout(height=500, xaxis_title="Total Investment (€)", yaxis_title="Recognition")
            fig_bcg.update_yaxes(tickformat='.0%')
            st.plotly_chart(fig_bcg, use_container_width=True, config=get_standard_chart_config())

        with col2:
            st.markdown("#### Quadrant Breakdown")

            # Categorize elements
            high_rec_high_inv = matrix_df[(matrix_df['Recognition'] >= median_recognition) & (matrix_df['Total Investment'] >= median_investment)]
            high_rec_low_inv = matrix_df[(matrix_df['Recognition'] >= median_recognition) & (matrix_df['Total Investment'] < median_investment)]
            low_rec_high_inv = matrix_df[(matrix_df['Recognition'] < median_recognition) & (matrix_df['Total Investment'] >= median_investment)]
            low_rec_low_inv = matrix_df[(matrix_df['Recognition'] < median_recognition) & (matrix_df['Total Investment'] < median_investment)]

            if len(high_rec_high_inv) > 0:
                st.success(f"**High Rec + High Inv ({len(high_rec_high_inv)})**")
                for _, row in high_rec_high_inv.iterrows():
                    st.write(f"• {row['Element']}")
                st.caption(f"€{high_rec_high_inv['Total Investment'].sum():,.0f} invested")

            if len(high_rec_low_inv) > 0:
                st.success(f"**High Rec + Low Inv ({len(high_rec_low_inv)})**")
                for _, row in high_rec_low_inv.iterrows():
                    st.write(f"• {row['Element']}")
                st.caption(f"€{high_rec_low_inv['Total Investment'].sum():,.0f} invested")

            if len(low_rec_high_inv) > 0:
                st.error(f"**Low Rec + High Inv ({len(low_rec_high_inv)})**")
                for _, row in low_rec_high_inv.iterrows():
                    st.write(f"• {row['Element']}")
                st.caption(f"€{low_rec_high_inv['Total Investment'].sum():,.0f} invested")

            if len(low_rec_low_inv) > 0:
                st.warning(f"**Low Rec + Low Inv ({len(low_rec_low_inv)})**")
                for _, row in low_rec_low_inv.iterrows():
                    st.write(f"• {row['Element']}")
                st.caption(f"€{low_rec_low_inv['Total Investment'].sum():,.0f} invested")

        st.markdown("---")

        # Matrix 2: Recognition vs Uniqueness
        st.markdown("#### 2️⃣ Recognition vs Uniqueness Matrix")
        st.caption("Brand equity positioning across recognition (fame) and uniqueness (attribution)")

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
                title='Recognition vs Uniqueness',
                color_continuous_scale='RdYlGn',
                size_max=30,
                hover_data={
                    'Recognition': ':.0%',
                    'Uniqueness': ':.0%',
                    'Total Investment': ':,.0f',
                    'Recognition ROI': ':.2f'
                }
            )

            # Add quadrant lines
            fig_equity.add_hline(y=median_recognition, line_dash="dash", line_color="gray", opacity=0.5)
            fig_equity.add_vline(x=median_uniqueness, line_dash="dash", line_color="gray", opacity=0.5)

            # Add neutral quadrant labels
            fig_equity.add_annotation(
                x=matrix_df['Uniqueness'].max() * 0.9,
                y=matrix_df['Recognition'].max() * 0.95,
                text="High Recognition<br>High Uniqueness",
                showarrow=False,
                font=dict(size=10, color="darkgreen")
            )
            fig_equity.add_annotation(
                x=matrix_df['Uniqueness'].min() * 1.2,
                y=matrix_df['Recognition'].max() * 0.95,
                text="High Recognition<br>Lower Uniqueness",
                showarrow=False,
                font=dict(size=10, color="orange")
            )
            fig_equity.add_annotation(
                x=matrix_df['Uniqueness'].max() * 0.9,
                y=matrix_df['Recognition'].min() * 1.5,
                text="Lower Recognition<br>High Uniqueness",
                showarrow=False,
                font=dict(size=10, color="blue")
            )
            fig_equity.add_annotation(
                x=matrix_df['Uniqueness'].min() * 1.2,
                y=matrix_df['Recognition'].min() * 1.5,
                text="Lower Recognition<br>Lower Uniqueness",
                showarrow=False,
                font=dict(size=10, color="red")
            )

            fig_equity = apply_standard_chart_styling(fig_equity, "")
            fig_equity.update_traces(textposition='top center')
            fig_equity.update_layout(height=500, xaxis_title="Uniqueness (Brand Attribution)", yaxis_title="Recognition (Fame)")
            fig_equity.update_xaxes(tickformat='.0%')
            fig_equity.update_yaxes(tickformat='.0%')
            st.plotly_chart(fig_equity, use_container_width=True, config=get_standard_chart_config())

        with col2:
            st.markdown("#### Performance Groups")

            # Categorize elements
            icons = matrix_df[(matrix_df['Recognition'] >= median_recognition) & (matrix_df['Uniqueness'] >= median_uniqueness)]
            generics = matrix_df[(matrix_df['Recognition'] >= median_recognition) & (matrix_df['Uniqueness'] < median_uniqueness)]
            hidden = matrix_df[(matrix_df['Recognition'] < median_recognition) & (matrix_df['Uniqueness'] >= median_uniqueness)]
            weak = matrix_df[(matrix_df['Recognition'] < median_recognition) & (matrix_df['Uniqueness'] < median_uniqueness)]

            if len(icons) > 0:
                st.success(f"**High/High ({len(icons)})**")
                for _, row in icons.iterrows():
                    st.write(f"• {row['Element']}: {row['Recognition']:.0%} | {row['Uniqueness']:.0%}")

            if len(generics) > 0:
                st.warning(f"**High/Lower ({len(generics)})**")
                for _, row in generics.iterrows():
                    st.write(f"• {row['Element']}: {row['Recognition']:.0%} | {row['Uniqueness']:.0%}")

            if len(hidden) > 0:
                st.info(f"**Lower/High ({len(hidden)})**")
                for _, row in hidden.iterrows():
                    st.write(f"• {row['Element']}: {row['Recognition']:.0%} | {row['Uniqueness']:.0%}")

            if len(weak) > 0:
                st.error(f"**Lower/Lower ({len(weak)})**")
                for _, row in weak.iterrows():
                    st.write(f"• {row['Element']}: {row['Recognition']:.0%} | {row['Uniqueness']:.0%}")

        st.markdown("---")

        # Matrix 3: Usage vs ROI
        st.markdown("#### 3️⃣ Usage vs ROI Matrix")
        st.caption("Campaign efficiency analysis comparing usage frequency and return on investment")

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
                title='Usage vs ROI',
                color_continuous_scale='RdYlGn',
                size_max=30,
                hover_data={
                    'Overall Usage': ':.0%',
                    'Recognition ROI': ':.2f',
                    'Recognition': ':.0%',
                    'Net Sentiment': ':+.1%'
                }
            )

            # Add quadrant lines
            fig_efficiency.add_hline(y=median_roi, line_dash="dash", line_color="gray", opacity=0.5)
            fig_efficiency.add_vline(x=median_usage, line_dash="dash", line_color="gray", opacity=0.5)

            # Add neutral quadrant labels
            fig_efficiency.add_annotation(
                x=matrix_df['Overall Usage'].max() * 0.85,
                y=matrix_df['Recognition ROI'].max() * 0.95,
                text="High Usage<br>High ROI",
                showarrow=False,
                font=dict(size=10, color="darkgreen")
            )
            fig_efficiency.add_annotation(
                x=matrix_df['Overall Usage'].min() * 1.5,
                y=matrix_df['Recognition ROI'].max() * 0.95,
                text="Low Usage<br>High ROI",
                showarrow=False,
                font=dict(size=10, color="blue")
            )
            fig_efficiency.add_annotation(
                x=matrix_df['Overall Usage'].max() * 0.85,
                y=matrix_df['Recognition ROI'].min() * 1.5,
                text="High Usage<br>Low ROI",
                showarrow=False,
                font=dict(size=10, color="red")
            )
            fig_efficiency.add_annotation(
                x=matrix_df['Overall Usage'].min() * 1.5,
                y=matrix_df['Recognition ROI'].min() * 1.5,
                text="Low Usage<br>Low ROI",
                showarrow=False,
                font=dict(size=10, color="orange")
            )

            fig_efficiency = apply_standard_chart_styling(fig_efficiency, "")
            fig_efficiency.update_traces(textposition='top center')
            fig_efficiency.update_layout(height=500, xaxis_title="Campaign Usage", yaxis_title="Recognition ROI (per €1M)")
            fig_efficiency.update_xaxes(tickformat='.0%')
            st.plotly_chart(fig_efficiency, use_container_width=True, config=get_standard_chart_config())

        with col2:
            st.markdown("#### Efficiency Groups")

            # Categorize elements
            high_use_high_roi = matrix_df[(matrix_df['Overall Usage'] >= median_usage) & (matrix_df['Recognition ROI'] >= median_roi)]
            low_use_high_roi = matrix_df[(matrix_df['Overall Usage'] < median_usage) & (matrix_df['Recognition ROI'] >= median_roi)]
            high_use_low_roi = matrix_df[(matrix_df['Overall Usage'] >= median_usage) & (matrix_df['Recognition ROI'] < median_roi)]
            low_use_low_roi = matrix_df[(matrix_df['Overall Usage'] < median_usage) & (matrix_df['Recognition ROI'] < median_roi)]

            if len(high_use_high_roi) > 0:
                st.success(f"**High Use + High ROI ({len(high_use_high_roi)})**")
                for _, row in high_use_high_roi.iterrows():
                    st.write(f"• {row['Element']}: {row['Overall Usage']:.0%} usage | {row['Recognition ROI']:.2f} ROI")

            if len(low_use_high_roi) > 0:
                st.info(f"**Low Use + High ROI ({len(low_use_high_roi)})**")
                for _, row in low_use_high_roi.iterrows():
                    st.write(f"• {row['Element']}: {row['Overall Usage']:.0%} usage | {row['Recognition ROI']:.2f} ROI")

            if len(high_use_low_roi) > 0:
                st.error(f"**High Use + Low ROI ({len(high_use_low_roi)})**")
                for _, row in high_use_low_roi.iterrows():
                    st.write(f"• {row['Element']}: {row['Overall Usage']:.0%} usage | {row['Recognition ROI']:.2f} ROI")

            if len(low_use_low_roi) > 0:
                st.warning(f"**Low Use + Low ROI ({len(low_use_low_roi)})**")
                for _, row in low_use_low_roi.iterrows():
                    st.write(f"• {row['Element']}: {row['Overall Usage']:.0%} usage | {row['Recognition ROI']:.2f} ROI")

    # ========== SUB-TAB 2: EFFICIENCY & ROI ==========
    with subtab2:
        st.markdown("### 💡 Multi-Dimensional ROI Analysis")
        st.caption("Compare efficiency across different investment and performance metrics")

        # Calculate for insights
        best_roi_elem = master_df.loc[master_df['Recognition ROI'].idxmax()]
        worst_roi_elem = master_df.loc[master_df['Recognition ROI'].idxmin()]
        roi_range = best_roi_elem['Recognition ROI'] / worst_roi_elem['Recognition ROI']

        # Key Insights Box
        render_tldr_box(
            "Key Insights at a Glance",
            [
                f"<b>{best_roi_elem['Element']}</b> delivers highest efficiency at {best_roi_elem['Recognition ROI']:.2f} recognition per €1M",
                f"<b>{roi_range:.1f}x efficiency gap</b> exists between best ({best_roi_elem['Element']}) and lowest ({worst_roi_elem['Element']}) performers",
                f"<b>4 value for moneys available</b>: Total Investment, Per-Ad, Average Investment, and Brand Equity Index"
            ]
        )

        # Summary cards
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🏆 Highest Efficiency", best_roi_elem['Element'], f"{best_roi_elem['Recognition ROI']:.2f} per €1M")
        with col2:
            st.metric("📊 Lowest Efficiency", worst_roi_elem['Element'], f"{worst_roi_elem['Recognition ROI']:.2f} per €1M")

        st.markdown("---")

        # ROI metric selector
        roi_metric = st.selectbox(
            "Select value for money:",
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
            insight_text = "**What this shows:** Recognition achieved relative to total campaign spend across all placements featuring this element."

        elif roi_metric == "Per-Ad Recognition Efficiency":
            master_df_roi['Selected ROI'] = master_df_roi.apply(
                lambda x: (x['Recognition'] / x['Num Ads'] * 100) if x['Num Ads'] > 0 else 0, axis=1
            )
            metric_label = "Recognition % per Ad Placement"
            insight_text = "**What this shows:** Average recognition gained per individual ad placement - indicates how quickly each element builds awareness."

        elif roi_metric == "Average Investment Efficiency":
            master_df_roi['Selected ROI'] = master_df_roi.apply(
                lambda x: (x['Recognition'] / x['Average Investment'] * 1_000_000) if x['Average Investment'] > 0 else 0, axis=1
            )
            metric_label = "Recognition % per €1M Average Placement Investment"
            insight_text = "**What this shows:** Cost-effectiveness per typical ad placement budget for this element."

        else:  # Brand Equity Efficiency Index
            master_df_roi['Selected ROI'] = master_df_roi.apply(
                lambda x: (x['Recognition'] * x['Uniqueness']) / (x['Total Investment'] / 1_000_000) if x['Total Investment'] > 0 else 0, axis=1
            )
            metric_label = "Brand Equity Index (Recognition × Uniqueness) per €1M"
            insight_text = "**What this shows:** Combined efficiency of building both fame (recognition) and differentiation (uniqueness) per euro invested."

        st.info(insight_text)

        col1, col2 = st.columns([2, 1])

        with col1:
            roi_df = master_df_roi.sort_values('Selected ROI', ascending=True)
            fig_roi = px.bar(
                roi_df,
                y='Element',
                x='Selected ROI',
                orientation='h',
                title=f'Efficiency Comparison: {metric_label}',
                text=roi_df['Selected ROI'].apply(lambda x: f'{x:.2f}'),
                color='Selected ROI',
                color_continuous_scale='RdYlGn',
                hover_data={
                    'Element': True,
                    'Selected ROI': ':.2f',
                    'Recognition': ':.0%',
                    'Total Investment': ':,.0f'
                }
            )
            fig_roi = apply_standard_chart_styling(fig_roi, "")
            fig_roi.update_traces(textposition='outside')
            fig_roi.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_roi, use_container_width=True, config=get_standard_chart_config())

        with col2:
            st.markdown("#### Top 3 Performers")
            top_3_roi = roi_df.nlargest(3, 'Selected ROI')
            for idx, row in top_3_roi.iterrows():
                st.success(f"**{row['Element']}**: {row['Selected ROI']:.2f}")
                with st.expander(f"Performance breakdown"):
                    if roi_metric == "Brand Equity Efficiency Index":
                        equity = row['Recognition'] * row['Uniqueness']
                        st.write(f"**Recognition:** {row['Recognition']:.0%}")
                        st.write(f"**Uniqueness:** {row['Uniqueness']:.0%}")
                        st.write(f"**Combined Equity:** {equity:.3f}")
                        st.write(f"**Investment:** €{row['Total Investment']:,.0f}")
                        st.write(f"**Pattern:** Achieves {equity:.3f} brand equity score with €{row['Total Investment']:,.0f} investment")
                    else:
                        st.write(f"**Recognition:** {row['Recognition']:.0%}")
                        st.write(f"**Investment:** €{row['Total Investment']:,.0f}")
                        st.write(f"**Usage:** {row['Overall Usage']:.0%}")
                        st.write(f"**Pattern:** {row['Recognition']:.0%} recognition from €{row['Total Investment']:,.0f} investment across {row['Overall Usage']:.0%} of campaigns")

            st.markdown("#### Bottom 3 Performers")
            bottom_3_roi = roi_df.nsmallest(3, 'Selected ROI')
            for idx, row in bottom_3_roi.iterrows():
                st.warning(f"**{row['Element']}**: {row['Selected ROI']:.2f}")
                with st.expander(f"Performance breakdown"):
                    if roi_metric == "Brand Equity Efficiency Index":
                        equity = row['Recognition'] * row['Uniqueness']
                        st.write(f"**Recognition:** {row['Recognition']:.0%}")
                        st.write(f"**Uniqueness:** {row['Uniqueness']:.0%}")
                        st.write(f"**Combined Equity:** {equity:.3f}")
                        st.write(f"**Investment:** €{row['Total Investment']:,.0f}")
                        st.write(f"**Pattern:** {equity:.3f} brand equity score from €{row['Total Investment']:,.0f} investment")
                    else:
                        st.write(f"**Recognition:** {row['Recognition']:.0%}")
                        st.write(f"**Investment:** €{row['Total Investment']:,.0f}")
                        st.write(f"**Usage:** {row['Overall Usage']:.0%}")
                        st.write(f"**Pattern:** {row['Recognition']:.0%} recognition from €{row['Total Investment']:,.0f} investment across {row['Overall Usage']:.0%} of campaigns")

        st.markdown("---")

        # Quadrant Analysis with neutral labels
        st.markdown("### 📊 Recognition vs Uniqueness Performance Groups")
        st.caption("Elements categorized by above/below median performance on both dimensions")

        with st.expander("📖 Understanding the categories"):
            st.markdown(f"""
            Elements are grouped based on their position relative to median values:
            - **Median Recognition:** {master_df['Recognition'].median():.0%}
            - **Median Uniqueness:** {master_df['Uniqueness'].median():.0%}

            **High Recognition + High Uniqueness:**
            - Above-median on both consumer recognition and who it belongs to

            **High Recognition + Lower Uniqueness:**
            - Above-median recognition but below-median who it belongs to

            **Lower Recognition + High Uniqueness:**
            - Strong who it belongs to but below-median recognition

            **Lower Recognition + Lower Uniqueness:**
            - Below-median on both recognition and who it belongs to
            """)

        # Calculate quadrants
        median_recognition_q = master_df['Recognition'].median()
        median_uniqueness_q = master_df['Uniqueness'].median()

        def get_quadrant_neutral(row):
            if row['Recognition'] >= median_recognition_q and row['Uniqueness'] >= median_uniqueness_q:
                return 'High/High ⭐'
            elif row['Recognition'] >= median_recognition_q and row['Uniqueness'] < median_uniqueness_q:
                return 'High/Lower 🔵'
            elif row['Recognition'] < median_recognition_q and row['Uniqueness'] >= median_uniqueness_q:
                return 'Lower/High 💎'
            else:
                return 'Lower/Lower ⚪'

        master_df_quad = master_df.copy()
        master_df_quad['Category'] = master_df_quad.apply(get_quadrant_neutral, axis=1)

        fig_quadrant = px.scatter(
            master_df_quad,
            x='Uniqueness',
            y='Recognition',
            color='Category',
            text='Element',
            size='Total Investment',
            size_max=50,
            title='Performance Category Distribution',
            color_discrete_map={
                'High/High ⭐': '#4CAF50',
                'High/Lower 🔵': '#2196F3',
                'Lower/High 💎': '#FF9800',
                'Lower/Lower ⚪': '#9E9E9E'
            },
            hover_data={
                'Recognition': ':.0%',
                'Uniqueness': ':.0%',
                'Total Investment': ':,.0f',
                'Recognition ROI': ':.2f'
            }
        )

        # Add median lines
        fig_quadrant.add_hline(y=median_recognition_q, line_dash="dash", line_color="gray", annotation_text="Median Recognition")
        fig_quadrant.add_vline(x=median_uniqueness_q, line_dash="dash", line_color="gray", annotation_text="Median Uniqueness")
        fig_quadrant = apply_standard_chart_styling(fig_quadrant, "")
        fig_quadrant.update_traces(textposition='top center')
        fig_quadrant.update_layout(height=600)
        fig_quadrant.update_xaxes(tickformat='.0%')
        fig_quadrant.update_yaxes(tickformat='.0%')

        st.plotly_chart(fig_quadrant, use_container_width=True, config=get_standard_chart_config())

        # Category breakdown
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            high_high = master_df_quad[master_df_quad['Category'] == 'High/High ⭐']
            st.success(f"**High/High ⭐** ({len(high_high)})")
            for idx, row in high_high.iterrows():
                st.write(f"• **{row['Element']}**")
                st.caption(f"{row['Recognition']:.0%} rec | {row['Uniqueness']:.0%} uniq")

        with col2:
            high_lower = master_df_quad[master_df_quad['Category'] == 'High/Lower 🔵']
            st.info(f"**High/Lower 🔵** ({len(high_lower)})")
            for idx, row in high_lower.iterrows():
                st.write(f"• **{row['Element']}**")
                st.caption(f"{row['Recognition']:.0%} rec | {row['Uniqueness']:.0%} uniq")

        with col3:
            lower_high = master_df_quad[master_df_quad['Category'] == 'Lower/High 💎']
            st.warning(f"**Lower/High 💎** ({len(lower_high)})")
            for idx, row in lower_high.iterrows():
                st.write(f"• **{row['Element']}**")
                st.caption(f"{row['Recognition']:.0%} rec | {row['Uniqueness']:.0%} uniq")

        with col4:
            lower_lower = master_df_quad[master_df_quad['Category'] == 'Lower/Lower ⚪']
            st.error(f"**Lower/Lower ⚪** ({len(lower_lower)})")
            for idx, row in lower_lower.iterrows():
                st.write(f"• **{row['Element']}**")
                st.caption(f"{row['Recognition']:.0%} rec | {row['Uniqueness']:.0%} uniq")

    # ========== SUB-TAB 3: COMBINATIONS & SYNERGIES ==========
    with subtab3:
        st.markdown("### 🔗 Element Combinations Analysis")
        st.caption("How elements perform when appearing together in campaigns")

        # Calculate metrics for insights
        avg_elements = audit_df[brand_elements].sum(axis=1).mean()
        most_paired = None
        for elem in brand_elements:
            elem_ads = audit_df[audit_df[elem] == True]
            if len(elem_ads) > 0:
                other_elements = [e for e in brand_elements if e != elem]
                avg_companions = elem_ads[other_elements].sum(axis=1).mean()
                if most_paired is None or avg_companions > most_paired[1]:
                    most_paired = (elem, avg_companions)

        # Key Insights Box
        render_tldr_box(
            "Key Insights at a Glance",
            [
                f"<b>Average {avg_elements:.1f} elements</b> deployed per campaign across {len(audit_df)} total ads",
                f"<b>{most_paired[0]}</b> appears most frequently in combinations with average {most_paired[1]:.1f} companion elements",
                f"<b>Symbol-based combinations</b> consistently achieve higher recognition levels in portfolio analysis"
            ]
        )

        st.markdown("---")

        # Demographic filters
        combo_filters = render_demographic_filters("combo")

        st.markdown("#### 📊 Multi-Element Distribution")
        st.caption("Number of brand elements used per campaign")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Calculate how many elements appear together
            audit_df['num_elements'] = audit_df[brand_elements].sum(axis=1)
            elements_per_ad = audit_df['num_elements'].value_counts().sort_index()

            fig_elements = go.Figure(go.Bar(
                x=elements_per_ad.index,
                y=elements_per_ad.values,
                marker_color='#4CAF50',
                text=elements_per_ad.values,
                textposition='outside',
                hovertemplate='<b>%{x} Elements</b><br>%{y} campaigns<extra></extra>'
            ))
            fig_elements = apply_standard_chart_styling(fig_elements, 'Distribution: Elements per Campaign')
            fig_elements.update_layout(
                xaxis_title='Number of Brand Elements',
                yaxis_title='Number of Campaigns',
                height=400
            )
            st.plotly_chart(fig_elements, use_container_width=True, config=get_standard_chart_config())

        with col2:
            st.markdown("#### Distribution Metrics")

            avg_elements = audit_df['num_elements'].mean()
            st.metric("Average Elements/Campaign", f"{avg_elements:.1f}")

            median_elements = audit_df['num_elements'].median()
            st.metric("Median Elements/Campaign", f"{int(median_elements)}")

            max_elements = audit_df['num_elements'].max()
            st.metric("Maximum Elements/Campaign", f"{int(max_elements)}")

            min_elements = audit_df['num_elements'].min()
            st.metric("Minimum Elements/Campaign", f"{int(min_elements)}")

        st.markdown("---")

        # Co-occurrence analysis
        st.markdown("#### 🔗 Element Co-occurrence Patterns")
        st.caption("How frequently different elements appear together in campaigns")

        # Create co-occurrence matrix
        cooccurrence_matrix = pd.DataFrame(0, index=brand_elements, columns=brand_elements, dtype=int)

        for element1 in brand_elements:
            for element2 in brand_elements:
                if element1 != element2:
                    both_present = audit_df[audit_df[element1] & audit_df[element2]].shape[0]
                    cooccurrence_matrix.loc[element1, element2] = both_present

        # Display as heatmap
        fig_cooccur = px.imshow(
            cooccurrence_matrix,
            labels=dict(x="Appears with", y="Element", color="Co-occurrences"),
            x=cooccurrence_matrix.columns,
            y=cooccurrence_matrix.index,
            color_continuous_scale='Blues',
            text_auto=True,
            aspect="auto",
            title="Element Co-occurrence Frequency"
        )
        fig_cooccur = apply_standard_chart_styling(fig_cooccur, "")
        fig_cooccur.update_layout(height=600)
        st.plotly_chart(fig_cooccur, use_container_width=True, config=get_standard_chart_config())

        st.markdown("---")

        # Most common combinations
        st.markdown("#### 🏆 Most Frequent Element Pairs")
        st.caption("Top 10 element combinations across all campaigns")

        combinations = []
        for element1 in brand_elements:
            for element2 in brand_elements:
                if element1 < element2:  # Avoid duplicates
                    both_present = audit_df[audit_df[element1] & audit_df[element2]].shape[0]
                    if both_present > 0:
                        # Get recognition for both elements
                        rec1 = master_df[master_df['Element'] == element1]['Recognition'].values[0]
                        rec2 = master_df[master_df['Element'] == element2]['Recognition'].values[0]
                        avg_rec = (rec1 + rec2) / 2

                        combinations.append({
                            'Pair': f"{element1} + {element2}",
                            'Campaigns': both_present,
                            'Avg Recognition': avg_rec
                        })

        combinations_df = pd.DataFrame(combinations).sort_values('Campaigns', ascending=False).head(10)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**By Frequency:**")
            for idx, row in combinations_df.iterrows():
                st.success(f"**{row['Pair']}**")
                st.caption(f"Appears in {row['Campaigns']} campaigns | Avg recognition: {row['Avg Recognition']:.0%}")

        with col2:
            # Sort by recognition
            combinations_by_rec = pd.DataFrame(combinations).sort_values('Avg Recognition', ascending=False).head(10)
            st.markdown("**By Recognition:**")
            for idx, row in combinations_by_rec.head(5).iterrows():
                st.info(f"**{row['Pair']}**")
                st.caption(f"{row['Avg Recognition']:.0%} avg recognition | {row['Campaigns']} campaigns")

        st.markdown("---")

        # Usage patterns
        st.markdown("#### 📈 Element Usage Patterns")

        usage_summary = []
        for element in brand_elements:
            campaigns_with_element = audit_df[audit_df[element] == True].shape[0]
            usage_pct = campaigns_with_element / len(audit_df) * 100
            recognition = master_df[master_df['Element'] == element]['Recognition'].values[0]
            uniqueness = master_df[master_df['Element'] == element]['Uniqueness'].values[0]

            usage_summary.append({
                'Element': element,
                'Campaigns': campaigns_with_element,
                'Usage %': usage_pct,
                'Recognition': recognition,
                'Uniqueness': uniqueness
            })

        usage_df = pd.DataFrame(usage_summary).sort_values('Usage %', ascending=False)

        fig_usage = px.bar(
            usage_df,
            x='Element',
            y='Usage %',
            color='Recognition',
            title='Element Usage Across Campaigns',
            text=usage_df['Usage %'].apply(lambda x: f'{x:.0f}%'),
            color_continuous_scale='RdYlGn',
            hover_data={
                'Usage %': ':.1f',
                'Recognition': ':.0%',
                'Uniqueness': ':.0%',
                'Campaigns': True
            }
        )
        fig_usage = apply_standard_chart_styling(fig_usage, "")
        fig_usage.update_traces(textposition='outside')
        fig_usage.update_layout(height=450, yaxis_title="Usage Percentage")
        st.plotly_chart(fig_usage, use_container_width=True, config=get_standard_chart_config())

    # ========== SUB-TAB 4: MARKET & CONSUMER INSIGHTS ==========
    with subtab4:
        st.markdown("### 🌍 Market Analysis & Consumer Language")
        st.caption("Market consistency, consumer associations (Q03), and who it belongs to (Q05)")

        # Calculate metrics for insights
        markets_count = audit_df['Market'].nunique()
        # Calculate variance in usage across markets
        market_variances = []
        for element in brand_elements:
            usages = []
            for market in audit_df['Market'].unique():
                market_df = audit_df[audit_df['Market'] == market]
                usage = market_df[element].sum() / len(market_df) if len(market_df) > 0 else 0
                usages.append(usage)
            market_variances.append((element, pd.Series(usages).std()))
        most_consistent = min(market_variances, key=lambda x: x[1])
        most_variable = max(market_variances, key=lambda x: x[1])

        # Key Insights Box
        render_tldr_box(
            "Key Insights at a Glance",
            [
                f"<b>{markets_count} markets analyzed</b> for cross-market consistency in element use patterns",
                f"<b>{most_consistent[0]}</b> shows most consistent usage across markets (lowest variation)",
                f"<b>{most_variable[0]}</b> shows highest market variation suggesting localized use strategies"
            ]
        )

        st.markdown("---")

        # Market Consistency Analysis
        st.markdown("### 🌍 Cross-Market Element Usage")
        st.caption("Consistency of brand element use across different markets")

        markets = sorted(audit_df['Market'].unique())
        selected_markets = st.multiselect("Select markets to compare:", markets, default=markets, key="market_selector")

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
                text=market_comparison['Usage'].apply(lambda x: f'{x:.0%}'),
                hover_data={'Usage': ':.1%'}
            )
            fig_market = apply_standard_chart_styling(fig_market, "")
            fig_market.update_traces(textposition='outside')
            fig_market.update_layout(height=450, yaxis_tickformat='.0%')
            st.plotly_chart(fig_market, use_container_width=True, config=get_standard_chart_config())

            # Consistency score
            consistency_scores = market_comparison.groupby('Element')['Usage'].std()
            most_consistent = consistency_scores.idxmin()
            least_consistent = consistency_scores.idxmax()

            col1, col2 = st.columns(2)
            with col1:
                if pd.notna(most_consistent):
                    st.success(f"**Most Consistent Usage:** {most_consistent} (σ={consistency_scores[most_consistent]:.3f})")
                    st.caption("Low standard deviation indicates consistent usage across markets")
                else:
                    st.info("Consistency data not available")
            with col2:
                if pd.notna(least_consistent):
                    st.warning(f"**Most Variable Usage:** {least_consistent} (σ={consistency_scores[least_consistent]:.3f})")
                    st.caption("High standard deviation indicates inconsistent usage across markets")
                else:
                    st.info("Consistency data not available")

        st.markdown("---")

        # Consumer Language Analysis (Q03)
        st.markdown("### 💬 Consumer Language Analysis (Q03)")
        st.caption("Words and phrases consumers use to describe brand elements")

        st.info("""
💡 **What this shows:** Analysis of open-ended responses revealing how consumers naturally describe Škoda brand elements in their own words (not predefined scales).
""")

        # Element selector
        selected_element_q03 = st.selectbox(
            "Select element to analyze:",
            list(q03_associations_data.keys()),
            key="q03_element_selector"
        )

        # Demographic filters for consumer language
        language_filters = render_demographic_filters("language")

        element_data = q03_associations_data[selected_element_q03]

        col1, col2 = st.columns([2, 1])

        with col1:
            # Top words bar chart
            st.markdown(f"#### Top 10 Words for {selected_element_q03}")

            words_df = pd.DataFrame({
                'Word': element_data['top_words'],
                'Frequency': element_data['frequencies']
            })

            fig_words = px.bar(
                words_df,
                x='Frequency',
                y='Word',
                orientation='h',
                title=f'Most Common Words: {selected_element_q03}',
                text=words_df['Frequency'].apply(lambda x: f'{x:.0%}'),
                color='Frequency',
                color_continuous_scale='Blues',
                hover_data={'Frequency': ':.1%'}
            )
            fig_words = apply_standard_chart_styling(fig_words, "")
            fig_words.update_layout(height=400, showlegend=False, xaxis_tickformat='.0%')
            fig_words.update_traces(textposition='outside')
            st.plotly_chart(fig_words, use_container_width=True, config=get_standard_chart_config())

        with col2:
            # Sentiment analysis from Q04 adjective scales
            st.markdown("#### Sentiment (Q04 Adjectives)")

            # Get sentiment data from research_data (Q04)
            sentiment_data_source = research_data[selected_element_q03]

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

        # Word associations table
        st.markdown(f"#### All Associations for {selected_element_q03}")
        st.caption("Full list of consumer descriptions (Q03 open-text responses)")

        all_words_df = pd.DataFrame({
            'Association': element_data['top_words'],
            'Frequency': element_data['frequencies']
        })

        st.dataframe(all_words_df.style.format({'Frequency': '{:.1%}'}),
                    use_container_width=True, hide_index=True)

        st.markdown("---")

        # Comparative sentiment analysis across all elements (Q04)
        st.markdown("### 📊 Sentiment Comparison Across All Elements")
        st.caption("Shows how people feel about each element: Bold, Stylish, Modern vs Cautious, Plain, Old-Fashioned")

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
            marker_color='#4CAF50',
            hovertemplate='<b>%{y}</b><br>Positive: %{x:.1%}<extra></extra>'
        ))

        fig_sent_comp.add_trace(go.Bar(
            name='Negative',
            y=sent_comparison_df['Element'],
            x=sent_comparison_df['Negative'],
            orientation='h',
            marker_color='#F44336',
            hovertemplate='<b>%{y}</b><br>Negative: %{x:.1%}<extra></extra>'
        ))

        fig_sent_comp = apply_standard_chart_styling(fig_sent_comp, 'Adjective Sentiment Analysis: All Elements')
        fig_sent_comp.update_layout(
            barmode='overlay',
            xaxis_title='Percentage',
            yaxis_title='',
            height=500,
            xaxis_tickformat='.0%'
        )

        st.plotly_chart(fig_sent_comp, use_container_width=True, config=get_standard_chart_config())

        # Key insights
        col1, col2 = st.columns(2)

        with col1:
            most_positive_sent = sent_comparison_df.iloc[-1]
            st.success(f"""
**Highest Positive Sentiment:**
- **{most_positive_sent['Element']}**: {most_positive_sent['Net']:+.1%} net sentiment
- {most_positive_sent['Positive']:.0%} positive adjectives
""")

        with col2:
            most_negative_sent = sent_comparison_df.iloc[0]
            st.warning(f"""
**Lowest Sentiment Score:**
- **{most_negative_sent['Element']}**: {most_negative_sent['Net']:+.1%} net sentiment
- {most_negative_sent['Negative']:.0%} negative adjectives
""")

        st.markdown("---")

        # Strategic Terms Search
        st.markdown("### 🔍 Strategic Brand Terms Search")
        st.caption("Search Q03 responses to see if desired brand values appear in consumer language")

        st.info("""
💡 **Purpose:** Identify whether strategic brand values (e.g., "Exploration", "Innovation", "Modern") appear naturally in consumer responses about Škoda brand elements.
""")

        # Search input
        search_term = st.text_input("Search for a word or phrase in consumer associations:",
                                    value="explore",
                                    placeholder="e.g., explore, innovation, modern, safe, boring",
                                    key="strategic_search")

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
                st.caption("This term does not appear prominently in consumer language about Škoda brand elements.")

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
            text=overall_df['Total Frequency'].apply(lambda x: f'{x:.1%}'),
            color='Total Frequency',
            color_continuous_scale='Viridis'
        )
        fig_overall = apply_standard_chart_styling(fig_overall, "")
        fig_overall.update_layout(height=500, showlegend=False, xaxis_tickformat='.0%')
        fig_overall.update_traces(textposition='outside')
        st.plotly_chart(fig_overall, use_container_width=True, config=get_standard_chart_config())

        st.markdown("---")

        # Brand Confusion Analysis (Q05)
        st.markdown("### 🎯 Brand Attribution Matrix (Q05)")
        st.caption("Do people think these belong to Škoda or competitors?")

        st.info("""
💡 **What this shows:** Brand attribution analysis reveals competitive positioning. High Škoda attribution indicates distinctive brand elements. High competitor attribution or "Don't Know" responses indicate confusion or weak brand association.
""")

        # Demographic filters for confusion matrix
        confusion_filters = render_demographic_filters("confusion")

        # Create confusion matrix using Q05 data
        confusion_df = pd.DataFrame(q05_confusion_data).T
        confusion_df = confusion_df[['Skoda', 'Other_mentions', 'Dont_know']]
        confusion_df.columns = ['Škoda', 'Other Brands', "Don't Know"]

        # Create display version with inverted competitor columns for color coding
        confusion_df_display = confusion_df.copy()

        # Invert competitor and "Don't Know" columns (1 - value) so high becomes low for coloring
        for col in ['Other Brands', "Don't Know"]:
            confusion_df_display[col] = 1 - confusion_df_display[col]

        # Create heatmap
        fig_confusion = px.imshow(
            confusion_df_display,
            labels=dict(x="Attributed Brand", y="Element", color="Score"),
            x=confusion_df_display.columns,
            y=confusion_df_display.index,
            color_continuous_scale='RdYlGn',
            text_auto=False,
            aspect="auto",
            title="Brand Attribution: Consumer Perception of Element Ownership"
        )

        # Add text annotations with actual percentages
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
        st.plotly_chart(fig_confusion, use_container_width=True, config=get_standard_chart_config())

        st.caption("""
**Color Guide:**
- 🟢 Green = Positive (High Škoda attribution OR Low competitor/generic confusion)
- 🔴 Red = Concern (Low Škoda attribution OR High competitor/generic confusion)
""")

        # Analysis columns
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ✅ Strongest Brand Attribution")
            distinctive = confusion_df.sort_values('Škoda', ascending=False).head(3)
            for element, row in distinctive.iterrows():
                st.success(f"**{element}**: {row['Škoda']:.0%} Škoda attribution")
                dont_know_val = row["Don't Know"]
                st.caption(f"Other brands: {row['Other Brands']:.0%} | Don't know: {dont_know_val:.0%}")

        with col2:
            st.markdown("#### 📊 Attribution Patterns")

            # Find elements with high other brand confusion
            high_other = confusion_df[confusion_df['Other Brands'] >= 0.20].sort_values('Other Brands', ascending=False)
            if len(high_other) > 0:
                st.warning("**Other Brand Attribution:**")
                for element, row in high_other.iterrows():
                    st.write(f"• **{element}**: {row['Other Brands']:.0%} attribute to other brands")

            # Find elements with high don't know
            high_dontknow = confusion_df[confusion_df["Don't Know"] >= 0.55].sort_values("Don't Know", ascending=False)
            if len(high_dontknow) > 0:
                st.info("**Low Recognition:**")
                for element, row in high_dontknow.iterrows():
                    dont_know_pct = row["Don't Know"]
                    st.write(f"• **{element}**: {dont_know_pct:.0%} don't recognize")

        # Attribution data table
        st.markdown("#### 📊 Complete Attribution Matrix")

        attribution_matrix = []
        for element in confusion_df.index:
            skoda_attr = confusion_df.loc[element, 'Škoda']
            other_brands = confusion_df.loc[element, 'Other Brands']
            dont_know = confusion_df.loc[element, "Don't Know"]

            attribution_matrix.append({
                'Element': element,
                'Škoda Attribution': skoda_attr,
                'Other Brands': other_brands,
                "Don't Know": dont_know
            })

        attr_df = pd.DataFrame(attribution_matrix).sort_values('Škoda Attribution', ascending=False)
        st.dataframe(attr_df.style.format({
            'Škoda Attribution': '{:.0%}',
            'Other Brands': '{:.0%}',
            "Don't Know": '{:.0%}'
        }), use_container_width=True, hide_index=True)

        # Detailed Competitor Breakdown
        st.markdown("---")
        st.markdown("### 🔍 Detailed Competitor Analysis")
        st.caption("Specific brands mentioned when consumers misattribute Škoda elements")

        # Load detailed competitor data
        try:
            with open('q05_competitor_detail_CLEANED.json', 'r', encoding='utf-8') as f:
                competitor_detail = json.load(f)

            # Global Summary
            st.markdown("#### 🌍 Automotive Competitor Mentions (Aggregated)")
            st.caption("Which car brands are most confused with Škoda elements?")

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
                        title="Automotive Competitor Confusion",
                        labels={'Mentions': 'Total Mentions', 'Brand': 'Competitor Brand'},
                        color='Mentions',
                        color_continuous_scale='Reds',
                        hover_data={'Mentions': True}
                    )
                    fig_global_comp = apply_standard_chart_styling(fig_global_comp, "")
                    fig_global_comp.update_traces(texttemplate='%{y}', textposition='outside')
                    fig_global_comp.update_layout(height=350, showlegend=False)
                    st.plotly_chart(fig_global_comp, use_container_width=True, config=get_standard_chart_config())

                with col2:
                    st.markdown("**Summary Statistics:**")
                    st.metric("Total Automotive Mentions", total_mentions)
                    st.caption(f"Out of ~726 total verbatim responses")

                    if global_auto_competitors:
                        top_competitor = max(global_auto_competitors.items(), key=lambda x: x[1])
                        st.metric("Top Competitor", top_competitor[0])
                        st.caption(f"{top_competitor[1]} mentions")

                        confusion_rate = (total_mentions / 726) * 100 if total_mentions > 0 else 0
                        st.info(f"""
**Overall Automotive Confusion: {confusion_rate:.1f}%**

Pattern indicates minimal competitive confusion - Škoda elements are generally not mistaken for competitor car brands.
""")
            else:
                st.info("No significant automotive competitor confusion detected in responses")

        except FileNotFoundError:
            st.warning("⚠️ Detailed competitor data file not found (q05_competitor_detail_CLEANED.json)")

with tab4:
    st.header("🎯 Asset Performance Framework")
    st.caption("Which elements perform best and why")

    # Auto-generate categories based on data
    high_performers = master_df[
        (master_df['Recognition'] >= 0.40) &
        (master_df['Uniqueness'] >= 0.15) &
        (master_df['Overall Usage'] >= 0.50)
    ].sort_values('Recognition', ascending=False)

    strong_potential = master_df[
        ((master_df['Recognition'] >= 0.35) | (master_df['Uniqueness'] >= 0.25))
    ].sort_values(['Recognition', 'Uniqueness'], ascending=False)
    strong_potential = strong_potential[~strong_potential['Element'].isin(high_performers['Element'])]

    development_opportunity = master_df[
        (master_df['Recognition'] < 0.40) &
        (master_df['Total Investment'] > master_df['Total Investment'].median())
    ]

    # Key Insights Box
    hp_list = ", ".join(high_performers['Element'].tolist()) if len(high_performers) > 0 else "None"
    render_tldr_box(
        "Key Insights at a Glance",
        [
            f"<b>{len(high_performers)} High Performers</b> meet all criteria: {hp_list}",
            f"<b>{len(strong_potential)} elements</b> show strong potential with above-average recognition or uniqueness",
            f"<b>{len(development_opportunity)} elements</b> require optimization despite receiving above-median investment"
        ]
    )

    # Display categories
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🟢 Category 1: High Performers")
        st.success(f"**{len(high_performers)} elements** meet all criteria: Recognition ≥40%, Uniqueness ≥15%, Usage ≥50%")

        if len(high_performers) > 0:
            for idx, row in high_performers.iterrows():
                with st.expander(f"**{row['Element']}** - Recognition: {row['Recognition']:.0%} | Usage: {row['Overall Usage']:.0%}"):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Recognition", f"{row['Recognition']:.0%}")
                        st.metric("Uniqueness", f"{row['Uniqueness']:.0%}")
                    with col_b:
                        st.metric("Usage", f"{row['Overall Usage']:.0%}")
                        st.metric("Investment", f"€{row['Total Investment']:,.0f}")
                    with col_c:
                        equity_score = row['Recognition'] * row['Uniqueness']
                        st.metric("Brand Equity", f"{equity_score:.3f}")
                        st.metric("ROI", f"{row['Recognition ROI']:.2f}")

                    st.markdown("**Performance Profile:**")
                    st.write(f"• **Recognition:** {row['Recognition']:.0%} - Above 40% threshold indicating strong consumer familiarity")
                    st.write(f"• **Uniqueness:** {row['Uniqueness']:.0%} - Above 15% threshold showing distinctive Škoda attribution")
                    st.write(f"• **Usage Pattern:** {row['Overall Usage']:.0%} of campaigns - High use frequency")
                    st.write(f"• **Investment:** €{row['Total Investment']:,.0f} delivering {row['Recognition']:.0%} recognition ({row['Recognition ROI']:.2f} ROI)")
                    st.write(f"• **Sentiment:** {row['Net Sentiment']:+.1%} net sentiment from consumer associations")

                    st.markdown("**Why These Metrics Indicate Strong Performance:**")
                    st.write(f"Combined recognition ({row['Recognition']:.0%}) and uniqueness ({row['Uniqueness']:.0%}) create brand equity score of {equity_score:.3f}. High usage ({row['Overall Usage']:.0%}) demonstrates established use patterns across campaigns.")
        else:
            st.info("No elements currently meet all Category 1 criteria")

        st.markdown("---")

        st.markdown("### 🟡 Category 2: Strong Potential")
        st.info(f"**{len(strong_potential)} elements** show strong performance on recognition OR uniqueness metrics")

        if len(strong_potential) > 0:
            for idx, row in strong_potential.iterrows():
                with st.expander(f"**{row['Element']}** - Recognition: {row['Recognition']:.0%} | Uniqueness: {row['Uniqueness']:.0%}"):
                    st.markdown("**Performance Strengths:**")
                    if row['Recognition'] >= 0.35:
                        st.write(f"• ✅ Recognition at {row['Recognition']:.0%} (above 35% threshold)")
                    if row['Uniqueness'] >= 0.25:
                        st.write(f"• ✅ Uniqueness at {row['Uniqueness']:.0%} (above 25% threshold)")
                    st.write(f"• Current usage: {row['Overall Usage']:.0%} of campaigns")
                    st.write(f"• Investment: €{row['Total Investment']:,.0f}")
                    st.write(f"• ROI: {row['Recognition ROI']:.2f} per €1M")

                    st.markdown("**Performance Context:**")
                    if row['Recognition'] >= 0.35 and row['Uniqueness'] < 0.25:
                        st.write(f"High recognition ({row['Recognition']:.0%}) with uniqueness at {row['Uniqueness']:.0%} - Strong awareness with moderate attribution")
                    elif row['Uniqueness'] >= 0.25 and row['Recognition'] < 0.40:
                        st.write(f"Strong uniqueness ({row['Uniqueness']:.0%}) with recognition at {row['Recognition']:.0%} - Distinctive attribution with growing awareness")
                    else:
                        st.write(f"Balanced performance across recognition ({row['Recognition']:.0%}) and uniqueness ({row['Uniqueness']:.0%})")
        else:
            st.info("No elements in Category 2")

        st.markdown("---")

        st.markdown("### 🔴 Category 3: Development Opportunities")
        st.warning(f"**{len(development_opportunity)} elements** show recognition below 40% despite above-median investment")

        if len(development_opportunity) > 0:
            for idx, row in development_opportunity.iterrows():
                with st.expander(f"**{row['Element']}** - Recognition: {row['Recognition']:.0%} | Investment: €{row['Total Investment']:,.0f}"):
                    st.markdown("**Current Performance Metrics:**")
                    st.write(f"• Recognition: {row['Recognition']:.0%} (below 40% threshold)")
                    st.write(f"• Investment: €{row['Total Investment']:,.0f} (above median of €{master_df['Total Investment'].median():,.0f})")
                    st.write(f"• Usage: {row['Overall Usage']:.0%} of campaigns")
                    st.write(f"• Uniqueness: {row['Uniqueness']:.0%}")
                    st.write(f"• ROI: {row['Recognition ROI']:.2f} per €1M")

                    st.markdown("**Performance Context:**")
                    st.write(f"Investment of €{row['Total Investment']:,.0f} is {((row['Total Investment'] / master_df['Total Investment'].median()) - 1) * 100:+.0f}% vs median")
                    st.write(f"Recognition ROI of {row['Recognition ROI']:.2f} compares to portfolio best of {master_df['Recognition ROI'].max():.2f}")

                    st.markdown("**Possible Contributing Factors:**")
                    st.write("• Recent investment timing (recognition building over time)")
                    st.write("• Attribution complexity (generic appearance reducing distinctiveness)")
                    st.write("• Deployment patterns (frequency, prominence, or placement)")
                    st.write("• Creative execution (visibility within campaign materials)")

                    st.markdown("**Performance Pattern:**")
                    if row['Uniqueness'] < 0.20:
                        st.write(f"Uniqueness at {row['Uniqueness']:.0%} suggests attribution challenges - element may not be strongly distinctive")
                    else:
                        st.write(f"Uniqueness at {row['Uniqueness']:.0%} indicates some distinctiveness - recognition may improve with time or increased prominence")
        else:
            st.info("No elements in Category 3")

    with col2:
        st.markdown("### 📊 Framework Overview")

        # Category definitions
        with st.expander("📖 Category Criteria"):
            st.markdown("""
**Category 1 Criteria:**
- Recognition ≥ 40%
- Uniqueness ≥ 15%
- Usage ≥ 50%

**Category 2 Criteria:**
- Recognition ≥ 35% OR
- Uniqueness ≥ 25%
(Excluding Category 1 elements)

**Category 3 Criteria:**
- Recognition < 40% AND
- Investment > Portfolio Median
""")

        # Download framework data
        framework_data = []
        for idx, row in master_df.iterrows():
            if row['Element'] in high_performers['Element'].values:
                category = "Category 1: High Performer"
            elif row['Element'] in strong_potential['Element'].values:
                category = "Category 2: Strong Potential"
            elif row['Element'] in development_opportunity['Element'].values:
                category = "Category 3: Development"
            else:
                category = "Not Categorized"

            framework_data.append({
                'Element': row['Element'],
                'Category': category,
                'Recognition': f"{row['Recognition']:.0%}",
                'Uniqueness': f"{row['Uniqueness']:.0%}",
                'Usage': f"{row['Overall Usage']:.0%}",
                'Investment': f"€{row['Total Investment']:,.0f}",
                'ROI': f"{row['Recognition ROI']:.2f}"
            })

        framework_df = pd.DataFrame(framework_data)
        framework_csv = framework_df.to_csv(index=False)

        st.download_button(
            label="📥 Download Framework (CSV)",
            data=framework_csv,
            file_name="skoda_asset_performance_framework.csv",
            mime="text/csv",
            key="download_btn_tab4"
        )

with tab5:
    st.header("🔮 Growth Opportunity Analysis")
    st.caption("Find hidden opportunities and improve spend efficiency")

    # High Potential Assets (underutilized)
    high_potential = master_df[
        (master_df['Uniqueness'] >= 0.25) &
        (master_df['Overall Usage'] < 0.40)
    ].sort_values('Uniqueness', ascending=False)

    # Key Insights Box
    underutilized_list = ", ".join(high_potential['Element'].tolist()) if len(high_potential) > 0 else "None identified"
    best_roi_elem = master_df.nlargest(1, 'Recognition ROI').iloc[0]
    most_consistent = master_df.nsmallest(1, 'Cross-Market Variation').iloc[0] if 'Cross-Market Variation' in master_df.columns else None

    insights = [
        f"<b>{len(high_potential)} underutilized elements</b> with high uniqueness (≥25%) but low usage (<40%): {underutilized_list}",
        f"<b>{best_roi_elem['Element']}</b> shows best ROI efficiency at {best_roi_elem['Recognition ROI']:.2f} per €1M invested"
    ]

    if most_consistent is not None:
        insights.append(f"<b>{most_consistent['Element']}</b> demonstrates most consistent cross-market performance")
    else:
        insights.append(f"Cross-market analysis identifies opportunities for standardized vs. localized use")

    render_tldr_box("Key Insights at a Glance", insights)

    st.markdown("### 💎 Underutilized High-Uniqueness Elements")

    if len(high_potential) > 0:
        st.success(f"**{len(high_potential)} elements identified** with uniqueness ≥25% but usage <40%")

        for idx, row in high_potential.iterrows():
            with st.expander(f"**{row['Element']}** - Uniqueness: {row['Uniqueness']:.0%} | Current Usage: {row['Overall Usage']:.0%}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Uniqueness Score", f"{row['Uniqueness']:.0%}")
                    st.metric("Current Usage", f"{row['Overall Usage']:.0%}")
                    st.metric("Recognition", f"{row['Recognition']:.0%}")

                with col2:
                    st.metric("Brand Equity", f"{(row['Recognition'] * row['Uniqueness']):.3f}")
                    st.metric("Current Investment", f"€{row['Total Investment']:,.0f}")
                    st.metric("Recognition ROI", f"{row['Recognition ROI']:.2f}")

                # Calculate relative context
                median_usage = master_df['Overall Usage'].median()
                max_recognition = master_df['Recognition'].max()
                median_investment = master_df['Total Investment'].median()

                st.markdown("**Performance Context:**")
                st.write(f"• Uniqueness at {row['Uniqueness']:.0%} indicates strong Škoda attribution in consumer surveys")
                st.write(f"• Usage at {row['Overall Usage']:.0%} vs portfolio median of {median_usage:.0%}")
                st.write(f"• Recognition at {row['Recognition']:.0%} vs portfolio maximum of {max_recognition:.0%}")

                if row['Total Investment'] < median_investment:
                    st.write(f"• Investment at €{row['Total Investment']:,.0f} (below median €{median_investment:,.0f})")
                else:
                    st.write(f"• Investment at €{row['Total Investment']:,.0f} (above median €{median_investment:,.0f})")

                st.markdown("**Opportunity Pattern:**")
                st.write(f"High distinctiveness ({row['Uniqueness']:.0%} uniqueness) combined with below-median use ({row['Overall Usage']:.0%} usage) represents potential for increased use while maintaining who it belongs to strength")

                if row['Recognition ROI'] >= master_df['Recognition ROI'].median():
                    st.write(f"ROI of {row['Recognition ROI']:.2f} is above median, indicating efficient performance relative to investment")
    else:
        st.info("No elements meet criteria: uniqueness ≥25% and usage <40%")

    st.markdown("---")

    # Investment Efficiency Analysis
    st.markdown("### 💰 Investment Efficiency Distribution")

    with st.expander("📖 Understanding Efficiency Scores"):
        st.markdown("""
**Efficiency Score Formula:** (Recognition × Uniqueness) / Investment (in millions)

**What this measures:** Brand equity (combined recognition and differentiation) generated per million euros invested.

**Interpretation:**
- **High efficiency:** Strong brand equity results relative to investment level
- **Low efficiency:** Investment level high relative to brand equity outcome
""")

    # Calculate efficiency scores
    master_df_eff = master_df.copy()
    master_df_eff['Efficiency Score'] = (master_df_eff['Recognition'] * master_df_eff['Uniqueness']) / (master_df_eff['Total Investment'] / 1000000)
    master_df_eff['Efficiency Score'] = master_df_eff['Efficiency Score'].fillna(0)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📈 Highest Efficiency Elements")
        high_efficiency = master_df_eff.nlargest(3, 'Efficiency Score')

        for idx, row in high_efficiency.iterrows():
            st.success(f"**{row['Element']}**")
            st.write(f"• Efficiency Score: {row['Efficiency Score']:.2f}")
            st.write(f"• Investment: €{row['Total Investment']:,.0f}")
            st.write(f"• Brand Equity: {(row['Recognition'] * row['Uniqueness']):.3f}")

            with st.expander(f"Efficiency breakdown for {row['Element']}"):
                st.write(f"**Recognition:** {row['Recognition']:.0%}")
                st.write(f"**Uniqueness:** {row['Uniqueness']:.0%}")
                st.write(f"**Investment:** €{row['Total Investment']:,.0f}")
                st.write(f"**Usage:** {row['Overall Usage']:.0%}")
                st.markdown("**Efficiency Pattern:**")
                st.write(f"Delivers brand equity score of {(row['Recognition'] * row['Uniqueness']):.3f} with investment of €{row['Total Investment']:,.0f}. Each €1M generates {row['Efficiency Score']:.2f} units of combined recognition and uniqueness.")
                if row['Total Investment'] < master_df_eff['Total Investment'].median():
                    st.write(f"Investment below portfolio median suggests potential for scaling")

    with col2:
        st.markdown("#### 📉 Lowest Efficiency Elements")
        low_efficiency = master_df_eff.nsmallest(3, 'Efficiency Score')

        for idx, row in low_efficiency.iterrows():
            st.warning(f"**{row['Element']}**")
            st.write(f"• Efficiency Score: {row['Efficiency Score']:.2f}")
            st.write(f"• Investment: €{row['Total Investment']:,.0f}")
            st.write(f"• Brand Equity: {(row['Recognition'] * row['Uniqueness']):.3f}")

            with st.expander(f"Efficiency breakdown for {row['Element']}"):
                st.write(f"**Recognition:** {row['Recognition']:.0%}")
                st.write(f"**Uniqueness:** {row['Uniqueness']:.0%}")
                st.write(f"**Investment:** €{row['Total Investment']:,.0f}")
                st.write(f"**Usage:** {row['Overall Usage']:.0%}")
                st.markdown("**Efficiency Pattern:**")
                brand_equity = row['Recognition'] * row['Uniqueness']
                st.write(f"Investment of €{row['Total Investment']:,.0f} delivers brand equity of {brand_equity:.3f}")
                if row['Recognition'] < 0.40:
                    st.write(f"Recognition at {row['Recognition']:.0%} relative to investment level")
                if row['Uniqueness'] < 0.20:
                    st.write(f"Uniqueness at {row['Uniqueness']:.0%} indicates lower who it belongs to")
                st.write("**Possible factors:** Recent launch timing, creative visibility, or distinctiveness challenges")

    st.markdown("---")

    # Consistency Analysis
    st.markdown("### 🎯 Cross-Market Consistency Patterns")

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
            'Consistency Score': 1 - std_dev
        })

    consistency_df = pd.DataFrame(consistency_data).sort_values('Consistency Score')

    st.info("**Elements with highest usage variation across markets** (standard deviation analysis)")

    # Create bar chart of variation
    fig_consistency = px.bar(
        consistency_df,
        x='Element',
        y='Std Dev',
        title='Usage Variation Across Markets (Standard Deviation)',
        text=consistency_df['Std Dev'].apply(lambda x: f'{x:.2f}'),
        color='Std Dev',
        color_continuous_scale='Reds',
        hover_data={'Avg Usage': ':.0%', 'Std Dev': ':.3f'}
    )
    fig_consistency = apply_standard_chart_styling(fig_consistency, "")
    fig_consistency.update_traces(textposition='outside')
    fig_consistency.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_consistency, use_container_width=True, config=get_standard_chart_config())

    st.markdown("**Top 5 Elements by Usage Variation:**")
    for idx, row in consistency_df.head(5).iterrows():
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.write(f"**{row['Element']}**")
        with col2:
            st.metric("Std Dev", f"{row['Std Dev']:.3f}")
        with col3:
            st.caption(f"Average usage: {row['Avg Usage']:.0%} across markets")

    st.markdown("---")

    # Summary metrics
    st.markdown("### 📊 Portfolio Growth Metrics Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Underutilized High-Uniqueness",
            len(high_potential),
            help="Distinctive but underused"
        )

    with col2:
        efficiency_range = master_df_eff['Efficiency Score'].max() - master_df_eff['Efficiency Score'].min()
        st.metric(
            "Efficiency Score Range",
            f"{efficiency_range:.2f}",
            help="Difference between highest and lowest efficiency scores"
        )

    with col3:
        avg_consistency = consistency_df['Consistency Score'].mean()
        st.metric(
            "Avg Consistency Score",
            f"{avg_consistency:.2f}",
            help="Average consistency across all elements (1 - std dev)"
        )

# ==================== TAB 6: DEEP DIVE ANALYSIS ====================
with tab6:
    st.header("🔍 Deep Dive Analysis")
    st.caption("Explore data your way with filters")

    # Calculate key metrics for insights
    total_ads = len(audit_df)
    markets = audit_df['Market'].nunique()
    media = audit_df['Medium'].nunique()
    total_spend = audit_df['Spend'].sum()
    most_used_elem = audit_df[brand_elements].sum().idxmax()
    most_used_count = audit_df[brand_elements].sum().max()

    # Key Insights Box
    render_tldr_box(
        "Key Insights at a Glance",
        [
            f"<b>{total_ads} ads analyzed</b> across {markets} markets and {media} media types with €{total_spend:,.0f} total spend",
            f"<b>{most_used_elem}</b> is most frequently deployed appearing in {most_used_count:.0f} ads ({most_used_count/total_ads:.0%} usage rate)",
            f"<b>Custom filtering available</b> to analyze investment patterns by market, medium, and placement combinations"
        ]
    )

    # ============ FILTERS ============
    st.markdown("### 🎯 Filters")
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

    # ============ INVESTMENT & USAGE ============
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 💰 Investment by Element")
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
        fig_invest = apply_standard_chart_styling(fig_invest, "")
        fig_invest.update_traces(textposition='outside', marker_color='#2196F3')
        fig_invest.update_layout(height=400)
        st.plotly_chart(fig_invest, use_container_width=True, config=get_standard_chart_config())

    with col2:
        st.markdown("#### 📊 Usage Frequency")
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
        fig_usage = apply_standard_chart_styling(fig_usage, "")
        fig_usage.update_traces(textposition='outside', marker_color='#4CAF50')
        fig_usage.update_layout(height=400)
        st.plotly_chart(fig_usage, use_container_width=True, config=get_standard_chart_config())

    st.markdown("---")

    # ============ BRAND PERSONALITY ANALYSIS ============
    st.markdown("### 🎨 Brand Personality Analysis")

    with st.expander("💡 About personality attributes"):
        st.markdown("""
        These 7 personality dimensions (Bold, Stylish, Modern, Simple, Human, Exciting, Playful) reveal the **emotional character** of each brand element.

        **What the scores show:**
        - **Emotional connection** patterns beyond rational features
        - **Personality consistency** across different assets
        - **Differentiation** through unique personality, not just visual recognition
        - **Message alignment** potential (e.g., "Exciting" aligns with launch campaigns, "Simple" with practical messaging)

        **Reading the data:**
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
            st.plotly_chart(fig_radar, use_container_width=True, config=get_standard_chart_config())

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
            fig_personality = apply_standard_chart_styling(fig_personality, "")
            fig_personality.update_traces(textposition='outside')
            fig_personality.update_layout(height=500)
            st.plotly_chart(fig_personality, use_container_width=True, config=get_standard_chart_config())

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

                # Sort by positive percentage (descending)
                bar_data.sort(key=lambda x: x['positive'], reverse=True)

                # Show key insights ABOVE the chart
                pos_strengths = [(item['pos_adj'].title(), item['positive']) for item in bar_data]
                neg_strengths = [(item['neg_adj'], abs(item['negative'])) for item in bar_data]

                pos_strengths.sort(key=lambda x: x[1], reverse=True)
                neg_strengths.sort(key=lambda x: x[1], reverse=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Top Positive:** {pos_strengths[0][0]} ({pos_strengths[0][1]:.0%})")
                with col2:
                    if neg_strengths[0][1] > 0.15:
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

                st.plotly_chart(fig_diverging, use_container_width=True, config=get_standard_chart_config())

                st.markdown("---")

            st.info("**How to read this chart:** Green bars (right) show % who chose positive adjectives (Bold, Stylish, Modern, etc.). " +
                    "Red bars (left) show % who chose negative adjectives (Cautious, Plain, Old-Fashioned, etc.). " +
                    "Longer bars indicate stronger associations. The center represents neutral responses.")

    st.markdown("---")

    # ============ RECOGNITION BY MARKET ============
    st.markdown("### 🌍 Recognition by Market")
    st.caption("How brand elements perform across different countries")

    # Add demographic selector for market recognition
    filters_market = render_demographic_filters(prefix="market_deep", use_global=False)

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
        fig_heatmap = apply_standard_chart_styling(fig_heatmap, "")
        fig_heatmap.update_layout(height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True, config=get_standard_chart_config())

    with col2:
        st.markdown("#### 📊 Key Patterns:")

        # Find strongest market per element
        st.markdown("**Strongest Markets:**")
        for element in brand_elements[:5]:
            countries_sorted = sorted(
                recognition_by_country[element].items(),
                key=lambda x: x[1],
                reverse=True
            )
            best_country = countries_sorted[0]
            st.success(f"**{element}**: {best_country[0]} ({best_country[1]:.0%})")

        st.markdown("**Variation Patterns:**")
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
                st.markdown("**Possible factors:**")
                st.write(f"• Market maturity differences: {max_c[0]} may have longer Škoda brand presence")
                st.write(f"• Media mix variations: {element} may feature more prominently in {max_c[0]} campaigns")
                st.write(f"• Cultural relevance: Design/messaging resonance differs across markets")
                st.write(f"• Competitive landscape: {min_c[0]} may have stronger local competitors")

    st.markdown("---")

    # ============ BRAND ATTRIBUTION (UNIQUENESS) BY MARKET ============
    if uniqueness_by_country:
        st.markdown("### 🎯 Brand Attribution (Uniqueness) by Market")
        st.caption("Shows which markets correctly identify each element as belonging to Škoda (not competitors)")

        with st.expander("📖 About market-level uniqueness"):
            st.markdown("""
            **Uniqueness** measures who it belongs to - the % of people who correctly identify an element as belonging to Škoda (vs competitors or generic design).

            **Why market variations matter:**
            - **Significant differences exist** between markets (e.g., Symbol: UK 23% vs Poland 55%)
            - Global averages can mask these variations
            - **Strategic implications:** Elements may need market-specific support or repositioning
            - **Investment context:** High-uniqueness markets can leverage assets more effectively

            **What to observe:**
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
            fig_uniqueness_heatmap = apply_standard_chart_styling(fig_uniqueness_heatmap, "")
            fig_uniqueness_heatmap.update_layout(height=500)
            st.plotly_chart(fig_uniqueness_heatmap, use_container_width=True, config=get_standard_chart_config())

        with col2:
            st.markdown("#### 📊 Key Patterns:")

            # Find strongest market per element for uniqueness
            st.markdown("**Strongest Attribution:**")
            for element in brand_elements[:5]:
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
                    st.markdown("**Attribution context:**")
                    if min_c[1] < 0.30:
                        st.warning(f"⚠️ In {min_c[0]}, consumers don't strongly associate {element} with Škoda - potential competitor confusion")
                    st.markdown("**Observed patterns:**")
                    st.write(f"• {max_c[0]} shows {max_c[1]:.0%} attribution - strong Škoda association")
                    st.write(f"• {min_c[0]} at {min_c[1]:.0%} attribution - weaker brand linkage")
                    st.write(f"• Co-branding with Symbol/Wordmark may strengthen association in {min_c[0]}")

    st.markdown("---")

    # ============ MARKET CONSISTENCY SCORE ============
    st.markdown("### 📊 Market Consistency: Which Assets Travel Well?")
    st.caption("Identifies which brand elements perform consistently across markets vs those with market-specific patterns")

    with st.expander("📖 About market consistency"):
        st.markdown("""
        **Market consistency** reveals which assets are "universal" vs "market-specific":

        **Universal assets** (low variation):
        - Perform similarly across all markets
        - Suitable for global campaigns
        - Easier to scale internationally
        - Example: Symbol recognition 44-55% across markets (11% variation)

        **Market-specific assets** (high variation):
        - Performance varies significantly by market
        - May require market-specific strategies
        - Could benefit from localization or repositioning
        - Example: Element X: 15% in UK, 45% in Poland (30% variation)

        **Strategic value:**
        - Identify assets for global rollout
        - Spot markets needing special attention
        - Optimize creative for regional differences
        """)

    # Calculate consistency scores for both recognition and uniqueness
    consistency_data_market = []

    for element in brand_elements:
        row = {'Element': element}

        # Recognition consistency
        if element in recognition_by_country:
            rec_values = list(recognition_by_country[element].values())
            if rec_values:
                rec_mean = sum(rec_values) / len(rec_values)
                rec_std = pd.Series(rec_values).std()
                rec_coef_var = (rec_std / rec_mean) if rec_mean > 0 else 0
                row['Recognition Mean'] = rec_mean
                row['Recognition StdDev'] = rec_std
                row['Recognition Consistency'] = 1 - rec_coef_var

        # Uniqueness consistency
        if element in uniqueness_by_country:
            uniq_values = list(uniqueness_by_country[element].values())
            if uniq_values:
                uniq_mean = sum(uniq_values) / len(uniq_values)
                uniq_std = pd.Series(uniq_values).std()
                uniq_coef_var = (uniq_std / uniq_mean) if uniq_mean > 0 else 0
                row['Uniqueness Mean'] = uniq_mean
                row['Uniqueness StdDev'] = uniq_std
                row['Uniqueness Consistency'] = 1 - uniq_coef_var

        consistency_data_market.append(row)

    consistency_df_market = pd.DataFrame(consistency_data_market)

    # Display consistency rankings
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌍 Most Consistent Assets")
        st.caption("Low variation across markets")

        if 'Recognition Consistency' in consistency_df_market.columns:
            consistent_assets = consistency_df_market.nlargest(5, 'Recognition Consistency')
            for _, row in consistent_assets.iterrows():
                st.success(f"**{row['Element']}**")
                st.write(f"  • Avg Recognition: {row['Recognition Mean']:.0%}")
                st.write(f"  • Variation: ±{row['Recognition StdDev']:.1%}")
                if 'Uniqueness Mean' in row:
                    st.write(f"  • Avg Uniqueness: {row['Uniqueness Mean']:.0%}")

    with col2:
        st.markdown("#### 🗺️ Most Variable Assets")
        st.caption("High variation - market-specific patterns")

        if 'Recognition Consistency' in consistency_df_market.columns:
            variable_assets = consistency_df_market.nsmallest(5, 'Recognition Consistency')
            for _, row in variable_assets.iterrows():
                st.warning(f"**{row['Element']}**")
                st.write(f"  • Avg Recognition: {row['Recognition Mean']:.0%}")
                st.write(f"  • Variation: ±{row['Recognition StdDev']:.1%}")
                if row['Element'] in recognition_by_country:
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
        if 'Recognition Consistency' in consistency_df_market.columns:
            rec_consistency = consistency_df_market[['Element', 'Recognition Consistency', 'Recognition StdDev']].sort_values('Recognition Consistency', ascending=True)

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

            st.plotly_chart(fig_rec_consistency, use_container_width=True, config=get_standard_chart_config())

    with col2:
        # Uniqueness consistency bar chart
        if 'Uniqueness Consistency' in consistency_df_market.columns:
            uniq_consistency = consistency_df_market[['Element', 'Uniqueness Consistency', 'Uniqueness StdDev']].sort_values('Uniqueness Consistency', ascending=True)

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

            st.plotly_chart(fig_uniq_consistency, use_container_width=True, config=get_standard_chart_config())

    # Patterns observed
    st.info("""
    **Observed Patterns:**

    **For Consistent Assets (Low Variation):**
    - Similar performance across markets
    - Suitable for standardized global campaigns
    - Learnings from one market likely apply to others

    **For Variable Assets (High Variation):**
    - "Hero markets" exist where asset performs well
    - Market-specific support plans may be beneficial
    - Localized creative adaptations could be considered
    """)

# ==================== TAB 7: RECOGNITION JOURNEY ====================
with tab7:
    st.header("🧭 Recognition Journey & Brand Discovery")
    st.caption("How people discover and connect with Škoda")

    # Key Insights Box
    render_tldr_box(
        "Key Insights at a Glance",
        [
            f"<b>56.3% never recognized</b> these elements as Škoda even after seeing 6 different brand elements",
            f"<b>Only 10.3% recognize</b> Škoda from a single element; recognition builds to 40.1% after all 6 elements",
            f"<b>33% familiar with Škoda</b> while 46% have heard the name but lack knowledge — revealing brand education opportunity"
        ]
    )

    # Critical finding callout
    st.error("""
    ### ⚠️ Critical Research Finding
    **56.3% of respondents never recognized these elements as Škoda** — even after seeing 6 different brand elements.

    This finding reveals:
    - The challenge of brand recognition in the automotive market
    - The significance of the Symbol (48% recognition) as the primary brand carrier
    - The importance of multiple touchpoints working together
    - The potential to strengthen brand identity through strategic asset use
    """)

    st.markdown("---")

    # ============ SECTION 1: RECOGNITION JOURNEY ============
    st.markdown("### 📈 The Recognition Build: When Do People Identify Škoda?")
    st.caption("Tracking how recognition accumulates as respondents see more brand elements")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Create waterfall-style visualization
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

        # Never recognized (red)
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

        # Recognition builders (green)
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

        st.plotly_chart(fig_journey, use_container_width=True, config=get_standard_chart_config())

    with col2:
        st.markdown("#### 🔍 Key Observations")

        st.metric("Immediate Recognition", "10.3%", "After just 1 element")
        st.caption("Only 1 in 10 recognize Škoda from a single brand element")

        st.metric("Maximum Recognition", "40.1%", "After all 6 elements")
        st.caption("Even with 6 touchpoints, less than half recognize the brand")

        st.metric("Never Recognized", "56.3%", delta="-56.3%", delta_color="inverse")
        st.caption("**Pattern observed:** More than half never connect elements to Škoda")

        st.markdown("---")

        st.markdown("#### 💡 What This Shows")
        st.markdown("""
        **Recognition patterns observed:**
        1. **Single elements are insufficient** - Recognition requires multiple exposures
        2. **Symbol shows highest recognition** - At 48%, it's the strongest individual carrier
        3. **Cumulative effect exists** - Each additional element adds ~5-7% recognition
        4. **56% gap represents opportunity** - Making assets more distinctively Škoda
        """)

    st.markdown("---")

    # ============ SECTION 2: FIRST RECOGNITION TRIGGER INDEX ============
    if first_recognition_trigger or recognition_by_age_gender:
        st.markdown("### 🎯 First Recognition Trigger Index")
        st.caption("Which elements are most likely to trigger brand recognition when shown first?")

        st.info("""
        **What this analysis shows:** Which brand elements are most effective at triggering
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

                st.plotly_chart(fig_trigger, use_container_width=True, config=get_standard_chart_config())
            else:
                st.warning("First recognition trigger data not available")

        with col2:
            st.markdown("#### 🔍 Key Pattern")

            if first_recognition_trigger:
                # Get top trigger
                top_trigger = max(first_recognition_trigger.items(), key=lambda x: x[1]['percent_of_total_first_triggers'])
                st.success(f"**Top Trigger:** {top_trigger[0]}")
                st.metric("% of First Recognitions", f"{top_trigger[1]['percent_of_total_first_triggers']:.1%}")
                st.caption(f"{top_trigger[1]['count']} people recognized Škoda when shown this element first")

                st.markdown("---")

                st.markdown("#### 💡 Pattern Observed")
                st.markdown(f"""
                **{top_trigger[0]}** shows strongest "first impression" effectiveness:
                - Highest recognition rate when shown as initial element
                - Most effective for teaser campaigns and new market launches
                - Priority for media with limited brand exposure time
                - Prominent placement ensures maximum brand linkage
                """)

        st.markdown("---")

        # ============ AGE MIGRATION ANALYSIS ============
        if recognition_by_age_gender and uniqueness_by_age_gender:
            st.markdown("### 👥 Age Cohort Analysis: Recognition Patterns Across Generations")
            st.caption("Shows which elements trigger recognition for different age groups and how distinctiveness varies")

            with st.expander("📖 About age cohort patterns"):
                st.markdown("""
                **Age cohort analysis** reveals how brand recognition patterns shift across generations:

                - **Different age groups** may recognize different brand elements first
                - **Recognition rates** for the same element vary by cohort (e.g., younger audiences may respond to modern elements)
                - **Distinctiveness (uniqueness)** also varies - what feels "Škoda" to 18-30 may differ from 43-55
                - **Strategic application:** Tailor asset use to target demographics

                This analysis shows:
                1. Which elements resonate with each age group
                2. Generation gaps in brand recognition
                3. Optimization opportunities for specific audience segments
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
                fig_rec_age = apply_standard_chart_styling(fig_rec_age, "")
                fig_rec_age.update_layout(height=400)
                st.plotly_chart(fig_rec_age, use_container_width=True, config=get_standard_chart_config())

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
                fig_uniq_age = apply_standard_chart_styling(fig_uniq_age, "")
                fig_uniq_age.update_layout(height=400)
                st.plotly_chart(fig_uniq_age, use_container_width=True, config=get_standard_chart_config())

            # Key insights
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📊 Recognition Patterns")
                if all(col in migration_df.columns for col in recognition_cols):
                    for element in brand_elements[:3]:
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
                    for element in brand_elements[:3]:
                        element_row = migration_df[migration_df['Element'] == element].iloc[0]
                        values = [element_row[col] for col in uniqueness_cols if col in element_row]
                        if values:
                            max_age_idx = values.index(max(values))
                            min_age_idx = values.index(min(values))
                            st.write(f"**{element}:**")
                            st.write(f"  • Most distinctive: {age_groups[max_age_idx]} ({values[max_age_idx]:.0%})")
                            st.write(f"  • Least distinctive: {age_groups[min_age_idx]} ({values[min_age_idx]:.0%})")

            st.info("""
            **Observed Pattern:** Age cohort data shows:
            - Younger audiences (18-30) may respond to elements with high recognition and uniqueness in their cohort
            - Traditional elements (like Symbol) often perform better with older cohorts
            - Cross-generational assets work consistently across all age groups
            """)

        st.markdown("---")

    # ============ SECTION 3: POST-REVEAL BRAND FAMILIARITY ============
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

        st.plotly_chart(fig_familiarity, use_container_width=True, config=get_standard_chart_config())

    with col2:
        st.markdown("#### 📊 Familiarity Breakdown")

        familiar_total = skoda_familiarity['very_familiar'] + skoda_familiarity['quite_familiar']
        st.metric("Familiar with Brand", f"{familiar_total:.0%}", "Very + Quite familiar")

        st.metric("Heard Name Only", f"{skoda_familiarity['heard_of_not_much']:.0%}", "Lack deeper knowledge")

        st.metric("Completely Unaware", f"{skoda_familiarity['never_heard']:.0%}", delta=f"-{skoda_familiarity['never_heard']:.0%}", delta_color="inverse")

        st.markdown("---")

        st.info("""
        **Familiarity Context:**

        Only **33% are familiar** with Škoda, while **46% have heard the name but lack knowledge**.

        This pattern helps explain why recognition scores are lower and highlights the opportunity for brand education.
        """)

    st.markdown("---")

    # ============ SECTION 4: EMOTIONAL RESPONSE TO BRAND REVEAL ============
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

        st.plotly_chart(fig_response, use_container_width=True, config=get_standard_chart_config())

    with col2:
        st.markdown("#### 🎭 Response Summary")

        st.metric("Fits Expectations", f"{response_to_reveal['fits_expectations']:.0%}", "Aligns with Škoda brand")

        st.metric("Does Not Fit", f"{response_to_reveal['does_not_fit']:.0%}", "Conflicts with brand perception")

        st.metric("Unaware of Škoda", f"{response_to_reveal['not_heard_of_skoda']:.0%}", "No prior brand knowledge")

        st.markdown("---")

        st.warning("""
        **Emotional Pattern:**

        **42% felt neutral** when learning these are Škoda elements.

        Combined with 47% positive reactions, this indicates the brand has moderate emotional connection strength.
        """)

    st.markdown("---")

    # ============ SECTION 5: INTEGRATED STRATEGIC VIEW ============
    st.markdown("### 🎯 Integrated View: The Complete Picture")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Recognition Pattern")
        st.markdown("""
        - **56%** never identify elements as Škoda
        - **10%** recognize after 1 element
        - **40%** maximum with 6 elements

        **Pattern:** Multiple touchpoints essential; Symbol leads recognition
        """)

    with col2:
        st.markdown("#### Awareness Pattern")
        st.markdown("""
        - **33%** familiar with brand
        - **46%** heard name only
        - **18%** completely unaware

        **Pattern:** Brand education opportunity; not just recognition issue
        """)

    with col3:
        st.markdown("#### Engagement Pattern")
        st.markdown("""
        - **47%** positive reaction
        - **42%** indifferent
        - **3%** disappointed

        **Pattern:** Strengthen emotional positioning; brand not rejected but not strongly connected
        """)

    st.markdown("---")

    # Key observations
    st.info("""
    ### 🔍 Key Observations Based on This Data

    **Pattern 1: Symbol as Primary Carrier** - At 48% recognition vs 20% average, the logo shows the strongest brand linkage. Prominent placement across all communications appears critical.

    **Pattern 2: Combination Effect** - Since single elements drive only 10% recognition, multiple elements appearing together show stronger effect. Minimum 3 elements per touchpoint observed in data.

    **Pattern 3: The 56% Recognition Gap** - More than half never connect elements to Škoda. This pattern suggests:
       - Bolder, more distinctive asset design potential
       - More consistent usage across markets needed
       - Stronger connection between elements and brand name

    **Pattern 4: Familiarity vs Recognition** - 46% have heard of Škoda but know little. Brand elements serve as educational tools, not just identity markers.

    **Pattern 5: Moderate Emotional Connection** - 42% neutral response suggests opportunity to move beyond functional attributes to emotional benefits in messaging.
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
        mime="text/csv",
        key="download_btn_journey"
    )


# =====================================================================
# FOOTER
# =====================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
<p><b>Škoda Brand Intelligence Dashboard</b></p>
<p style='font-size: 0.9em;'>© 2025 Saffron Brand Consultants</p>
</div>
""", unsafe_allow_html=True)
