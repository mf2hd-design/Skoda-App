"""
Element Combination Effectiveness Data
Generated from Savanta survey P045556 - Additive element testing

This module contains pre-calculated metrics for element combination performance
across different demographic segments.

Data source: 2,011 respondents across UK, Spain, Germany, Poland
Methodology: Additive testing - elements shown sequentially, recognition tracked at each step
"""

import pickle

# Load the pre-calculated metrics
def load_metrics():
    """Load all combination effectiveness metrics"""
    import os

    # Try project directory first, then /tmp/ as fallback
    possible_paths = [
        'combination_metrics.pkl',  # Project directory
        '/tmp/skoda_combination_metrics.pkl',  # Fallback for temp processing
        os.path.join(os.path.dirname(__file__), 'combination_metrics.pkl')  # Explicit path
    ]

    for path in possible_paths:
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            continue

    print("Warning: Metrics file not found. Please run the data processing script.")
    print(f"Tried paths: {possible_paths}")
    return None

# Element definitions
ELEMENTS_GROUP_1 = {
    1: "Electric Green",
    2: "Emerald Facets",
    3: "Type",
    4: "Symbol",
    5: "Sonic",
    6: "Wordmark"
}

ELEMENTS_GROUP_2 = {
    7: "Emerald Green",
    8: "Háček",
    9: "Tagline",
    4: "Symbol",
    5: "Sonic",
    6: "Wordmark"
}

# All unique elements tested
ALL_ELEMENTS = [
    "Electric Green",
    "Emerald Green",
    "Emerald Facets",
    "Type",
    "Symbol",
    "Sonic",
    "Wordmark",
    "Háček",
    "Tagline"
]

# Randomization sequences
RANDOMIZATION_SEQUENCES = {
    1: [1, 2, 3, 4, 5, 6],   # Group 1, order 1
    2: [2, 5, 4, 1, 6, 3],   # Group 1, order 2
    3: [4, 6, 2, 5, 3, 1],   # Group 1, order 3
    4: [5, 1, 4, 2, 6, 3],   # Group 1, order 4
    5: [3, 5, 1, 4, 6, 2],   # Group 1, order 5
    6: [6, 2, 3, 4, 1, 5],   # Group 1, order 6
    7: [7, 8, 9, 4, 5, 6],   # Group 2, order 7
    8: [8, 5, 4, 7, 6, 9],   # Group 2, order 8
    9: [4, 6, 8, 5, 9, 7],   # Group 2, order 9
    10: [5, 7, 4, 8, 6, 9],  # Group 2, order 10
    11: [9, 5, 7, 4, 6, 8],  # Group 2, order 11
    12: [6, 8, 9, 4, 7, 5],  # Group 2, order 12
}

# Sample sizes by segment
SEGMENT_SIZES = {
    'Overall': 2011,
    'Škoda owner': 120,
    'Non-Škoda owner': 1223,
    'No car': 668,
    '18-30': 734,
    '31-42': 657,
    '43-55': 620,
    'UK': 501,
    'Spain': 502,
    'Germany': 505,
    'Poland': 503,
}

# Recognition summary (overall)
# At what point did respondents recognize Škoda?
RECOGNITION_SUMMARY = {
    'recognized_at_1_element': 194,   # 9.6%
    'recognized_at_2_elements': 15,   # 0.7%
    'recognized_at_3_elements': 253,  # 12.6%
    'recognized_at_4_elements': 313,  # 15.6%
    'recognized_at_5_elements': 52,   # 2.6%
    'recognized_at_6_elements': 17,   # 0.8%
    'never_recognized': 1167,         # 58.0%
    'total_respondents': 2011
}

# Key insights for recommendations
KEY_INSIGHTS = {
    'best_single_element': {
        'element': 'Symbol',
        'recognition_rate': 0.096,  # 9.6% recognized from Symbol alone
        'note': 'Only element that drives meaningful solo recognition'
    },
    'best_pair': {
        'elements': ['Symbol', 'Wordmark'],
        'recognition_count': 11,
        'note': 'Most effective 2-element combination'
    },
    'owner_advantage': {
        'owners_avg_elements': 3.8,  # Average elements needed for recognition
        'non_owners_avg_elements': 4.4,
        'difference': 0.6,
        'note': 'Owners recognize faster than non-owners'
    },
    'recognition_plateau': {
        'plateau_point': 4,
        'recognition_at_4': 0.44,  # 44% cumulative recognition
        'recognition_at_6': 0.48,  # 48% cumulative recognition
        'note': 'Diminishing returns after 4 elements'
    }
}

# Demographic segment definitions
DEMOGRAPHIC_SEGMENTS = {
    'ownership': ['Škoda owner', 'Non-Škoda owner', 'No car'],
    'age': ['18-30', '31-42', '43-55'],
    'country': ['UK', 'Spain', 'Germany', 'Poland'],
}

if __name__ == '__main__':
    # Test loading
    metrics = load_metrics()
    if metrics:
        print("✓ Successfully loaded combination effectiveness metrics")
        print(f"  - Recognition journey: {len(metrics['recognition_journey'])} segments")
        print(f"  - Combination metrics: {len(metrics['combination_metrics'])} combinations")
        print(f"  - Top combos by segment: {len(metrics['top_combos_by_segment'])} segments")
        print(f"  - Pair metrics: {len(metrics['pair_metrics'])} pairs")
    else:
        print("✗ Failed to load metrics")
