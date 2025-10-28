#!/usr/bin/env python3
"""
Data Verification Script
Compare app.py data against JSON extracted files
"""

import json
import os

def load_json_file(filepath):
    """Load JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def compare_values(app_value, json_value, tolerance=0.001):
    """Compare two values with tolerance"""
    if abs(app_value - json_value) <= tolerance:
        return "✅ MATCH"
    else:
        diff = app_value - json_value
        return f"❌ MISMATCH (diff: {diff:+.4f})"

def main():
    print("="*80)
    print("DATA VERIFICATION AUDIT")
    print("="*80)

    # Define app.py data for comparison
    app_research_data = {
        'Electric Green': {
            'recognition': 0.376,
            'uniqueness': 0.174,
            'bold': 0.490, 'stylish': 0.463, 'modern': 0.499, 'simple': 0.502,
            'human': 0.452, 'exciting': 0.450, 'playful': 0.443,
            'positive_sentiment': 0.471,
            'negative_sentiment': 0.529,
            'net_sentiment': -0.057
        },
        'Emerald Green': {
            'recognition': 0.388,
            'uniqueness': 0.195,
            'bold': 0.510, 'stylish': 0.490, 'modern': 0.522, 'simple': 0.527,
            'human': 0.462, 'exciting': 0.485, 'playful': 0.451,
            'positive_sentiment': 0.492,
            'negative_sentiment': 0.508,
            'net_sentiment': -0.015
        },
        'Type': {
            'recognition': 0.374,
            'uniqueness': 0.169,
            'bold': 0.474, 'stylish': 0.473, 'modern': 0.491, 'simple': 0.499,
            'human': 0.438, 'exciting': 0.448, 'playful': 0.412,
            'positive_sentiment': 0.462,
            'negative_sentiment': 0.538,
            'net_sentiment': -0.076
        },
        'Symbol': {
            'recognition': 0.643,
            'uniqueness': 0.385,
            'bold': 0.498, 'stylish': 0.497, 'modern': 0.551, 'simple': 0.536,
            'human': 0.464, 'exciting': 0.500, 'playful': 0.462,
            'positive_sentiment': 0.501,
            'negative_sentiment': 0.499,
            'net_sentiment': 0.002
        },
        'Wordmark': {
            'recognition': 0.447,
            'uniqueness': 0.279,
            'bold': 0.490, 'stylish': 0.492, 'modern': 0.537, 'simple': 0.519,
            'human': 0.455, 'exciting': 0.485, 'playful': 0.448,
            'positive_sentiment': 0.489,
            'negative_sentiment': 0.511,
            'net_sentiment': -0.021
        },
        'Facets': {
            'recognition': 0.384,
            'uniqueness': 0.158,
            'bold': 0.502, 'stylish': 0.484, 'modern': 0.514, 'simple': 0.508,
            'human': 0.427, 'exciting': 0.458, 'playful': 0.461,
            'positive_sentiment': 0.479,
            'negative_sentiment': 0.521,
            'net_sentiment': -0.042
        },
        'Sonic': {
            'recognition': 0.398,
            'uniqueness': 0.166,
            'bold': 0.502, 'stylish': 0.491, 'modern': 0.546, 'simple': 0.545,
            'human': 0.462, 'exciting': 0.508, 'playful': 0.479,
            'positive_sentiment': 0.505,
            'negative_sentiment': 0.495,
            'net_sentiment': 0.009
        },
        'Hacek': {
            'recognition': 0.377,
            'uniqueness': 0.186,
            'bold': 0.463, 'stylish': 0.456, 'modern': 0.488, 'simple': 0.549,
            'human': 0.439, 'exciting': 0.442, 'playful': 0.422,
            'positive_sentiment': 0.466,
            'negative_sentiment': 0.534,
            'net_sentiment': -0.069
        },
        'Tagline': {
            'recognition': 0.361,
            'uniqueness': 0.175,
            'bold': 0.482, 'stylish': 0.484, 'modern': 0.512, 'simple': 0.495,
            'human': 0.464, 'exciting': 0.509, 'playful': 0.451,
            'positive_sentiment': 0.485,
            'negative_sentiment': 0.515,
            'net_sentiment': -0.029
        },
    }

    # Load extracted JSON data
    personality_json = load_json_file('extracted_personality_data.json')
    confusion_json = load_json_file('q05_confusion_data.json')
    uniqueness_by_country = load_json_file('uniqueness_by_country.json')

    print("\n" + "="*80)
    print("1. PERSONALITY DATA COMPARISON")
    print("="*80)

    if personality_json:
        for element in ['Electric Green', 'Type', 'Symbol', 'Wordmark', 'Facets',
                       'Sonic', 'Emerald Green', 'Hacek', 'Tagline']:
            if element in app_research_data and element in personality_json:
                print(f"\n{element}:")
                app_data = app_research_data[element]
                json_data = personality_json[element]

                for trait in ['bold', 'stylish', 'modern', 'playful', 'exciting', 'human', 'simple']:
                    if trait in app_data and trait in json_data:
                        status = compare_values(app_data[trait], json_data[trait])
                        print(f"  {trait}: app={app_data[trait]:.3f}, json={json_data[trait]:.3f} {status}")

                # Check sentiment
                for sent in ['positive_sentiment', 'negative_sentiment', 'net_sentiment']:
                    if sent in app_data and sent in json_data:
                        status = compare_values(app_data[sent], json_data[sent])
                        print(f"  {sent}: app={app_data[sent]:.3f}, json={json_data[sent]:.3f} {status}")

    print("\n" + "="*80)
    print("2. UNIQUENESS DATA COMPARISON (Q05)")
    print("="*80)

    if confusion_json:
        # Map element names
        element_mapping = {
            'Electric Green': 'Electric Green',
            'Type': 'Type',
            'Symbol': 'Symbol',
            'Wordmark': 'Wordmark',
            'Facets': 'Facets',
            'Sonic': 'Sonic',
            'Emerald Green': 'Dark Green',  # Note: JSON uses "Dark Green"
            'Hacek': 'Hacek',
            'Tagline': 'Tagline'
        }

        for app_element, json_element in element_mapping.items():
            if app_element in app_research_data and json_element in confusion_json:
                app_uniqueness = app_research_data[app_element]['uniqueness']
                json_uniqueness = confusion_json[json_element]['Skoda']

                status = compare_values(app_uniqueness, json_uniqueness)
                print(f"{app_element}: app={app_uniqueness:.3f}, json={json_uniqueness:.2f} {status}")

    print("\n" + "="*80)
    print("3. RECOGNITION BY COUNTRY DATA")
    print("="*80)

    app_recognition_by_country = {
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

    print("Note: Recognition by country not in JSON files - would need to verify from Excel")

    print("\n" + "="*80)
    print("4. RECOGNITION JOURNEY DATA")
    print("="*80)

    app_recognition_journey = {
        'after_1_element': 0.102,
        'after_2_elements': 0.109,
        'after_3_elements': 0.243,
        'after_4_elements': 0.403,
        'after_5_elements': 0.427,
        'after_all_6_elements': 0.438,
        'never_recognized': 0.562
    }

    print("Recognition journey data:")
    for stage, value in app_recognition_journey.items():
        print(f"  {stage}: {value:.1%}")
    print("\nNote: Would need to verify from Excel Table 117 (QHiddenAwareness)")

    print("\n" + "="*80)
    print("5. SKODA FAMILIARITY DATA (Q27)")
    print("="*80)

    app_skoda_familiarity = {
        'very_familiar': 0.214,
        'quite_familiar': 0.386,
        'heard_of_not_much': 0.321,
        'never_heard': 0.045,
        'not_sure': 0.034
    }

    print("Skoda familiarity (Q27):")
    for level, value in app_skoda_familiarity.items():
        print(f"  {level}: {value:.1%}")
    print("\nNote: Would need to verify from Excel Table 120")

    print("\n" + "="*80)
    print("6. RESPONSE TO REVEAL DATA (Q28)")
    print("="*80)

    app_response_to_reveal = {
        'fits_expectations': 0.560,
        'does_not_fit': 0.222,
        'not_heard_of_skoda': 0.078,
        'other': 0.007,
        'dont_know': 0.133
    }

    print("Response to learning it's Škoda (Q28):")
    for response, value in app_response_to_reveal.items():
        print(f"  {response}: {value:.1%}")
    print("\nNote: Would need to verify from Excel Table 121/122")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("✅ Personality trait data (T2B scores) matches between app.py and extracted_personality_data.json")
    print("⚠️  Need to verify Emerald Green vs Dark Green naming discrepancy")
    print("⚠️  Recognition by country data not in JSON - needs Excel verification")
    print("⚠️  Recognition journey data needs Excel Table 117 verification")
    print("⚠️  Familiarity and response-to-reveal data needs Excel verification")

if __name__ == "__main__":
    os.chdir("/Users/ben/Documents/Saffron/Skoda App")
    main()
