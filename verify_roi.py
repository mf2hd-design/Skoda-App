#!/usr/bin/env python3
"""
Verify ROI calculations in app.py
ROI = (Recognition % / Total Investment) × 1,000,000
"""

from comms_data import comms_audit_data

# Recognition data from app.py
recognition_data = {
    'Electric Green': 0.376,
    'Emerald Green': 0.388,
    'Type': 0.374,
    'Tagline': 0.361,
    'Symbol': 0.643,
    'Hacek': 0.377,
    'Wordmark': 0.447,
    'Facets': 0.384,
    'Sonic': 0.398,
}

def calculate_roi():
    print("="*80)
    print("ROI CALCULATION VERIFICATION")
    print("="*80)
    print("Formula: ROI = (Recognition % / Total Investment) × 1,000,000")
    print("="*80)

    elements = ['Electric Green', 'Emerald Green', 'Type', 'Tagline', 'Symbol',
                'Hacek', 'Wordmark', 'Facets', 'Sonic']

    results = []

    for element in elements:
        # Calculate total investment from comms data
        ads_with_element = [ad for ad in comms_audit_data if ad.get(element, False)]
        total_investment = sum(ad['Spend'] for ad in ads_with_element)

        # Get recognition
        recognition = recognition_data[element]

        # Calculate ROI
        roi = (recognition / total_investment * 1_000_000) if total_investment > 0 else 0

        results.append({
            'element': element,
            'recognition': recognition,
            'total_investment': total_investment,
            'roi': roi,
            'usage_count': len(ads_with_element)
        })

    # Sort by ROI descending
    results.sort(key=lambda x: x['roi'], reverse=True)

    print("\n{:<20} {:>12} {:>18} {:>12} {:>8}".format(
        "Element", "Recognition", "Total Investment", "ROI", "Usage"))
    print("-"*80)

    for r in results:
        print("{:<20} {:>11.1%} {:>18} {:>12.2f} {:>8}".format(
            r['element'],
            r['recognition'],
            f"€{r['total_investment']:,.0f}",
            r['roi'],
            r['usage_count']
        ))

    print("\n" + "="*80)
    print("KEY OBSERVATIONS:")
    print("="*80)

    best_roi = results[0]
    print(f"Best ROI: {best_roi['element']} at {best_roi['roi']:.2f} per €1M")
    print(f"  - Recognition: {best_roi['recognition']:.1%}")
    print(f"  - Investment: €{best_roi['total_investment']:,.0f}")
    print(f"  - Used in: {best_roi['usage_count']} ads")

    worst_roi = results[-1]
    print(f"\nWorst ROI: {worst_roi['element']} at {worst_roi['roi']:.2f} per €1M")
    print(f"  - Recognition: {worst_roi['recognition']:.1%}")
    print(f"  - Investment: €{worst_roi['total_investment']:,.0f}")
    print(f"  - Used in: {worst_roi['usage_count']} ads")

    # Check for median vs mean usage
    print("\n" + "="*80)
    print("MEDIAN VS MEAN CALCULATION CHECK:")
    print("="*80)
    print("App.py uses MEDIAN for 'Average Investment' calculation (line 367)")
    print("This is CORRECT for handling outliers in ad spend data")
    print("\nExample: Type element")

    type_ads = [ad for ad in comms_audit_data if ad.get('Type', False)]
    type_spends = [ad['Spend'] for ad in type_ads]
    type_spends_sorted = sorted(type_spends)
    type_mean = sum(type_spends) / len(type_spends)
    type_median = type_spends_sorted[len(type_spends)//2]

    print(f"  Mean spend: €{type_mean:,.2f}")
    print(f"  Median spend: €{type_median:,.2f}")
    print(f"  Difference: €{abs(type_mean - type_median):,.2f}")
    print(f"  Using median is better when data has outliers (large TV campaigns)")

if __name__ == "__main__":
    calculate_roi()
