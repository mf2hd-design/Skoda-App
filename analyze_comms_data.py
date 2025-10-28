#!/usr/bin/env python3
"""
Analyze comms_data.py to calculate total spend and usage
"""

from comms_data import comms_audit_data

def analyze():
    total_ads = len(comms_audit_data)
    print(f"Total ads in comms audit: {total_ads}")

    # Calculate total spend
    total_spend = sum(ad['Spend'] for ad in comms_audit_data)
    print(f"Total spend across all ads: €{total_spend:,.2f}")

    # Calculate spend by element
    elements = ['Electric Green', 'Emerald Green', 'Type', 'Tagline', 'Symbol',
                'Hacek', 'Wordmark', 'Facets', 'Sonic']

    print("\n" + "="*80)
    print("ELEMENT USAGE AND SPEND")
    print("="*80)

    for element in elements:
        ads_with_element = [ad for ad in comms_audit_data if ad.get(element, False)]
        count = len(ads_with_element)
        usage_pct = count / total_ads if total_ads > 0 else 0
        element_spend = sum(ad['Spend'] for ad in ads_with_element)
        avg_spend = element_spend / count if count > 0 else 0
        median_spend = sorted([ad['Spend'] for ad in ads_with_element])[count//2] if count > 0 else 0

        print(f"\n{element}:")
        print(f"  Usage: {count}/{total_ads} ads ({usage_pct:.1%})")
        print(f"  Total Investment: €{element_spend:,.2f}")
        print(f"  Average Investment: €{avg_spend:,.2f}")
        print(f"  Median Investment: €{median_spend:,.2f}")

    # Medium breakdown
    print("\n" + "="*80)
    print("MEDIUM BREAKDOWN")
    print("="*80)

    video_ads = [ad for ad in comms_audit_data if ad['Medium'] == 'Video']
    image_ads = [ad for ad in comms_audit_data if ad['Medium'] == 'Image']

    print(f"Video ads: {len(video_ads)} ({len(video_ads)/total_ads:.1%})")
    print(f"Image ads: {len(image_ads)} ({len(image_ads)/total_ads:.1%})")

    # Market breakdown
    print("\n" + "="*80)
    print("MARKET BREAKDOWN")
    print("="*80)

    markets = {}
    for ad in comms_audit_data:
        market = ad.get('Market', 'Unknown')
        if market not in markets:
            markets[market] = 0
        markets[market] += 1

    for market, count in sorted(markets.items()):
        print(f"{market}: {count} ads ({count/total_ads:.1%})")

if __name__ == "__main__":
    analyze()
