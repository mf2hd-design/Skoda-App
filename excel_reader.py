#!/usr/bin/env python3
"""
Excel Data Extraction Script
Reads Excel files and extracts data for verification
"""

import sys
import os

# Try importing pandas
try:
    import pandas as pd
    print("pandas available")
    PANDAS_OK = True
except ImportError:
    print("ERROR: pandas not available")
    PANDAS_OK = False
    sys.exit(1)

def read_excel_file(filepath, sheet_name=None):
    """Read Excel file and return basic information"""
    try:
        # First, just list all sheets
        xl_file = pd.ExcelFile(filepath)
        print(f"\n{'='*80}")
        print(f"FILE: {os.path.basename(filepath)}")
        print(f"{'='*80}")
        print(f"Available sheets: {xl_file.sheet_names}\n")

        return xl_file
    except Exception as e:
        print(f"ERROR reading {filepath}: {e}")
        return None

def search_for_questions(xl_file, search_terms):
    """Search for specific questions/tables in Excel sheets"""
    results = {}

    for sheet_name in xl_file.sheet_names:
        print(f"\nSearching sheet: {sheet_name}")

        try:
            # Read sheet
            df = pd.read_excel(xl_file, sheet_name=sheet_name, header=None)

            # Search for terms in all cells
            for term in search_terms:
                for idx, row in df.iterrows():
                    for col_idx, cell_value in enumerate(row):
                        if pd.notna(cell_value) and isinstance(cell_value, str):
                            if term.upper() in str(cell_value).upper():
                                print(f"  Found '{term}' in row {idx+1}, col {col_idx+1}: {cell_value}")

                                # Try to extract nearby data
                                if sheet_name not in results:
                                    results[sheet_name] = {}
                                if term not in results[sheet_name]:
                                    results[sheet_name][term] = []

                                results[sheet_name][term].append({
                                    'row': idx+1,
                                    'col': col_idx+1,
                                    'value': cell_value
                                })
        except Exception as e:
            print(f"  Error reading sheet {sheet_name}: {e}")

    return results

def main():
    base_path = "/Users/ben/Documents/Saffron/Skoda App"

    # Main research data file
    main_excel = os.path.join(base_path, "P045556_ALL_Tables_20251020_Private.xlsx")

    if os.path.exists(main_excel):
        xl_file = read_excel_file(main_excel)

        if xl_file:
            # Search for key question codes
            search_terms = ['Q02', 'Q05', 'Q04', 'Q27', 'Q28', 'QHiddenAwareness',
                          'recognition', 'uniqueness', 'personality', 'Electric Green',
                          'Symbol', 'Wordmark']

            print("\n" + "="*80)
            print("SEARCHING FOR KEY DATA POINTS")
            print("="*80)

            results = search_for_questions(xl_file, search_terms)

            # Print summary
            print("\n" + "="*80)
            print("SEARCH RESULTS SUMMARY")
            print("="*80)
            for sheet_name, terms in results.items():
                print(f"\nSheet: {sheet_name}")
                for term, occurrences in terms.items():
                    print(f"  {term}: {len(occurrences)} occurrence(s)")
    else:
        print(f"ERROR: File not found: {main_excel}")

    # Also check brand assets file
    brand_excel = os.path.join(base_path, "2025-10-06 P045556 - Saffron Brand Assets - Final - V2 - Private.xlsx")
    if os.path.exists(brand_excel):
        print("\n\n")
        xl_file = read_excel_file(brand_excel)

    # Ad spend file
    ad_excel = os.path.join(base_path, "250915_SKO_Ads Overview.xlsx")
    if os.path.exists(ad_excel):
        print("\n\n")
        xl_file = read_excel_file(ad_excel)

        if xl_file:
            # Read first sheet to see ad spend data
            first_sheet = xl_file.sheet_names[0]
            print(f"\nReading first sheet: {first_sheet}")
            df = pd.read_excel(xl_file, sheet_name=first_sheet)
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"\nFirst few rows:")
            print(df.head(10))

if __name__ == "__main__":
    if not PANDAS_OK:
        sys.exit(1)
    main()
