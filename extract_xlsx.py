#!/usr/bin/env python3
"""
Extract data from Excel files using zipfile (xlsx is a ZIP archive)
"""

import zipfile
import xml.etree.ElementTree as ET
import os

def extract_xlsx_sheets(xlsx_path):
    """Extract sheet names from xlsx file"""
    print(f"\n{'='*80}")
    print(f"FILE: {os.path.basename(xlsx_path)}")
    print(f"{'='*80}")

    try:
        with zipfile.ZipFile(xlsx_path, 'r') as zip_ref:
            # List all files in the archive
            print("\nFiles in archive:")
            for name in zip_ref.namelist()[:20]:  # First 20 files
                print(f"  {name}")

            # Try to read workbook.xml to get sheet names
            if 'xl/workbook.xml' in zip_ref.namelist():
                print("\nExtracting sheet names from workbook.xml...")
                workbook_xml = zip_ref.read('xl/workbook.xml')
                root = ET.fromstring(workbook_xml)

                # Find sheets
                namespaces = {'': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
                sheets = root.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheet')

                print(f"\nFound {len(sheets)} sheets:")
                for sheet in sheets:
                    name = sheet.get('name')
                    sheet_id = sheet.get('sheetId')
                    r_id = sheet.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
                    print(f"  Sheet {sheet_id}: {name} (r:id={r_id})")

            # Try to read shared strings (for text values)
            if 'xl/sharedStrings.xml' in zip_ref.namelist():
                print("\nReading shared strings...")
                shared_strings_xml = zip_ref.read('xl/sharedStrings.xml')
                root = ET.fromstring(shared_strings_xml)

                # Get all text values
                texts = []
                for si in root.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t'):
                    if si.text:
                        texts.append(si.text)

                print(f"Found {len(texts)} shared string values")

                # Search for key terms
                search_terms = ['Q02', 'Q05', 'Q04', 'Q27', 'Q28', 'QHiddenAwareness',
                              'Electric Green', 'Symbol', 'recognition', 'uniqueness']

                print("\nSearching for key terms in shared strings:")
                for term in search_terms:
                    matches = [t for t in texts if term.lower() in t.lower()]
                    if matches:
                        print(f"\n  '{term}' found in {len(matches)} strings:")
                        for match in matches[:5]:  # Show first 5
                            print(f"    - {match[:100]}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    base_path = "/Users/ben/Documents/Saffron/Skoda App"

    files = [
        "P045556_ALL_Tables_20251020_Private.xlsx",
        "2025-10-06 P045556 - Saffron Brand Assets - Final - V2 - Private.xlsx",
        "250915_SKO_Ads Overview.xlsx"
    ]

    for filename in files:
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            extract_xlsx_sheets(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == "__main__":
    main()
