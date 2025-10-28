#!/usr/bin/env python3
"""Script to fix indentation in Tab 3 and reorganize into sub-tabs"""

import re

def fix_tab3_structure(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    new_lines = []
    in_tab3 = False
    in_subtab1 = False
    in_subtab2 = False
    in_subtab3 = False
    in_subtab4 = False
    subtab1_start = None
    subtab2_start_marker = "# Recognition ROI Analysis - Multi-Dimensional"
    subtab3_start_marker = "# Element Combinations Analysis"
    subtab4_start_marker = "# Market Consistency Analysis"
    tab4_marker = "# ==================== TAB 4:"

    base_indent = "    "  # 4 spaces for tab3 content
    subtab_indent = "        "  # 8 spaces for subtab content

    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect Tab 3 start
        if "# ==================== TAB 3:" in line:
            in_tab3 = True
            new_lines.append(line)
            i += 1
            continue

        # Detect Tab 4 start (end of Tab 3)
        if in_tab3 and tab4_marker in line:
            in_tab3 = False
            in_subtab1 = False
            in_subtab2 = False
            in_subtab3 = False
            in_subtab4 = False
            new_lines.append(line)
            i += 1
            continue

        if in_tab3:
            # Detect subtab1 start
            if "with subtab1:" in line:
                in_subtab1 = True
                subtab1_start = len(new_lines)
                new_lines.append(line)
                i += 1
                continue

            # Detect subtab2 start marker
            if subtab2_start_marker in line and in_subtab1:
                # Close subtab1, open subtab2
                in_subtab1 = False
                in_subtab2 = True
                new_lines.append("\n")
                new_lines.append(base_indent + "# ========== SUB-TAB 2: EFFICIENCY & ROI ==========\n")
                new_lines.append(base_indent + "with subtab2:\n")
                # Re-add the marker line with proper indent
                new_lines.append(subtab_indent + line.lstrip())
                i += 1
                continue

            # Detect subtab3 start marker
            if subtab3_start_marker in line and in_subtab2:
                in_subtab2 = False
                in_subtab3 = True
                new_lines.append("\n")
                new_lines.append(base_indent + "# ========== SUB-TAB 3: COMBINATIONS & SYNERGIES ==========\n")
                new_lines.append(base_indent + "with subtab3:\n")
                new_lines.append(subtab_indent + line.lstrip())
                i += 1
                continue

            # Detect subtab4 start marker
            if subtab4_start_marker in line and in_subtab3:
                in_subtab3 = False
                in_subtab4 = True
                new_lines.append("\n")
                new_lines.append(base_indent + "# ========== SUB-TAB 4: MARKET & CONSUMER INSIGHTS ==========\n")
                new_lines.append(base_indent + "with subtab4:\n")
                new_lines.append(subtab_indent + line.lstrip())
                i += 1
                continue

            # Process lines within subtabs
            if in_subtab1 or in_subtab2 or in_subtab3 or in_subtab4:
                # Check if line already starts with base_indent
                if line.startswith(base_indent):
                    # Add 4 more spaces for subtab indent
                    new_lines.append(subtab_indent + line[len(base_indent):])
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

        i += 1

    # Write back
    with open(filename, 'w') as f:
        f.writelines(new_lines)

    print(f"✅ Fixed Tab 3 structure in {filename}")
    print(f"   - Added 4 sub-tabs")
    print(f"   - Fixed indentation")
    return True

if __name__ == "__main__":
    fix_tab3_structure("/Users/ben/Documents/Saffron/Skoda App/app.py")
