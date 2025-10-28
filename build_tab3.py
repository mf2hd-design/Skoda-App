#!/usr/bin/env python3
"""
Script to build the new Tab 3 structure for app_v2.py
Extracts content from original app.py and reorganizes into 4 sub-tabs
"""

def extract_lines(filename, start, end):
    """Extract lines from a file"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    return lines[start-1:end]

def adjust_indentation(lines, spaces_to_add):
    """Add indentation to lines"""
    result = []
    for line in lines:
        if line.strip():  # If line is not empty
            result.append(' ' * spaces_to_add + line)
        else:
            result.append(line)
    return result

def build_tab3():
    """Build the complete Tab 3 with sub-tabs"""

    original_file = "/Users/ben/Documents/Saffron/Skoda App/app.py"

    tab3_content = []

    # Header
    tab3_content.append("\n# =====================================================================\n")
    tab3_content.append("# TAB 3: STRATEGIC INSIGHTS (REORGANIZED WITH SUB-TABS)\n")
    tab3_content.append("# =====================================================================\n\n")

    tab3_content.append("with tab3:\n")
    tab3_content.append("    st.header(\"📈 Strategic Insights Dashboard\")\n")
    tab3_content.append("    st.caption(\"Advanced analytics organized into focused categories for easy navigation\")\n\n")

    # Quick summary
    tab3_content.append("    # Quick summary\n")
    tab3_content.append("    st.info(\"\"\"\n")
    tab3_content.append("    ### 🎯 Quick Insights Summary\n\n")
    tab3_content.append("    **🏆 ROI Winners:** Sonic (best efficiency) | Symbol (best overall value)\n\n")
    tab3_content.append("    **🔗 Element Combinations:** Symbol-based pairs show highest recognition | Min 3 elements/ad\n\n")
    tab3_content.append("    **💰 Investment:** Portfolio Strategy tab shows where to invest, hold, or cut budget\n")
    tab3_content.append("    \"\"\")\n\n")

    # Sub-tabs
    tab3_content.append("    # Create 4 focused sub-tabs\n")
    tab3_content.append("    subtab1, subtab2, subtab3, subtab4 = st.tabs([\n")
    tab3_content.append("        \"🎯 Portfolio Strategy\",\n")
    tab3_content.append("        \"💰 Efficiency & ROI\",\n")
    tab3_content.append("        \"🔗 Combinations & Synergies\",\n")
    tab3_content.append("        \"🌍 Market & Consumer Insights\"\n")
    tab3_content.append("    ])\n\n")

    # SUB-TAB 1: Portfolio Strategy
    tab3_content.append("    # ========== SUB-TAB 1: PORTFOLIO STRATEGY ==========\n")
    tab3_content.append("    with subtab1:\n")
    tab3_content.append("        st.markdown(\"### 📊 BCG-Style Portfolio Matrices\")\n")
    tab3_content.append("        st.caption(\"Strategic positioning - where to invest, hold, or cut\")\n\n")

    # Add summary cards
    tab3_content.append("        # Summary cards\n")
    tab3_content.append("        col1, col2, col3, col4 = st.columns(4)\n")
    tab3_content.append("        with col1:\n")
    tab3_content.append("            stars_count = len(master_df[(master_df['Recognition'] >= master_df['Recognition'].median()) & (master_df['Total Investment'] >= master_df['Total Investment'].median())])\n")
    tab3_content.append("            st.metric(\"⭐ Stars\", stars_count, help=\"High recognition + High investment\")\n")
    tab3_content.append("        with col2:\n")
    tab3_content.append("            gems_count = len(master_df[(master_df['Recognition'] >= master_df['Recognition'].median()) & (master_df['Total Investment'] < master_df['Total Investment'].median())])\n")
    tab3_content.append("            st.metric(\"💎 Hidden Gems\", gems_count, help=\"High recognition + Low investment - SCALE UP\")\n")
    tab3_content.append("        with col3:\n")
    tab3_content.append("            dogs_count = len(master_df[(master_df['Recognition'] < master_df['Recognition'].median()) & (master_df['Total Investment'] >= master_df['Total Investment'].median())])\n")
    tab3_content.append("            st.metric(\"🔴 Dogs\", dogs_count, help=\"Low recognition + High investment - CUT\")\n")
    tab3_content.append("        with col4:\n")
    tab3_content.append("            q_count = len(master_df[(master_df['Recognition'] < master_df['Recognition'].median()) & (master_df['Total Investment'] < master_df['Total Investment'].median())])\n")
    tab3_content.append("            st.metric(\"❓ Question Marks\", q_count, help=\"Test or hold\")\n\n")

    tab3_content.append("        st.markdown(\"---\")\n\n")

    # Add filter component
    tab3_content.append("        # Demographic filter\n")
    tab3_content.append("        matrix_filters = render_demographic_filters(\"matrix\")\n")
    tab3_content.append("        matrix_df = apply_demographic_filters(master_df, matrix_filters, brand_elements)\n\n")

    # Extract Portfolio content (lines 1938-2164 from original, but need to add indentation)
    portfolio_lines = extract_lines(original_file, 1938, 2164)
    # These lines are currently indented with 4 spaces (tab3 level), need 8 spaces (subtab level)
    portfolio_lines = adjust_indentation(portfolio_lines, 4)  # Add 4 more spaces
    tab3_content.extend(portfolio_lines)

    # SUB-TAB 2: Efficiency & ROI
    tab3_content.append("\n    # ========== SUB-TAB 2: EFFICIENCY & ROI ==========\n")
    tab3_content.append("    with subtab2:\n")
    tab3_content.append("        st.markdown(\"### 💡 Multi-Dimensional ROI Analysis\")\n")
    tab3_content.append("        st.caption(\"Compare efficiency across total investment, per-ad, and brand equity metrics\")\n\n")

    # Add summary cards
    tab3_content.append("        # Summary cards\n")
    tab3_content.append("        col1, col2 = st.columns(2)\n")
    tab3_content.append("        with col1:\n")
    tab3_content.append("            best_roi = master_df.loc[master_df['Recognition ROI'].idxmax()]\n")
    tab3_content.append("            st.metric(\"🏆 Best ROI\", best_roi['Element'], f\"{best_roi['Recognition ROI']:.2f} per €1M\")\n")
    tab3_content.append("        with col2:\n")
    tab3_content.append("            worst_roi = master_df.loc[master_df['Recognition ROI'].idxmin()]\n")
    tab3_content.append("            st.metric(\"⚠️ Needs Attention\", worst_roi['Element'], f\"{worst_roi['Recognition ROI']:.2f} per €1M\")\n\n")

    tab3_content.append("        st.markdown(\"---\")\n\n")

    # Extract ROI content (lines 2168-2413)
    roi_lines = extract_lines(original_file, 2168, 2413)
    roi_lines = adjust_indentation(roi_lines, 4)
    tab3_content.extend(roi_lines)

    # SUB-TAB 3: Combinations & Synergies
    tab3_content.append("\n    # ========== SUB-TAB 3: COMBINATIONS & SYNERGIES ==========\n")
    tab3_content.append("    with subtab3:\n")
    tab3_content.append("        st.markdown(\"### 🔗 Element Combinations Analysis\")\n")
    tab3_content.append("        st.caption(\"Recognition and attribution when elements appear together\")\n\n")

    tab3_content.append("        # Key insight\n")
    tab3_content.append("        st.info(\"\"\"**Key Finding:** Symbol-based combinations consistently deliver highest recognition. \n")
    tab3_content.append("        Minimum 3 elements per ad recommended for effective brand recognition.\"\"\")\n\n")

    tab3_content.append("        st.markdown(\"---\")\n\n")

    # Add filter
    tab3_content.append("        # Demographic filter\n")
    tab3_content.append("        combo_filters = render_demographic_filters(\"combo\")\n\n")

    # Extract Combinations content (lines 2505-2786, skipping old filter code 2466-2502)
    combos_lines = extract_lines(original_file, 2505, 2786)
    combos_lines = adjust_indentation(combos_lines, 4)
    tab3_content.extend(combos_lines)

    # SUB-TAB 4: Market & Consumer Insights
    tab3_content.append("\n    # ========== SUB-TAB 4: MARKET & CONSUMER INSIGHTS ==========\n")
    tab3_content.append("    with subtab4:\n")
    tab3_content.append("        st.markdown(\"### 🌍 Market Analysis & Consumer Language\")\n")
    tab3_content.append("        st.caption(\"Market consistency, Q03 consumer associations, and Q05 brand confusion\")\n\n")

    tab3_content.append("        st.markdown(\"---\")\n\n")

    # Extract Market Consistency (lines 2414-2461)
    market_lines = extract_lines(original_file, 2414, 2461)
    market_lines = adjust_indentation(market_lines, 4)
    tab3_content.extend(market_lines)

    # Extract Consumer Language Q03 (lines 2789-3061)
    q03_lines = extract_lines(original_file, 2789, 3061)
    q03_lines = adjust_indentation(q03_lines, 4)
    tab3_content.extend(q03_lines)

    # Extract Brand Confusion Q05 from Tab 2 (lines 1433-1836)
    tab3_content.append("\n        st.markdown(\"---\")\n\n")
    tab3_content.append("        # Brand Confusion Analysis (Q05) - Moved from Sentiment tab for better logical grouping\n")
    q05_lines = extract_lines(original_file, 1433, 1836)
    q05_lines = adjust_indentation(q05_lines, 4)
    tab3_content.extend(q05_lines)

    return ''.join(tab3_content)

if __name__ == "__main__":
    tab3_code = build_tab3()
    output_file = "/Users/ben/Documents/Saffron/Skoda App/tab3_generated.py"
    with open(output_file, 'w') as f:
        f.write(tab3_code)
    print(f"✅ Generated Tab 3 code: {output_file}")
    print(f"   Lines: {len(tab3_code.splitlines())}")
