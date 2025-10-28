# =====================================================================
# TAB 3: STRATEGIC INSIGHTS (REORGANIZED WITH SUB-TABS)
# =====================================================================
# This template shows the new structure for Tab 3
# Content will be extracted from original app.py lines 1837-3062
#
# NEW STRUCTURE:
# - Sub-tab 1: Portfolio Strategy (lines 1863-2164)
# - Sub-tab 2: Efficiency & ROI (lines 2167-2413)
# - Sub-tab 3: Combinations & Synergies (lines 2462-2786)
# - Sub-tab 4: Market & Consumer Insights (lines 2414-2461 + 2787-3061 + Q05 from Tab2: 1433-1836)
#
# Key improvements:
# - Use shared demographic filter component
# - Remove duplicate filter code
# - Add summary cards at top of each sub-tab
# - Consistent color coding
# =====================================================================

with tab3:
    st.header("📈 Strategic Insights Dashboard")
    st.caption("Advanced analytics organized into focused categories for easy navigation")

    # Quick summary at top level
    st.info("""
    ### 🎯 Quick Insights Summary

    **🏆 ROI Winners:** Sonic (best efficiency) | Symbol (best overall value)

    **🔗 Element Combinations:** Symbol-based pairs show highest recognition | Min 3 elements/ad recommended

    **💰 Investment:** Portfolio Strategy tab shows where to invest, hold, or cut budget
    """)

    # Create 4 focused sub-tabs
    subtab1, subtab2, subtab3, subtab4 = st.tabs([
        "🎯 Portfolio Strategy",
        "💰 Efficiency & ROI",
        "🔗 Combinations & Synergies",
        "🌍 Market & Consumer Insights"
    ])

    # ========== SUB-TAB 1: PORTFOLIO STRATEGY ==========
    with subtab1:
        st.markdown("### 📊 BCG-Style Portfolio Matrices")
        st.caption("Strategic positioning analysis - where to invest, hold, or cut")

        # Summary cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            stars_count = len(master_df[(master_df['Recognition'] >= master_df['Recognition'].median()) &
                                        (master_df['Total Investment'] >= master_df['Total Investment'].median())])
            st.metric("⭐ Stars", stars_count, help="High recognition + High investment")
        with col2:
            gems_count = len(master_df[(master_df['Recognition'] >= master_df['Recognition'].median()) &
                                       (master_df['Total Investment'] < master_df['Total Investment'].median())])
            st.metric("💎 Hidden Gems", gems_count, help="High recognition + Low investment")
        with col3:
            dogs_count = len(master_df[(master_df['Recognition'] < master_df['Recognition'].median()) &
                                       (master_df['Total Investment'] >= master_df['Total Investment'].median())])
            st.metric("🔴 Dogs", dogs_count, help="Low recognition + High investment - CUT")
        with col4:
            q_count = len(master_df[(master_df['Recognition'] < master_df['Recognition'].median()) &
                                    (master_df['Total Investment'] < master_df['Total Investment'].median())])
            st.metric("❓ Question Marks", q_count, help="Low recognition + Low investment")

        st.markdown("---")

        # Shared demographic filter
        matrix_filters = render_demographic_filters("matrix")
        matrix_df = apply_demographic_filters(master_df, matrix_filters, brand_elements)

        # [REST OF PORTFOLIO CONTENT FROM LINES 1938-2164]

    # ========== SUB-TAB 2: EFFICIENCY & ROI ==========
    with subtab2:
        st.markdown("### 💡 Multi-Dimensional ROI Analysis")
        st.caption("Compare efficiency across different metrics: total investment, per-ad, and brand equity")

        # Summary cards
        col1, col2 = st.columns(2)
        with col1:
            best_roi = master_df.loc[master_df['Recognition ROI'].idxmax()]
            st.metric("🏆 Best ROI", best_roi['Element'],
                     f"{best_roi['Recognition ROI']:.2f} per €1M")
        with col2:
            worst_roi = master_df.loc[master_df['Recognition ROI'].idxmin()]
            st.metric("⚠️ Needs Attention", worst_roi['Element'],
                     f"{worst_roi['Recognition ROI']:.2f} per €1M")

        st.markdown("---")

        # [REST OF ROI CONTENT FROM LINES 2167-2413]

    # ========== SUB-TAB 3: COMBINATIONS & SYNERGIES ==========
    with subtab3:
        st.markdown("### 🔗 Element Combinations Analysis")
        st.caption("What works together? Recognition and attribution when elements appear in pairs")

        # Summary card
        st.info("""
        **Key Finding:** Symbol-based combinations consistently deliver highest recognition.
        Minimum 3 elements per ad recommended for effective brand recognition.
        """)

        st.markdown("---")

        # Shared demographic filter
        combo_filters = render_demographic_filters("combo")

        # [REST OF COMBINATIONS CONTENT FROM LINES 2462-2786]

    # ========== SUB-TAB 4: MARKET & CONSUMER INSIGHTS ==========
    with subtab4:
        st.markdown("### 🌍 Market Analysis & Consumer Language")
        st.caption("Market consistency, consumer associations, and brand confusion (Q05)")

        # Summary cards
        col1, col2 = st.columns(2)
        with col1:
            # Calculate most consistent element
            markets = sorted(audit_df['Market'].unique())
            if len(markets) > 0:
                market_data = []
                for market in markets:
                    market_df = audit_df[audit_df['Market'] == market]
                    total_ads = len(market_df)
                    for element in brand_elements:
                        usage = market_df[element].sum() / total_ads if total_ads > 0 else 0
                        market_data.append({'Market': market, 'Element': element, 'Usage': usage})
                market_comparison = pd.DataFrame(market_data)
                consistency_scores = market_comparison.groupby('Element')['Usage'].std()
                most_consistent = consistency_scores.idxmin()
                st.metric("🌍 Most Consistent Asset", most_consistent,
                         help="Asset with most consistent usage across markets")

        with col2:
            # Show distinctive assets count
            distinctive_count = len(master_df[master_df['Uniqueness'] >= master_df['Uniqueness'].median()])
            st.metric("🎯 Distinctive Assets", distinctive_count,
                     f"{distinctive_count}/{len(master_df)} above median uniqueness")

        st.markdown("---")

        # [MARKET CONSISTENCY CONTENT FROM LINES 2414-2461]
        # [CONSUMER LANGUAGE Q03 CONTENT FROM LINES 2787-3061]
        # [BRAND CONFUSION Q05 CONTENT FROM TAB 2 LINES 1433-1836]
