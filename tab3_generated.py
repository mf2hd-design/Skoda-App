
# =====================================================================
# TAB 3: STRATEGIC INSIGHTS (REORGANIZED WITH SUB-TABS)
# =====================================================================

with tab3:
    st.header("📈 Strategic Insights Dashboard")
    st.caption("Advanced analytics organized into focused categories for easy navigation")

    # Quick summary
    st.info("""
    ### 🎯 Quick Insights Summary

    **🏆 ROI Winners:** Sonic (best efficiency) | Symbol (best overall value)

    **🔗 Element Combinations:** Symbol-based pairs show highest recognition | Min 3 elements/ad

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
        st.caption("Strategic positioning - where to invest, hold, or cut")

        # Summary cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            stars_count = len(master_df[(master_df['Recognition'] >= master_df['Recognition'].median()) & (master_df['Total Investment'] >= master_df['Total Investment'].median())])
            st.metric("⭐ Stars", stars_count, help="High recognition + High investment")
        with col2:
            gems_count = len(master_df[(master_df['Recognition'] >= master_df['Recognition'].median()) & (master_df['Total Investment'] < master_df['Total Investment'].median())])
            st.metric("💎 Hidden Gems", gems_count, help="High recognition + Low investment - SCALE UP")
        with col3:
            dogs_count = len(master_df[(master_df['Recognition'] < master_df['Recognition'].median()) & (master_df['Total Investment'] >= master_df['Total Investment'].median())])
            st.metric("🔴 Dogs", dogs_count, help="Low recognition + High investment - CUT")
        with col4:
            q_count = len(master_df[(master_df['Recognition'] < master_df['Recognition'].median()) & (master_df['Total Investment'] < master_df['Total Investment'].median())])
            st.metric("❓ Question Marks", q_count, help="Test or hold")

        st.markdown("---")

        # Demographic filter
        matrix_filters = render_demographic_filters("matrix")
        matrix_df = apply_demographic_filters(master_df, matrix_filters, brand_elements)

        median_recognition = matrix_df['Recognition'].median()
        median_investment = matrix_df['Total Investment'].median()
        median_uniqueness = matrix_df['Uniqueness'].median()
        median_usage = matrix_df['Overall Usage'].median()
        median_roi = matrix_df['Recognition ROI'].median()

        # Matrix 1: Recognition vs Investment (BCG Matrix)
        st.markdown("#### 1️⃣ Recognition vs Investment Matrix")
        st.caption("Strategic positioning: Stars, Cash Cows, Question Marks, Dogs")
    
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
                title='Recognition vs Investment (BCG Matrix)',
                color_continuous_scale='RdYlGn',
                size_max=30
            )
        
            # Add quadrant lines
            fig_bcg.add_hline(y=median_recognition, line_dash="dash", line_color="gray", opacity=0.5)
            fig_bcg.add_vline(x=median_investment, line_dash="dash", line_color="gray", opacity=0.5)
        
            # Add quadrant labels
            fig_bcg.add_annotation(x=matrix_df['Total Investment'].max() * 0.75, y=matrix_df['Recognition'].max() * 0.95,
                                   text="STARS<br>(High Rec, High Inv)", showarrow=False, font=dict(size=10, color="green"))
            fig_bcg.add_annotation(x=matrix_df['Total Investment'].min() * 1.5, y=matrix_df['Recognition'].max() * 0.95,
                                   text="HIDDEN GEMS<br>(High Rec, Low Inv)", showarrow=False, font=dict(size=10, color="darkgreen"))
            fig_bcg.add_annotation(x=matrix_df['Total Investment'].max() * 0.75, y=matrix_df['Recognition'].min() * 1.5,
                                   text="DOGS<br>(Low Rec, High Inv)<br>⚠️ CUT", showarrow=False, font=dict(size=10, color="red"))
            fig_bcg.add_annotation(x=matrix_df['Total Investment'].min() * 1.5, y=matrix_df['Recognition'].min() * 1.5,
                                   text="QUESTION MARKS<br>(Low Rec, Low Inv)", showarrow=False, font=dict(size=10, color="orange"))
        
            fig_bcg.update_traces(textposition='top center')
            fig_bcg.update_layout(height=500, xaxis_title="Total Investment (€)", yaxis_title="Recognition %")
            fig_bcg.update_yaxes(tickformat='.0%')
            st.plotly_chart(fig_bcg, use_container_width=True)
    
        with col2:
            st.markdown("#### Quadrant Analysis")
        
            # Categorize elements
            stars = matrix_df[(matrix_df['Recognition'] >= median_recognition) & (matrix_df['Total Investment'] >= median_investment)]
            gems = matrix_df[(matrix_df['Recognition'] >= median_recognition) & (matrix_df['Total Investment'] < median_investment)]
            dogs = matrix_df[(matrix_df['Recognition'] < median_recognition) & (matrix_df['Total Investment'] >= median_investment)]
            questions = matrix_df[(matrix_df['Recognition'] < median_recognition) & (matrix_df['Total Investment'] < median_investment)]
        
            if len(stars) > 0:
                st.success(f"**STARS ({len(stars)}):**")
                for _, row in stars.iterrows():
                    st.write(f"• {row['Element']}")
                st.caption("Maintain investment")
        
            if len(gems) > 0:
                st.success(f"**HIDDEN GEMS ({len(gems)}):**")
                for _, row in gems.iterrows():
                    st.write(f"• {row['Element']}")
                st.caption("⬆️ Increase investment")
        
            if len(dogs) > 0:
                st.error(f"**DOGS ({len(dogs)}):**")
                for _, row in dogs.iterrows():
                    st.write(f"• {row['Element']}")
                st.caption("⚠️ Cut or redesign")
        
            if len(questions) > 0:
                st.warning(f"**QUESTION MARKS ({len(questions)}):**")
                for _, row in questions.iterrows():
                    st.write(f"• {row['Element']}")
                st.caption("Test or hold")

        st.markdown("---")

        # Matrix 2: Recognition vs Uniqueness (Brand Equity Matrix)
        st.markdown("#### 2️⃣ Recognition vs Uniqueness Matrix")
        st.caption("Brand equity positioning: Icons, Famous Generics, Hidden Gems, Weak")
    
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
                title='Recognition vs Uniqueness (Brand Equity Matrix)',
                color_continuous_scale='RdYlGn',
                size_max=30
            )
        
            # Add quadrant lines
            fig_equity.add_hline(y=median_recognition, line_dash="dash", line_color="gray", opacity=0.5)
            fig_equity.add_vline(x=median_uniqueness, line_dash="dash", line_color="gray", opacity=0.5)
        
            # Add quadrant labels
            fig_equity.add_annotation(x=matrix_df['Uniqueness'].max() * 0.9, y=matrix_df['Recognition'].max() * 0.95,
                                      text="BRAND ICONS<br>(High Rec, High Uniq)<br>🏆 PROTECT", showarrow=False, font=dict(size=10, color="darkgreen"))
            fig_equity.add_annotation(x=matrix_df['Uniqueness'].min() * 1.2, y=matrix_df['Recognition'].max() * 0.95,
                                      text="FAMOUS GENERICS<br>(High Rec, Low Uniq)<br>⚠️ Risk", showarrow=False, font=dict(size=10, color="orange"))
            fig_equity.add_annotation(x=matrix_df['Uniqueness'].max() * 0.9, y=matrix_df['Recognition'].min() * 1.5,
                                      text="HIDDEN GEMS<br>(Low Rec, High Uniq)<br>💎 Invest", showarrow=False, font=dict(size=10, color="blue"))
            fig_equity.add_annotation(x=matrix_df['Uniqueness'].min() * 1.2, y=matrix_df['Recognition'].min() * 1.5,
                                      text="WEAK<br>(Low Rec, Low Uniq)<br>🔴 Fix", showarrow=False, font=dict(size=10, color="red"))
        
            fig_equity.update_traces(textposition='top center')
            fig_equity.update_layout(height=500, xaxis_title="Uniqueness %", yaxis_title="Recognition %")
            fig_equity.update_xaxes(tickformat='.0%')
            fig_equity.update_yaxes(tickformat='.0%')
            st.plotly_chart(fig_equity, use_container_width=True)
    
        with col2:
            st.markdown("#### Strategic Actions")
        
            # Categorize elements
            icons = matrix_df[(matrix_df['Recognition'] >= median_recognition) & (matrix_df['Uniqueness'] >= median_uniqueness)]
            generics = matrix_df[(matrix_df['Recognition'] >= median_recognition) & (matrix_df['Uniqueness'] < median_uniqueness)]
            hidden = matrix_df[(matrix_df['Recognition'] < median_recognition) & (matrix_df['Uniqueness'] >= median_uniqueness)]
            weak = matrix_df[(matrix_df['Recognition'] < median_recognition) & (matrix_df['Uniqueness'] < median_uniqueness)]
        
            if len(icons) > 0:
                st.success(f"**BRAND ICONS ({len(icons)}):**")
                for _, row in icons.iterrows():
                    st.write(f"• {row['Element']}")
                st.caption("🏆 Core assets - protect")
        
            if len(generics) > 0:
                st.warning(f"**FAMOUS GENERICS ({len(generics)}):**")
                for _, row in generics.iterrows():
                    st.write(f"• {row['Element']}")
                st.caption("⚠️ Increase distinctiveness")
        
            if len(hidden) > 0:
                st.info(f"**HIDDEN GEMS ({len(hidden)}):**")
                for _, row in hidden.iterrows():
                    st.write(f"• {row['Element']}")
                st.caption("💎 Build awareness")
        
            if len(weak) > 0:
                st.error(f"**WEAK ({len(weak)}):**")
                for _, row in weak.iterrows():
                    st.write(f"• {row['Element']}")
                st.caption("🔴 Redesign urgently")

        st.markdown("---")

        # Matrix 3: Usage vs ROI (Efficiency Matrix)
        st.markdown("#### 3️⃣ Usage vs ROI Matrix")
        st.caption("Investment efficiency: Workhorses, Overused, Untapped Potential, Underperformers")
    
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
                title='Usage vs ROI (Efficiency Matrix)',
                color_continuous_scale='RdYlGn',
                size_max=30
            )
        
            # Add quadrant lines
            fig_efficiency.add_hline(y=median_roi, line_dash="dash", line_color="gray", opacity=0.5)
            fig_efficiency.add_vline(x=median_usage, line_dash="dash", line_color="gray", opacity=0.5)
        
            # Add quadrant labels
            fig_efficiency.add_annotation(x=matrix_df['Overall Usage'].max() * 0.85, y=matrix_df['Recognition ROI'].max() * 0.95,
                                          text="WORKHORSES<br>(High Use, High ROI)<br>✅ Perfect", showarrow=False, font=dict(size=10, color="darkgreen"))
            fig_efficiency.add_annotation(x=matrix_df['Overall Usage'].min() * 1.5, y=matrix_df['Recognition ROI'].max() * 0.95,
                                          text="UNTAPPED<br>(Low Use, High ROI)<br>⬆️ Use More", showarrow=False, font=dict(size=10, color="blue"))
            fig_efficiency.add_annotation(x=matrix_df['Overall Usage'].max() * 0.85, y=matrix_df['Recognition ROI'].min() * 1.5,
                                          text="OVERUSED<br>(High Use, Low ROI)<br>⬇️ Cut Back", showarrow=False, font=dict(size=10, color="red"))
            fig_efficiency.add_annotation(x=matrix_df['Overall Usage'].min() * 1.5, y=matrix_df['Recognition ROI'].min() * 1.5,
                                          text="UNDERPERFORMERS<br>(Low Use, Low ROI)", showarrow=False, font=dict(size=10, color="orange"))
        
            fig_efficiency.update_traces(textposition='top center')
            fig_efficiency.update_layout(height=500, xaxis_title="Usage %", yaxis_title="Recognition ROI")
            fig_efficiency.update_xaxes(tickformat='.0%')
            st.plotly_chart(fig_efficiency, use_container_width=True)
    
        with col2:
            st.markdown("#### Optimization Actions")
        
            # Categorize elements
            workhorses = matrix_df[(matrix_df['Overall Usage'] >= median_usage) & (matrix_df['Recognition ROI'] >= median_roi)]
            untapped = matrix_df[(matrix_df['Overall Usage'] < median_usage) & (matrix_df['Recognition ROI'] >= median_roi)]
            overused = matrix_df[(matrix_df['Overall Usage'] >= median_usage) & (matrix_df['Recognition ROI'] < median_roi)]
            underperf = matrix_df[(matrix_df['Overall Usage'] < median_usage) & (matrix_df['Recognition ROI'] < median_roi)]
        
            if len(workhorses) > 0:
                st.success(f"**WORKHORSES ({len(workhorses)}):**")
                for _, row in workhorses.iterrows():
                    st.write(f"• {row['Element']}")
                st.caption("✅ Maintain strategy")
        
            if len(untapped) > 0:
                st.info(f"**UNTAPPED ({len(untapped)}):**")
                for _, row in untapped.iterrows():
                    st.write(f"• {row['Element']}")
                st.caption("⬆️ Increase usage")
        
            if len(overused) > 0:
                st.error(f"**OVERUSED ({len(overused)}):**")
                for _, row in overused.iterrows():
                    st.write(f"• {row['Element']}")
                st.caption("⬇️ Reduce investment")
        
            if len(underperf) > 0:
                st.warning(f"**UNDERPERFORMERS ({len(underperf)}):**")
                for _, row in underperf.iterrows():
                    st.write(f"• {row['Element']}")
                st.caption("Monitor or retire")


    # ========== SUB-TAB 2: EFFICIENCY & ROI ==========
    with subtab2:
        st.markdown("### 💡 Multi-Dimensional ROI Analysis")
        st.caption("Compare efficiency across total investment, per-ad, and brand equity metrics")

        # Summary cards
        col1, col2 = st.columns(2)
        with col1:
            best_roi = master_df.loc[master_df['Recognition ROI'].idxmax()]
            st.metric("🏆 Best ROI", best_roi['Element'], f"{best_roi['Recognition ROI']:.2f} per €1M")
        with col2:
            worst_roi = master_df.loc[master_df['Recognition ROI'].idxmin()]
            st.metric("⚠️ Needs Attention", worst_roi['Element'], f"{worst_roi['Recognition ROI']:.2f} per €1M")

        st.markdown("---")


        # ROI metric selector
        roi_metric = st.selectbox(
            "Choose your efficiency perspective:",
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
            insight_text = "**Shows which assets achieved recognition with minimal total campaign spend.** High scorers are 'hidden gems' that punched above their weight."

        elif roi_metric == "Per-Ad Recognition Efficiency":
            master_df_roi['Selected ROI'] = master_df_roi.apply(
                lambda x: (x['Recognition'] / x['Num Ads'] * 100) if x['Num Ads'] > 0 else 0, axis=1
            )
            metric_label = "Recognition % per Ad"
            insight_text = "**Shows how many ad exposures are needed to build recognition.** High scorers build awareness faster with fewer placements."

        elif roi_metric == "Average Investment Efficiency":
            master_df_roi['Selected ROI'] = master_df_roi.apply(
                lambda x: (x['Recognition'] / x['Average Investment'] * 1_000_000) if x['Average Investment'] > 0 else 0, axis=1
            )
            metric_label = "Recognition % per €1M Average Investment"
            insight_text = "**Shows cost-effectiveness per individual ad placement.** High scorers deliver better recognition per placement budget."

        else:  # Brand Equity Efficiency Index
            master_df_roi['Selected ROI'] = master_df_roi.apply(
                lambda x: (x['Recognition'] * x['Uniqueness']) / (x['Total Investment'] / 1_000_000) if x['Total Investment'] > 0 else 0, axis=1
            )
            metric_label = "Brand Equity Index (Recognition × Uniqueness) per €1M"
            insight_text = "**Holistic efficiency combining fame and differentiation.** High scorers deliver the most long-term brand equity per euro - ideal for identifying non-negotiables."

        st.info(insight_text)

        col1, col2 = st.columns([2, 1])

        with col1:
            roi_df = master_df_roi.sort_values('Selected ROI', ascending=True)
            fig_roi = px.bar(
                roi_df,
                y='Element',
                x='Selected ROI',
                orientation='h',
                title=f'Efficiency Analysis: {metric_label}',
                text=roi_df['Selected ROI'].apply(lambda x: f'{x:.2f}'),
                color='Selected ROI',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_roi, use_container_width=True)

        with col2:
            st.markdown("#### Top 3 Performers:")
            top_3_roi = roi_df.nlargest(3, 'Selected ROI')
            for idx, row in top_3_roi.iterrows():
                st.success(f"**{row['Element']}**: {row['Selected ROI']:.2f}")
                with st.expander(f"Why {row['Element']}?"):
                    if roi_metric == "Brand Equity Efficiency Index":
                        equity = row['Recognition'] * row['Uniqueness']
                        st.write(f"**Recognition:** {row['Recognition']:.0%}")
                        st.write(f"**Uniqueness:** {row['Uniqueness']:.0%}")
                        st.write(f"**Brand Equity:** {equity:.3f}")
                        st.write(f"**Investment:** €{row['Total Investment']:,.0f}")
                        st.write(f"**Why efficient:** Achieves {equity:.3f} brand equity with only €{row['Total Investment']:,.0f} - delivers maximum long-term value per euro")
                    else:
                        st.write(f"**Recognition:** {row['Recognition']:.0%}")
                        st.write(f"**Investment:** €{row['Total Investment']:,.0f}")
                        st.write(f"**Usage:** {row['Overall Usage']:.0%}")
                        st.write(f"**Why efficient:** High recognition relative to investment/usage indicates strong inherent memorability and strategic placement")

            st.markdown("#### Bottom 3:")
            bottom_3_roi = roi_df.nsmallest(3, 'Selected ROI')
            for idx, row in bottom_3_roi.iterrows():
                st.warning(f"**{row['Element']}**: {row['Selected ROI']:.2f}")
                with st.expander(f"Why {row['Element']}?"):
                    if roi_metric == "Brand Equity Efficiency Index":
                        equity = row['Recognition'] * row['Uniqueness']
                        st.write(f"**Recognition:** {row['Recognition']:.0%}")
                        st.write(f"**Uniqueness:** {row['Uniqueness']:.0%}")
                        st.write(f"**Brand Equity:** {equity:.3f}")
                        st.write(f"**Investment:** €{row['Total Investment']:,.0f}")
                        if row['Selected ROI'] < 0.01:
                            st.write(f"**Why low/zero:** Very high investment (€{row['Total Investment']:,.0f}) relative to brand equity outcome ({equity:.3f}). This could indicate: 1) Recent investment not yet reflected in recognition, 2) Generic element that lacks Škoda distinctiveness, or 3) Inefficient deployment")
                        else:
                            st.write(f"**Why lower:** Investment (€{row['Total Investment']:,.0f}) is high relative to the brand equity delivered ({equity:.3f})")
                    else:
                        st.write(f"**Recognition:** {row['Recognition']:.0%}")
                        st.write(f"**Investment:** €{row['Total Investment']:,.0f}")
                        st.write(f"**Usage:** {row['Overall Usage']:.0%}")
                        st.write(f"**Why lower:** High investment/usage but recognition hasn't grown proportionally - may need creative optimization or more time to build awareness")

            st.markdown("#### Strategic Implication:")
            if roi_metric == "Total Investment Efficiency":
                st.write("• Scale up top performers")
                st.write("• Review bottom performers' deployment")
            elif roi_metric == "Per-Ad Recognition Efficiency":
                st.write("• Top assets build awareness faster")
                st.write("• Increase frequency for top performers")
            elif roi_metric == "Average Investment Efficiency":
                st.write("• Optimize placement budgets")
                st.write("• Reallocate to efficient assets")
            else:
                st.write("• **Top = Non-negotiables candidates**")
                st.write("• **Bottom = Requires optimization**")

        st.markdown("---")

        # Efficiency Quadrant Analysis
        st.markdown("### 📊 Asset Performance Quadrants")
        st.info("**Insight:** Categorize assets by recognition and uniqueness performance")

        with st.expander("📖 How to read the quadrants"):
            st.markdown("""
            This analysis categorizes brand assets into 4 strategic groups based on their performance:

            **⭐ Stars (Top-Right):** High Recognition + High Uniqueness
            - **What this means:** Above-median consumer recognition AND brand attribution
            - **Strategy:** Protect and amplify - these are your brand-building powerhouses

            **🐴 Workhorses (Top-Left):** High Recognition + Lower Uniqueness
            - **What this means:** Above-median recognition but below-median brand attribution
            - **Strategy:** Maintain awareness but pair with unique assets for differentiation

            **💎 Hidden Gems (Bottom-Right):** Lower Recognition + High Uniqueness
            - **What this means:** Strong brand attribution but below-median recognition
            - **Strategy:** Invest more - these have untapped potential for differentiation

            **❓ Question Marks (Bottom-Left):** Lower Recognition + Lower Uniqueness
            - **What this means:** Below-median on both recognition and brand attribution
            - **Strategy:** Evaluate - optimize deployment or reconsider as core asset
            """)

        # Calculate quadrants
        median_recognition = master_df['Recognition'].median()
        median_uniqueness = master_df['Uniqueness'].median()

        def get_quadrant(row):
            if row['Recognition'] >= median_recognition and row['Uniqueness'] >= median_uniqueness:
                return 'Stars ⭐'
            elif row['Recognition'] >= median_recognition and row['Uniqueness'] < median_uniqueness:
                return 'Workhorses 🐴'
            elif row['Recognition'] < median_recognition and row['Uniqueness'] >= median_uniqueness:
                return 'Hidden Gems 💎'
            else:
                return 'Question Marks ❓'

        master_df['Quadrant'] = master_df.apply(get_quadrant, axis=1)

        fig_quadrant = px.scatter(
            master_df,
            x='Uniqueness',
            y='Recognition',
            color='Quadrant',
            text='Element',
            size='Total Investment',
            size_max=50,
            title='Asset Performance Quadrants',
            color_discrete_map={
                'Stars ⭐': '#4CAF50',
                'Workhorses 🐴': '#2196F3',
                'Hidden Gems 💎': '#FF9800',
                'Question Marks ❓': '#F44336'
            }
        )

        # Add median lines
        fig_quadrant.add_hline(y=median_recognition, line_dash="dash", line_color="gray", annotation_text="Median Recognition")
        fig_quadrant.add_vline(x=median_uniqueness, line_dash="dash", line_color="gray", annotation_text="Median Uniqueness")
        fig_quadrant.update_traces(textposition='top center')
        fig_quadrant.update_layout(height=600)

        st.plotly_chart(fig_quadrant, use_container_width=True)

        # Quadrant breakdown
        col1, col2, col3, col4 = st.columns(4)

        quadrant_counts = master_df['Quadrant'].value_counts()

        with col1:
            stars = master_df[master_df['Quadrant'] == 'Stars ⭐']
            st.success(f"**Stars ⭐** ({len(stars)})")
            st.write("High Recognition + High Uniqueness")
            for idx, row in stars.iterrows():
                st.write(f"• **{row['Element']}**")
                with st.expander(f"Why {row['Element']} is a Star"):
                    st.write(f"**Recognition:** {row['Recognition']:.0%} (above median {median_recognition:.0%})")
                    st.write(f"**Uniqueness:** {row['Uniqueness']:.0%} (above median {median_uniqueness:.0%})")
                    st.write(f"**Usage:** {row['Overall Usage']:.0%} of campaigns")
                    st.write(f"**Investment:** €{row['Total Investment']:,.0f}")
                    st.write(f"**Why a star:** High exposure ({row['Overall Usage']:.0%} usage) + distinctive Škoda identity ({row['Uniqueness']:.0%} uniqueness) = maximum brand equity builder")

        with col2:
            workhorses = master_df[master_df['Quadrant'] == 'Workhorses 🐴']
            st.info(f"**Workhorses 🐴** ({len(workhorses)})")
            st.write("High Recognition + Lower Uniqueness")
            for idx, row in workhorses.iterrows():
                st.write(f"• **{row['Element']}**")
                with st.expander(f"Why {row['Element']} is a Workhorse"):
                    st.write(f"**Recognition:** {row['Recognition']:.0%} (above median {median_recognition:.0%})")
                    st.write(f"**Uniqueness:** {row['Uniqueness']:.0%} (below median {median_uniqueness:.0%})")
                    st.write(f"**Usage:** {row['Overall Usage']:.0%} of campaigns")
                    st.write(f"**Why a workhorse:** High familiarity but lower distinctiveness suggests this may be a more generic element. Useful for awareness but pair with unique assets for differentiation")

        with col3:
            gems = master_df[master_df['Quadrant'] == 'Hidden Gems 💎']
            st.warning(f"**Hidden Gems 💎** ({len(gems)})")
            st.write("Lower Recognition + High Uniqueness")
            for idx, row in gems.iterrows():
                st.write(f"• **{row['Element']}**")
                with st.expander(f"Why {row['Element']} is a Hidden Gem"):
                    st.write(f"**Recognition:** {row['Recognition']:.0%} (below median {median_recognition:.0%})")
                    st.write(f"**Uniqueness:** {row['Uniqueness']:.0%} (above median {median_uniqueness:.0%})")
                    st.write(f"**Usage:** {row['Overall Usage']:.0%} of campaigns")
                    st.write(f"**Investment:** €{row['Total Investment']:,.0f}")
                    st.write(f"**Why a hidden gem:** Highly distinctive ({row['Uniqueness']:.0%} uniqueness) but underexposed ({row['Overall Usage']:.0%} usage). **BIG OPPORTUNITY** - increase deployment to build recognition while maintaining differentiation")

        with col4:
            questions = master_df[master_df['Quadrant'] == 'Question Marks ❓']
            st.error(f"**Question Marks ❓** ({len(questions)})")
            st.write("Lower Recognition + Lower Uniqueness")
            for idx, row in questions.iterrows():
                st.write(f"• **{row['Element']}**")
                with st.expander(f"Why {row['Element']} is a Question Mark"):
                    st.write(f"**Recognition:** {row['Recognition']:.0%} (below median {median_recognition:.0%})")
                    st.write(f"**Uniqueness:** {row['Uniqueness']:.0%} (below median {median_uniqueness:.0%})")
                    st.write(f"**Usage:** {row['Overall Usage']:.0%} of campaigns")
                    st.write(f"**Investment:** €{row['Total Investment']:,.0f}")
                    st.write(f"**Why a question mark:** Lower recognition AND lower distinctiveness. Could be due to: 1) Limited usage ({row['Overall Usage']:.0%}), 2) Recent introduction, 3) Generic design, or 4) Ineffective deployment. Requires strategic review")

        st.markdown("---")

        # Market Consistency Analysis

    # ========== SUB-TAB 3: COMBINATIONS & SYNERGIES ==========
    with subtab3:
        st.markdown("### 🔗 Element Combinations Analysis")
        st.caption("Recognition and attribution when elements appear together")

        # Key insight
        st.info("""**Key Finding:** Symbol-based combinations consistently deliver highest recognition. 
        Minimum 3 elements per ad recommended for effective brand recognition.""")

        st.markdown("---")

        # Demographic filter
        combo_filters = render_demographic_filters("combo")

        st.markdown("#### Recognition When Elements Appear Together")
        st.info("Shows the average recognition level when element pairs appear together in ads. Green indicates high recognition, red indicates low recognition.")

        # Create recognition matrix for co-occurring elements
        recognition_matrix = pd.DataFrame(0.0, index=brand_elements, columns=brand_elements, dtype=float)
    
        for element1 in brand_elements:
            for element2 in brand_elements:
                if element1 != element2:
                    # Find ads where both elements appear
                    both_present = audit_df[audit_df[element1] & audit_df[element2]]
                
                    if len(both_present) > 0:
                        # Calculate average recognition across all countries when both appear
                        rec1 = recognition_by_country[element1]
                        rec2 = recognition_by_country[element2]
                    
                        # Average recognition of both elements
                        avg_recognition = (sum(rec1.values()) + sum(rec2.values())) / (2 * len(rec1))
                        recognition_matrix.loc[element1, element2] = avg_recognition

        # Display as heatmap with red-yellow-green scale
        fig_recognition = px.imshow(
            recognition_matrix,
            labels=dict(x="Combined with", y="Element", color="Recognition Level"),
            x=recognition_matrix.columns,
            y=recognition_matrix.index,
            color_continuous_scale='RdYlGn',  # Red to Yellow to Green
            text_auto='.0%',
            aspect="auto",
            title="Recognition Heatmap: Element Combinations"
        )
        fig_recognition.update_layout(height=600)
        st.plotly_chart(fig_recognition, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🏆 Highest Recognition Combinations")
        
            # Find top combinations by recognition
            combinations = []
            for element1 in brand_elements:
                for element2 in brand_elements:
                    if element1 < element2:  # Avoid duplicates
                        combined_recognition = recognition_matrix.loc[element1, element2]
                        if combined_recognition > 0:
                            # Count how often they appear together
                            both_present = audit_df[audit_df[element1] & audit_df[element2]].shape[0]
                        
                            combinations.append({
                                'Pair': f"{element1} + {element2}",
                                'Recognition': combined_recognition,
                                'Appearances': both_present
                            })
        
            combinations_df = pd.DataFrame(combinations).sort_values('Recognition', ascending=False).head(5)
        
            for _, row in combinations_df.iterrows():
                st.success(f"**{row['Pair']}**")
                st.write(f"   Recognition: {row['Recognition']:.0%} | Appears together: {row['Appearances']} ads")

        with col2:
            st.markdown("#### 💡 Strategic Recommendations")
        
            # Find Symbol's best recognition partners
            symbol_recognition = recognition_matrix.loc['Symbol'].sort_values(ascending=False)
            top_symbol_partner = symbol_recognition.index[0]
        
            st.markdown(f"""
            **Key Findings:**
        
            1. **Symbol combinations perform best:** Highest recognition when Symbol pairs with {top_symbol_partner} ({symbol_recognition.iloc[0]:.0%})
        
            2. **Minimum combinations:** Use at least 3 elements together (recognition builds from 10% with 1 element to 40% with 6)
        
            3. **Top performing pairings:**
               - Look for green cells in the heatmap
               - Symbol-based combinations consistently score higher
               - Avoid red combinations (low recognition)
        
            4. **Avoid:** Single element use (only 10% recognition)
            """)

        st.markdown("---")

        # Recognition lift analysis
        st.markdown("#### 📈 Recognition Lift: Multi-Element Effect")
    
        col1, col2 = st.columns([2, 1])
    
        with col1:
            # Calculate how many elements typically appear together
            audit_df['num_elements'] = audit_df[brand_elements].sum(axis=1)
            elements_per_ad = audit_df['num_elements'].value_counts().sort_index()
        
            fig_elements = go.Figure(go.Bar(
                x=elements_per_ad.index,
                y=elements_per_ad.values,
                marker_color='#4CAF50',
                text=elements_per_ad.values,
                textposition='outside'
            ))
            fig_elements.update_layout(
                title='Distribution: Number of Elements per Ad',
                xaxis_title='Number of Brand Elements',
                yaxis_title='Number of Ads',
                height=400
            )
            st.plotly_chart(fig_elements, use_container_width=True)
    
        with col2:
            st.markdown("#### Key Stats")
        
            avg_elements = audit_df['num_elements'].mean()
            st.metric("Avg Elements/Ad", f"{avg_elements:.1f}")
        
            median_elements = audit_df['num_elements'].median()
            st.metric("Median Elements/Ad", f"{int(median_elements)}")
        
            max_elements = audit_df['num_elements'].max()
            st.metric("Max Elements/Ad", f"{int(max_elements)}")
        
            st.info("""
            **Insight:** Based on recognition journey data, ads with 3+ elements are more likely to drive brand recognition.
            """)

        st.markdown("---")

        # NEW: Cross-Asset Synergies - Attribution (Uniqueness) Analysis
        st.markdown("### 🎯 Cross-Asset Synergies: Brand Attribution Analysis")
        st.caption("Which element pairs or combinations drive correct Škoda attribution (uniqueness)?")

        st.info("""
        **CRITICAL INSIGHT:** Recognition isn't enough - we need **correct attribution**. This analysis shows which
        element combinations make consumers correctly identify the brand as Škoda (not competitors or generic).

        **Why this matters:**
        - "Colour + Facets" may achieve 40% recognition, but only 25% uniqueness (confused with competitors)
        - "Symbol + Wordmark" may achieve 45% recognition AND 60% uniqueness (clearly Škoda)

        **Gold for creative guidelines:** Shows which pairs maximize brand-building vs just awareness-building.
        """)

        # Create uniqueness matrix for co-occurring elements
        st.markdown("#### 🏆 Brand Attribution (Uniqueness) When Elements Appear Together")

        # Check if uniqueness_by_country data exists
        if uniqueness_by_country:
            uniqueness_matrix = pd.DataFrame(0.0, index=brand_elements, columns=brand_elements, dtype=float)

            for element1 in brand_elements:
                for element2 in brand_elements:
                    if element1 != element2:
                        # Find ads where both elements appear
                        both_present = audit_df[audit_df[element1] & audit_df[element2]]

                        if len(both_present) > 0 and element1 in uniqueness_by_country and element2 in uniqueness_by_country:
                            # Calculate average uniqueness across all countries when both appear
                            uniq1 = uniqueness_by_country[element1]
                            uniq2 = uniqueness_by_country[element2]

                            # Average uniqueness of both elements
                            avg_uniqueness = (sum(uniq1.values()) + sum(uniq2.values())) / (2 * len(uniq1))
                            uniqueness_matrix.loc[element1, element2] = avg_uniqueness

            # Display as heatmap
            fig_uniqueness = px.imshow(
                uniqueness_matrix,
                labels=dict(x="Combined with", y="Element", color="Brand Attribution (Uniqueness)"),
                x=uniqueness_matrix.columns,
                y=uniqueness_matrix.index,
                color_continuous_scale='RdYlGn',
                text_auto='.0%',
                aspect="auto",
                title="Brand Attribution Heatmap: Element Combinations"
            )
            fig_uniqueness.update_layout(height=600)
            st.plotly_chart(fig_uniqueness, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### ✅ Highest Attribution Pairs (Best for Brand-Building)")

                # Find top combinations by uniqueness
                attr_combinations = []
                for element1 in brand_elements:
                    for element2 in brand_elements:
                        if element1 < element2:  # Avoid duplicates
                            combined_uniqueness = uniqueness_matrix.loc[element1, element2]
                            combined_recognition = recognition_matrix.loc[element1, element2]
                            if combined_uniqueness > 0:
                                # Count how often they appear together
                                both_present = audit_df[audit_df[element1] & audit_df[element2]].shape[0]

                                # Calculate brand equity score (recognition × uniqueness)
                                brand_equity = combined_recognition * combined_uniqueness

                                attr_combinations.append({
                                    'Pair': f"{element1} + {element2}",
                                    'Uniqueness': combined_uniqueness,
                                    'Recognition': combined_recognition,
                                    'Brand Equity': brand_equity,
                                    'Appearances': both_present
                                })

                attr_combinations_df = pd.DataFrame(attr_combinations).sort_values('Uniqueness', ascending=False).head(5)

                for _, row in attr_combinations_df.iterrows():
                    st.success(f"**{row['Pair']}**")
                    st.write(f"   • Attribution: {row['Uniqueness']:.0%}")
                    st.write(f"   • Recognition: {row['Recognition']:.0%}")
                    st.write(f"   • Brand Equity: {row['Brand Equity']:.3f}")
                    st.caption(f"   Appears together: {row['Appearances']} ads")

            with col2:
                st.markdown("#### ⚠️ High Recognition but Low Attribution (Risk)")

                # Find combinations with high recognition but lower uniqueness (awareness without attribution)
                risky_combinations = []
                for _, row in pd.DataFrame(attr_combinations).iterrows():
                    # High recognition (>35%) but low uniqueness (<30%)
                    if row['Recognition'] > 0.35 and row['Uniqueness'] < 0.30:
                        risky_combinations.append(row)

                if risky_combinations:
                    risky_df = pd.DataFrame(risky_combinations).sort_values('Recognition', ascending=False).head(3)
                    for _, row in risky_df.iterrows():
                        st.warning(f"**{row['Pair']}**")
                        st.write(f"   • Recognition: {row['Recognition']:.0%} ✅")
                        st.write(f"   • Attribution: {row['Uniqueness']:.0%} ⚠️")
                        st.caption("High awareness but confused with competitors")
                else:
                    st.info("No high-risk combinations identified - all pairs with high recognition also have reasonable attribution")

            st.markdown("---")

            # Top strategic insight
            st.markdown("#### 💡 Strategic Playbook: Pair Selection Guidelines")

            # Get top attribution pair
            top_attr_pair = attr_combinations_df.iloc[0]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**🥇 Gold Standard Pairs**")
                st.write("High Attribution + High Recognition")
                for _, row in attr_combinations_df.head(3).iterrows():
                    if row['Uniqueness'] > 0.35 and row['Recognition'] > 0.35:
                        st.write(f"• {row['Pair']}")
                st.caption("Use for brand-building campaigns")

            with col2:
                st.markdown("**🥈 Awareness Builders**")
                st.write("High Recognition, Lower Attribution")
                for _, row in pd.DataFrame(attr_combinations).sort_values('Recognition', ascending=False).head(5).iterrows():
                    if row['Recognition'] > 0.40 and row['Uniqueness'] < 0.35:
                        st.write(f"• {row['Pair']}")
                        break
                st.caption("Pair with Symbol/Wordmark to boost attribution")

            with col3:
                st.markdown("**💎 Hidden Gems**")
                st.write("High Attribution, Lower Recognition")
                for _, row in attr_combinations_df.iterrows():
                    if row['Uniqueness'] > 0.35 and row['Recognition'] < 0.35:
                        st.write(f"• {row['Pair']}")
                st.caption("Increase usage to build awareness")

            st.success(f"""
            **Key Finding:** The top attribution pair is **{top_attr_pair['Pair']}** with {top_attr_pair['Uniqueness']:.0%} uniqueness and {top_attr_pair['Recognition']:.0%} recognition.

            **Creative Guideline:** Prioritize this combination in all brand communications to maximize both awareness AND correct Škoda attribution.
            Avoid pairs with high recognition but low uniqueness - they build awareness for competitors, not Škoda.
            """)

        else:
            st.warning("Uniqueness by country data not available. Cannot calculate brand attribution for element combinations.")

        st.markdown("---")

    # ========== SUB-TAB 4: MARKET & CONSUMER INSIGHTS ==========
    with subtab4:
        st.markdown("### 🌍 Market Analysis & Consumer Language")
        st.caption("Market consistency, Q03 consumer associations, and Q05 brand confusion")

        st.markdown("---")

        st.markdown("### 🌍 Market Consistency Analysis")
        st.info("**Insight:** Are assets used consistently across markets?")

        markets = sorted(audit_df['Market'].unique())
        selected_markets = st.multiselect("Select markets to compare:", markets, default=markets)

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
                text=market_comparison['Usage'].apply(lambda x: f'{x:.0%}')
            )
            st.plotly_chart(fig_market, use_container_width=True)

            # Consistency score
            consistency_scores = market_comparison.groupby('Element')['Usage'].std()
            most_consistent = consistency_scores.idxmin()
            least_consistent = consistency_scores.idxmax()

            col1, col2 = st.columns(2)
            with col1:
                if pd.notna(most_consistent):
                    st.success(f"**Most Consistent:** {most_consistent} (σ={consistency_scores[most_consistent]:.3f})")
                else:
                    st.info("Consistency data not available")
            with col2:
                if pd.notna(least_consistent):
                    st.warning(f"**Least Consistent:** {least_consistent} (σ={consistency_scores[least_consistent]:.3f})")
                else:
                    st.info("Consistency data not available")

        st.markdown("---")

        # ========== SUB-TAB 3: COMBINATIONS & SYNERGIES ==========
        st.markdown("### 💬 Consumer Language Analysis (Q03)")
        st.caption("What words do consumers use to describe brand elements? Sentiment classification and theme analysis.")

        st.info("""
        **Methodology:** Text analysis of open-ended responses using NLP sentiment classification and theme clustering.
        Shows what consumers actually say (not just predefined scales).
        """)

        # Element selector
        selected_element = st.selectbox(
            "Select element to analyze:",
            list(q03_associations_data.keys()),
            key="q03_element_selector"
        )

        # Add demographic selector for consumer language
        st.markdown("#### 🎯 Filter by Demographics")
        language_demo_col1, language_demo_col2, language_demo_col3 = st.columns(3)

        with language_demo_col1:
            language_country = st.selectbox(
                "Country:",
                ["All Countries", "UK", "Spain", "Germany", "Poland"],
                key="language_country"
            )

        with language_demo_col2:
            language_age = st.selectbox(
                "Age Group:",
                ["All Ages", "18-30", "31-42", "43-55"],
                key="language_age"
            )

        with language_demo_col3:
            language_gender = st.selectbox(
                "Gender:",
                ["All Genders", "Male", "Female"],
                key="language_gender"
            )

        # Show demographic context
        language_demo_text = []
        if language_country != "All Countries":
            language_demo_text.append(f"**{language_country}**")
        if language_age != "All Ages":
            language_demo_text.append(f"**{language_age}**")
        if language_gender != "All Genders":
            language_demo_text.append(f"**{language_gender}**")

        if language_demo_text:
            st.caption(f"Showing data for: {' | '.join(language_demo_text)}")
        else:
            st.caption("Showing data for: **All Demographics**")

        element_data = q03_associations_data[selected_element]

        col1, col2 = st.columns([2, 1])

        with col1:
            # Top words bar chart
            st.markdown(f"#### Top 10 Words for {selected_element}")
        
            words_df = pd.DataFrame({
                'Word': element_data['top_words'],
                'Frequency': element_data['frequencies']
            })
        
            fig_words = px.bar(
                words_df,
                x='Frequency',
                y='Word',
                orientation='h',
                title=f'Most Common Words: {selected_element}',
                text=words_df['Frequency'].apply(lambda x: f'{x:.0%}'),
                color='Frequency',
                color_continuous_scale='Blues'
            )
            fig_words.update_layout(height=400, showlegend=False)
            fig_words.update_traces(textposition='outside')
            st.plotly_chart(fig_words, use_container_width=True)

        with col2:
            # Sentiment analysis from Q04 adjective scales
            st.markdown("#### Sentiment (Q04 Adjectives)")

            # Get sentiment data from research_data (Q04), not q03_associations_data
            sentiment_data_source = research_data[selected_element]

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

        # Word cloud alternative - show all associations
        st.markdown(f"#### All Associations for {selected_element}")
        st.caption("Full list of consumer descriptions (Q03 open-text responses)")

        all_words_df = pd.DataFrame({
            'Association': element_data['top_words'],
            'Frequency': element_data['frequencies']
        })

        st.dataframe(all_words_df.style.format({'Frequency': '{:.1%}'}),
                    use_container_width=True, hide_index=True)

        # Comparative sentiment analysis across all elements (Q04)
        st.markdown("---")
        st.markdown("### 📊 Sentiment Comparison Across All Elements")
        st.caption("Based on Q04 adjective scales: Bold, Stylish, Modern vs Cautious, Plain, Old-Fashioned")

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
            marker_color='#4CAF50'
        ))

        fig_sent_comp.add_trace(go.Bar(
            name='Negative',
            y=sent_comparison_df['Element'],
            x=sent_comparison_df['Negative'],
            orientation='h',
            marker_color='#F44336'
        ))

        fig_sent_comp.update_layout(
            barmode='overlay',
            title='Adjective Sentiment Analysis: All Elements',
            xaxis_title='Percentage',
            yaxis_title='',
            height=500,
            xaxis_tickformat='.0%'
        )

        st.plotly_chart(fig_sent_comp, use_container_width=True)

        # Key insights
        col1, col2 = st.columns(2)

        with col1:
            most_positive = sent_comparison_df.iloc[-1]
            st.success(f"""
            **Most Positive Sentiment:**
            - **{most_positive['Element']}**: {most_positive['Net']:+.1%} net sentiment
            - {most_positive['Positive']:.0%} positive adjectives
            """)

        with col2:
            most_negative = sent_comparison_df.iloc[0]
            st.warning(f"""
            **Most Negative Sentiment:**
            - **{most_negative['Element']}**: {most_negative['Net']:+.1%} net sentiment
            - {most_negative['Negative']:.0%} negative adjectives
            """)

        st.markdown("---")

        # Search for Strategic Terms in Consumer Language
        st.markdown("### 🔍 Strategic Brand Terms Search")
        st.caption("Search Q03 responses to see if desired brand values appear in consumer language")

        st.info("""
        **Purpose:** The client asked: *"Can we see whether 'Exploration' naturally clusters with our key assets or if people describe us with unrelated adjectives?"*

        Use this search to find if strategic brand values (Exploration, Innovation, Modern, etc.) appear in actual consumer responses.
        """)

        # Search input
        search_term = st.text_input("Search for a word or phrase in consumer associations:",
                                    value="explore",
                                    placeholder="e.g., explore, innovation, modern, safe, boring")

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
                st.caption("This suggests the term is not prominent in consumer language about Škoda brand elements.")

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
            text=overall_df['Total Frequency'].apply(lambda x: f'{x:.1%}')
        )
        fig_overall.update_layout(height=500, showlegend=False)
        fig_overall.update_traces(textposition='outside')
        st.plotly_chart(fig_overall, use_container_width=True)

        st.success("""
        **Strategic Insight:**

        The most common words reveal what consumers **actually say** about Škoda elements, vs what the brand **wants** them to say.

        - ✅ **If "Škoda" appears frequently**: Strong brand recognition
        - ⚠️ **If "Confusing" or "Boring" appear**: Perception issues need addressing
        - 💡 **If "Exploration" is absent**: Gap between brand aspiration and consumer reality
        """)

    # ==================== TAB 4: NON-NEGOTIABLES ====================

        st.markdown("---")

        # Brand Confusion Analysis (Q05) - Moved from Sentiment tab for better logical grouping
        st.markdown("### 🎯 Brand Confusion Matrix (Q05)")
        st.caption("Which brands do consumers think these elements belong to?")

        st.info("""
        **Key Insight:** Brand confusion analysis reveals competitive threats. High Škoda attribution = distinctive asset.
        High competitor attribution = confusion risk. High "Generic" = lacks brand identity.
        """)

        # Add demographic selector for confusion matrix
        st.markdown("#### 🎯 Filter by Demographics")
        confusion_demo_col1, confusion_demo_col2, confusion_demo_col3 = st.columns(3)

        with confusion_demo_col1:
            confusion_country = st.selectbox(
                "Country:",
                ["All Countries", "UK", "Spain", "Germany", "Poland"],
                key="confusion_country"
            )

        with confusion_demo_col2:
            confusion_age = st.selectbox(
                "Age Group:",
                ["All Ages", "18-30", "31-42", "43-55"],
                key="confusion_age"
            )

        with confusion_demo_col3:
            confusion_gender = st.selectbox(
                "Gender:",
                ["All Genders", "Male", "Female"],
                key="confusion_gender"
            )

        # Show demographic context
        demo_text = []
        if confusion_country != "All Countries":
            demo_text.append(f"**{confusion_country}**")
        if confusion_age != "All Ages":
            demo_text.append(f"**{confusion_age}**")
        if confusion_gender != "All Genders":
            demo_text.append(f"**{confusion_gender}**")

        if demo_text:
            st.caption(f"Showing data for: {' | '.join(demo_text)}")
        else:
            st.caption("Showing data for: **All Demographics**")

        # Create confusion matrix using REAL Q05 data
        confusion_df = pd.DataFrame(q05_confusion_data).T
        confusion_df = confusion_df[['Skoda', 'Other_mentions', 'Dont_know']]
        confusion_df.columns = ['Škoda', 'Other Brands', "Don't Know"]

        # Create confusion matrix with inverted scale for competitors
        # We need to invert competitor columns so high values = red, low values = green
        confusion_df_display = confusion_df.copy()

        # Invert competitor and generic columns (1 - value) so high becomes low for coloring
        for col in ['Other Brands', "Don't Know"]:
            confusion_df_display[col] = 1 - confusion_df_display[col]

        # Keep Škoda as-is (high = green is correct)
    
        # Create heatmap with consistent color scale
        fig_confusion = px.imshow(
            confusion_df_display,
            labels=dict(x="Attributed Brand", y="Element", color="Score"),
            x=confusion_df_display.columns,
            y=confusion_df_display.index,
            color_continuous_scale='RdYlGn',  # Red = bad, Green = good
            text_auto=False,  # We'll add custom text
            aspect="auto",
            title="Brand Attribution: Who Do Consumers Think Owns These Elements?"
        )
    
        # Add text annotations with actual percentages (not inverted display values)
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
        st.plotly_chart(fig_confusion, use_container_width=True)
    
        st.caption("""
        **Color Guide:** 
        - 🟢 Green = Good (High Škoda attribution OR Low competitor/generic confusion)
        - 🔴 Red = Bad (Low Škoda attribution OR High competitor/generic confusion)
        """)

        # Analysis columns
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ✅ Distinctive Assets")
            distinctive = confusion_df.sort_values('Škoda', ascending=False).head(3)
            for element, row in distinctive.iterrows():
                st.success(f"**{element}**: {row['Škoda']:.0%} Škoda")
                dont_know_pct = row["Don't Know"]
                st.caption(f"Other brands: {row['Other Brands']:.0%} | Don't know: {dont_know_pct:.0%}")

        with col2:
            st.markdown("#### ⚠️ Confusion Risks")

            # Find elements with high other brand confusion
            high_other = confusion_df[confusion_df['Other Brands'] >= 0.20].sort_values('Other Brands', ascending=False)
            if len(high_other) > 0:
                st.warning("**Other Brand Confusion (Brand Dilution Risk):**")
                for element, row in high_other.iterrows():
                    st.write(f"• **{element}**: {row['Other Brands']:.0%} think it's another brand")

            # Find elements with high don't know
            high_dontknow = confusion_df[confusion_df["Don't Know"] >= 0.55].sort_values("Don't Know", ascending=False)
            if len(high_dontknow) > 0:
                st.warning("**High 'Don't Know' (Low Distinctiveness):**")
                for element, row in high_dontknow.iterrows():
                    dont_know_val = row["Don't Know"]
                    st.write(f"• **{element}**: {dont_know_val:.0%} don't recognize")

        # Confusion Matrix
        st.markdown("#### 📊 Confusion Matrix")

        confusion_df['Brand_Confusion_Risk'] = confusion_df['Other Brands']
        confusion_df['Distinctiveness_Score'] = confusion_df['Škoda'] - confusion_df['Brand_Confusion_Risk']

        threat_matrix = []
        for element in confusion_df.index:
            skoda_attr = confusion_df.loc[element, 'Škoda']
            other_brands = confusion_df.loc[element, 'Other Brands']
            dont_know = confusion_df.loc[element, "Don't Know"]

            threat_matrix.append({
                'Element': element,
                'Škoda Attribution': skoda_attr,
                'Other Brands': other_brands,
                "Don't Know": dont_know
            })

        threat_df = pd.DataFrame(threat_matrix).sort_values('Other Brands', ascending=False)
        st.dataframe(threat_df.style.format({
            'Škoda Attribution': '{:.0%}',
            'Other Brands': '{:.0%}',
            "Don't Know": '{:.0%}'
        }), use_container_width=True)

        # Detailed Competitor Breakdown
        st.markdown("---")
        st.markdown("### 🔍 Detailed Competitor Breakdown")
        st.caption("Top brands mentioned when consumers misattribute Škoda elements (from open-text responses)")

        # Load detailed competitor data (CLEANED VERSION - verbatim responses recoded)
        try:
            with open('q05_competitor_detail_CLEANED.json', 'r', encoding='utf-8') as f:
                competitor_detail = json.load(f)

            # Global Summary: Top Automotive Competitors Across All Elements
            st.markdown("#### 🌍 Global View: Top Automotive Competitor Mentions")
            st.caption("Aggregated across all brand elements - which car brands are most confused with Škoda?")

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
                        title="Global Automotive Competitor Confusion",
                        labels={'Mentions': 'Total Mentions', 'Brand': 'Competitor Brand'},
                        color='Mentions',
                        color_continuous_scale='Reds'
                    )
                    fig_global_comp.update_traces(texttemplate='%{y}', textposition='outside')
                    fig_global_comp.update_layout(height=350, showlegend=False)
                    st.plotly_chart(fig_global_comp, use_container_width=True)

                with col2:
                    st.markdown("**Key Findings:**")
                    st.metric("Total Automotive Mentions", total_mentions)
                    st.caption(f"Out of ~726 total verbatim responses")

                    top_competitor = max(global_auto_competitors.items(), key=lambda x: x[1])
                    st.metric("Top Competitor", top_competitor[0])
                    st.caption(f"{top_competitor[1]} mentions")

                    confusion_rate = (total_mentions / 726) * 100 if total_mentions > 0 else 0
                    st.success(f"""
                    **Overall Automotive Confusion: {confusion_rate:.1f}%**

                    Minimal competitive threat - Škoda elements are NOT being confused with competitor car brands.
                    """)
            else:
                st.success("✅ **Excellent News:** Zero automotive competitor mentions across all elements. No competitive brand confusion detected.")

            st.markdown("---")

            # Market-Level Breakdown using q05_confusion_by_country
            if q05_confusion_by_country:
                st.markdown("#### 🗺️ Market-Level Competitor Confusion")
                st.caption("Automotive confusion breakdown by market (UK, Spain, Germany, Poland)")

                # Calculate automotive confusion by market for each element
                markets = ["UK", "Spain", "Germany", "Poland"]
                market_confusion_data = []

                for element in brand_elements:
                    if element in q05_confusion_by_country:
                        for market in markets:
                            if market in q05_confusion_by_country[element]:
                                # "Other_mentions" includes all non-Skoda mentions (automotive + non-automotive + confused)
                                other_pct = q05_confusion_by_country[element][market].get('Other', 0) or \
                                           q05_confusion_by_country[element][market].get('Other_mentions', 0)

                                market_confusion_data.append({
                                    'Element': element,
                                    'Market': market,
                                    'Other_Mentions': other_pct
                                })

                if market_confusion_data:
                    market_df = pd.DataFrame(market_confusion_data)

                    # Create heatmap
                    market_pivot = market_df.pivot(index='Element', columns='Market', values='Other_Mentions')

                    fig_market_heat = px.imshow(
                        market_pivot,
                        labels=dict(x="Market", y="Brand Element", color="Other Brand Mentions %"),
                        text_auto='.0%',
                        aspect="auto",
                        color_continuous_scale='Reds',
                        title="Non-Škoda Brand Mentions by Market (includes all misattributions)"
                    )
                    fig_market_heat.update_layout(height=450)
                    st.plotly_chart(fig_market_heat, use_container_width=True)

                    # Market insights
                    col1, col2, col3, col4 = st.columns(4)

                    for idx, market in enumerate(markets):
                        with [col1, col2, col3, col4][idx]:
                            market_data = market_df[market_df['Market'] == market]
                            avg_confusion = market_data['Other_Mentions'].mean()

                            st.metric(
                                market,
                                f"{avg_confusion:.0%}",
                                help=f"Average 'Other brand' mentions across all elements in {market}"
                            )

                            # Find most confused element in this market
                            most_confused = market_data.loc[market_data['Other_Mentions'].idxmax()]
                            st.caption(f"Highest: {most_confused['Element']} ({most_confused['Other_Mentions']:.0%})")

                    st.info("""
                    **Note:** "Other Mentions" includes automotive competitors, non-automotive brands, and generic/confused responses.
                    Based on global data, only ~0.55% of these are actual automotive competitors - the vast majority are non-threats.
                    """)

            st.markdown("---")

            # Element selector for detailed view
            detail_element = st.selectbox(
                "Select element to view detailed competitor mentions:",
                options=[e for e in research_data.keys() if e in competitor_detail],
                key="detail_element"
            )

            if detail_element in competitor_detail:
                detail_data = competitor_detail[detail_element]

                if detail_data['total_responses'] > 0:
                    # Show summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Verbatim Responses Coded", detail_data['total_responses'])
                    with col2:
                        st.metric("Škoda Attribution", f"{detail_data['skoda_percentage']:.1%}")
                    with col3:
                        st.metric("Don't Know", f"{detail_data['dont_know_percentage']:.1%}")
                    with col4:
                        auto_count = detail_data['automotive_competitors']['count']
                        st.metric("🚗 Automotive Competitors", auto_count,
                                 help="Actual car brands mentioned - the real competitive threat")

                    # Show data quality note
                    if '_data_quality_note' in detail_data:
                        st.caption(f"ℹ️ {detail_data['_data_quality_note']}")

                    # Key Insight Box
                    st.markdown("#### 💡 Key Finding")
                    auto_pct = detail_data['automotive_competitors']['percentage']

                    if auto_pct < 0.05:  # Less than 5%
                        st.success(f"""
                        **Minimal Automotive Confusion ({auto_pct:.1%})**
                        This element shows very low confusion with competitor car brands. When misattributed,
                        it's primarily to non-automotive brands or generic responses - not competitive threats.
                        """)
                    elif auto_pct < 0.15:  # 5-15%
                        st.warning(f"""
                        **Moderate Automotive Confusion ({auto_pct:.1%})**
                        Some confusion with competitor brands exists, but most misattribution is to non-automotive brands.
                        """)
                    else:
                        st.error(f"""
                        **High Automotive Confusion ({auto_pct:.1%})**
                        Significant confusion with competitor car brands - this element may be too generic or similar to competitors.
                        """)

                    # Breakdown by category
                    st.markdown("#### 📊 Misattribution Breakdown")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        auto_data = detail_data['automotive_competitors']
                        st.metric("🚗 Automotive Competitors",
                                 f"{auto_data['percentage']:.1%}",
                                 help="Car brands - real competitive threat")
                        if auto_data['brands']:
                            st.caption(f"{auto_data['count']} mentions: " +
                                     ", ".join([f"{b['brand']} ({b['count']})" for b in auto_data['brands'][:3]]))
                        else:
                            st.caption("No automotive brands mentioned ✅")

                    with col2:
                        non_auto_data = detail_data['non_automotive_brands']
                        st.metric("🏪 Non-Automotive Brands",
                                 f"{non_auto_data['percentage']:.1%}",
                                 help="Consumer brands outside automotive - not competitive threats")
                        if non_auto_data.get('top_mentions'):
                            top_brands = ", ".join([b['brand'] for b in non_auto_data['top_mentions'][:3]])
                            st.caption(f"Top: {top_brands}")

                    with col3:
                        confused_data = detail_data['could_not_identify']
                        st.metric("❓ Generic/Confused",
                                 f"{confused_data['percentage']:.1%}",
                                 help="Non-brand responses (e.g., 'car', 'green', unclear answers)")
                        st.caption(confused_data['description'])

                    # Automotive competitors detail (if any)
                    if detail_data['automotive_competitors']['brands']:
                        st.markdown("#### 🚗 Automotive Competitor Details")
                        st.error("⚠️ **These represent actual competitive threats in the automotive market:**")

                        auto_df = pd.DataFrame(detail_data['automotive_competitors']['brands'])
                        auto_df['percentage_display'] = auto_df['percentage'].apply(lambda x: f"{x:.2%}")

                        st.dataframe(
                            auto_df[['brand', 'count', 'percentage_display']].rename(columns={
                                'brand': 'Competitor Brand',
                                'count': 'Mentions',
                                'percentage_display': '% of Responses'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )

                    # Non-automotive brands (collapsed)
                    if detail_data['non_automotive_brands']['count'] > 0:
                        with st.expander(f"📋 View non-automotive brands ({detail_data['non_automotive_brands']['count']} mentions)"):
                            st.caption("These are not competitive threats - just brand confusion outside automotive sector")
                            if detail_data['non_automotive_brands'].get('top_mentions'):
                                for brand_data in detail_data['non_automotive_brands']['top_mentions']:
                                    st.write(f"- **{brand_data['brand']}**: {brand_data['count']} mentions")

                else:
                    st.info("No verbatim responses coded for this element.")

        except FileNotFoundError:
            st.warning("⚠️ Detailed competitor data file not found. Please ensure q05_competitor_detail_CLEANED.json is in the app directory.")
        except Exception as e:
            st.error(f"Error loading competitor detail data: {str(e)}")

