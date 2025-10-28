# ✅ Testing Checklist - app_v2.py

Use this checklist to verify app_v2.py works correctly before switching permanently.

---

## 🚀 Basic Startup

- [ ] App starts without errors: `streamlit run app_v2.py`
- [ ] Header displays correctly: "Škoda Brand Intelligence Dashboard"
- [ ] "Enhanced UX Edition" subtitle shows
- [ ] All 8 tabs visible in navigation

---

## 📊 Tab 1: Executive Summary

- [ ] Page loads without errors
- [ ] "Most Recognised Asset" metric displays
- [ ] "Most Unique Asset" metric displays
- [ ] Expandable explanations work ("Why is this...")
- [ ] All 4 metrics columns render
- [ ] Complete Tier Overview table appears
- [ ] Brand Equity Matrix chart renders
- [ ] Demographic filters work (country, age, gender)
- [ ] Chart updates when filters change
- [ ] Recognition Trigger Index displays

---

## 💚 Tab 2: Sentiment Analysis

- [ ] Page loads without errors
- [ ] Sentiment comparison charts render
- [ ] Net sentiment ranking displays
- [ ] Top 3 / Bottom 3 sections work
- [ ] Detailed sentiment data table appears
- [ ] Strategic implications section shows
- [ ] **Q05 Brand Confusion section is ABSENT** (moved to Tab 3.4)
- [ ] Demographic filters work correctly

---

## 📈 Tab 3: Strategic Insights ⭐ CRITICAL

### Main Tab Level
- [ ] Page loads without errors
- [ ] Quick Insights Summary box displays
- [ ] 4 sub-tabs visible: Portfolio | ROI | Combinations | Market

### Sub-tab 1: 🎯 Portfolio Strategy
- [ ] Summary cards display (Stars/Gems/Dogs/Question Marks)
- [ ] Demographic filters render
- [ ] Filters apply to data correctly
- [ ] Recognition vs Investment Matrix (BCG) renders
- [ ] Quadrant labels show correctly
- [ ] Recognition vs Uniqueness Matrix renders
- [ ] Usage vs ROI Matrix renders
- [ ] Asset Performance Quadrants chart displays
- [ ] Quadrant breakdown shows correct elements

### Sub-tab 2: 💰 Efficiency & ROI
- [ ] Summary cards display (Best ROI / Needs Attention)
- [ ] ROI metric selector dropdown works
- [ ] Chart updates when metric selection changes
- [ ] Top 3 performers list displays
- [ ] Bottom 3 performers list displays
- [ ] Expandable "Why?" sections work
- [ ] Strategic implications update per metric

### Sub-tab 3: 🔗 Combinations & Synergies
- [ ] Key Finding info box displays
- [ ] Demographic filters render
- [ ] Recognition heatmap loads
- [ ] Heatmap shows correct color scale (red-yellow-green)
- [ ] Element labels readable
- [ ] Highest recognition combinations list shows
- [ ] Strategic recommendations section displays
- [ ] Multi-element effect chart renders
- [ ] Key stats metrics display (avg/median/max elements)
- [ ] Brand attribution heatmap displays
- [ ] Highest attribution pairs list shows
- [ ] Risk section (high recognition, low attribution) works
- [ ] Strategic playbook section displays

### Sub-tab 4: 🌍 Market & Consumer Insights
- [ ] Market consistency analysis section loads
- [ ] Market selector multiselect works
- [ ] Usage by market chart renders
- [ ] Most/least consistent metrics display
- [ ] Consumer Language Analysis (Q03) section loads
- [ ] Element selector dropdown works
- [ ] Demographic filters render
- [ ] Top words chart displays
- [ ] Sentiment breakdown shows
- [ ] Search functionality works (strategic terms)
- [ ] Most common words chart renders
- [ ] **Q05 Brand Confusion section is PRESENT** (moved from Tab 2)
- [ ] Competitor confusion matrix displays
- [ ] Market-level breakdown works
- [ ] Distinctive vs confusion risk lists show

---

## 🎯 Tab 4: Non-Negotiables

- [ ] Page loads without errors
- [ ] Must-Use assets list displays
- [ ] Recommended assets list displays
- [ ] Requires Attention section shows
- [ ] Quick Reference table works
- [ ] All metrics calculate correctly

---

## 🔮 Tab 5: Future-Proofing

- [ ] Page loads without errors
- [ ] High Potential Assets section displays
- [ ] Investment Optimization charts render
- [ ] Consistency Improvement table shows
- [ ] Action Plan sections display (short/long term)

---

## 🔍 Tab 6: Deep Dive Analysis

- [ ] Page loads without errors
- [ ] Filters section works
- [ ] Element selector updates charts
- [ ] Investment by Element chart renders
- [ ] Usage Frequency chart renders
- [ ] Brand Personality Analysis section displays
- [ ] Recognition by Market section works
- [ ] Demographic filters apply correctly
- [ ] Brand Attribution by Market shows
- [ ] Market Consistency Score displays
- [ ] Consistency score chart renders

---

## 📄 Tab 7: Data Explorer

- [ ] Page loads without errors
- [ ] Comms Audit Data table displays
- [ ] Research Data table displays
- [ ] Combined Metrics table displays
- [ ] Survey Demographics section shows
- [ ] Sample by Country table renders
- [ ] Demographics breakdown displays
- [ ] All 2,011 respondents accounted for

---

## 🧭 Tab 8: Recognition Journey

- [ ] Page loads without errors
- [ ] Recognition Build chart renders
- [ ] Key Insights section displays
- [ ] Strategic Implications section shows
- [ ] First Recognition Trigger Index displays
- [ ] Demographic filters work
- [ ] Age Migration Analysis (if demographic data exists)
- [ ] Post-Reveal familiarity chart renders
- [ ] Emotional Response chart displays
- [ ] Strategic Integration section shows

---

## 🔧 Functionality Tests

### Demographic Filters
- [ ] Country filter works in all locations used
- [ ] Age filter works in all locations used
- [ ] Gender filter works in all locations used
- [ ] Filter context text updates correctly
- [ ] Charts/data update when filters change
- [ ] No duplicate filter code visible

### Shared Components
- [ ] `render_demographic_filters()` works consistently
- [ ] `apply_demographic_filters()` updates data correctly
- [ ] No console errors when using filters

### Data Integrity
- [ ] All recognition percentages match original app
- [ ] All uniqueness percentages match original app
- [ ] All ROI calculations match original app
- [ ] Investment totals match original app
- [ ] Element counts match original app

### Charts & Visualizations
- [ ] All Plotly charts render
- [ ] Hover tooltips work
- [ ] Color scales display correctly (RdYlGn)
- [ ] Axis labels readable
- [ ] Legends display when needed

### Performance
- [ ] App loads in reasonable time (<10 seconds)
- [ ] Tab switching is responsive
- [ ] Filter changes update quickly (<2 seconds)
- [ ] No memory issues with prolonged use
- [ ] Charts render smoothly

---

## 🐛 Common Issues to Check

### If charts don't render:
- [ ] Check browser console for errors
- [ ] Verify Plotly is installed: `pip list | grep plotly`
- [ ] Try refreshing the page

### If data seems wrong:
- [ ] Compare specific values with app.py
- [ ] Check that all JSON files are present
- [ ] Verify comms_data.py hasn't changed

### If filters don't work:
- [ ] Check unique key names (no duplicates)
- [ ] Verify demographic data files loaded
- [ ] Check console for Python errors

### If sub-tabs missing:
- [ ] Look for syntax errors in Tab 3 section
- [ ] Verify `with subtab1:` through `with subtab4:` present
- [ ] Check indentation is correct

---

## ✅ Final Verification

### Compare with Original
- [ ] Open original app: `streamlit run app.py`
- [ ] Navigate to Tab 3
- [ ] Scroll through entire tab
- [ ] Open app_v2.py: `streamlit run app_v2.py`
- [ ] Navigate to Tab 3
- [ ] Experience sub-tab navigation
- [ ] Confirm all content present, just reorganized

### Performance Comparison
- [ ] Original Tab 3 scroll length: ~1,227 lines worth
- [ ] New Tab 3.1 scroll length: ~250 lines
- [ ] New Tab 3.2 scroll length: ~260 lines
- [ ] New Tab 3.3 scroll length: ~300 lines
- [ ] New Tab 3.4 scroll length: ~400 lines
- [ ] Confirm: Easier to navigate with sub-tabs

### User Experience Test
- [ ] Ask: "Where do I find ROI analysis?"
  - Original: Scroll through Tab 3
  - New: Click Tab 3 → Sub-tab 2
- [ ] Ask: "What element pairs work best together?"
  - Original: Scroll through Tab 3
  - New: Click Tab 3 → Sub-tab 3
- [ ] Ask: "Show me brand confusion data?"
  - Original: Tab 2 (unexpected location)
  - New: Tab 3 → Sub-tab 4 (logical location)

---

## 📋 Sign-off

**Tested by:** _______________
**Date:** _______________
**Version:** app_v2.py

**Issues found:** ___ (0 = ready for production)

**Recommendation:**
- [ ] ✅ Deploy app_v2.py as primary version
- [ ] ⚠️ Minor issues to fix first (list below)
- [ ] ❌ Keep using app.py for now

**Notes:**
```
[Space for testing notes]
```

---

## 🎉 Success Criteria

**All boxes checked above = Ready to deploy!**

Minimum requirements:
✅ All 8 tabs load without errors
✅ All 4 sub-tabs in Tab 3 work correctly
✅ Demographic filters function properly
✅ All charts render
✅ Data matches original app
✅ Navigation is faster/easier than original

---

*Testing checklist for app_v2.py - Enhanced UX Edition*
*Last updated: 2025-10-28*
