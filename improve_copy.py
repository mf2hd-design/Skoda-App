#!/usr/bin/env python3
"""
Improve copy throughout app.py based on UX audit
Priority 1: Remove jargon, add context, shorten text
"""

import re

# Read the file
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Dictionary of replacements (jargon removal and clarity improvements)
replacements = {
    # Research jargon → Plain language
    'respondents who': 'people who',
    'Respondents who': 'People who',
    'consumers who': 'people who',
    'Consumers who': 'People who',
    'semantic differential scales': 'rating questions',
    'MaxDiff ranking': 'ranking survey',
    'Top 2 Box': 'Most Positive Ratings',
    'Bottom 2 Box': 'Most Negative Ratings',
    'T2B': 'Most Positive',
    'B2B': 'Most Negative',

    # Verbose → Concise (in captions and info boxes)
    'Data-driven categorization of brand elements by performance metrics': 'Which elements perform best and why',
    'Advanced portfolio analytics organized into focused categories': 'Deep dive into element performance and strategy',
    'Identification of underutilized assets and investment efficiency patterns': 'Find hidden opportunities and improve spend efficiency',
    'Detailed breakdowns and custom filtering': 'Explore data your way with filters',
    'How consumers discover and recognize Škoda through brand elements': 'How people discover and connect with Škoda',

    # Add context (why it matters)
    'Based on Q04 adjective scales': 'Shows how people feel about each element',
    'Based on Q03 open-text responses': 'What people say in their own words',

    # Passive → Active voice
    'which do you feel are most strongly linked': 'which you connect most strongly',
    'elements are shown individually': 'we show elements individually',
    'respondents were asked': 'we asked people',
    'were shown': 'saw',

    # Technical → Accessible
    'brand attribution': 'who it belongs to',
    'distinctive ownership': 'unique to Škoda',
    'brand asset': 'brand element',
    'performance tier': 'performance level',
    'efficiency metric': 'value for money',
    'implementation': 'use',
    'deployment': 'use',
    'utilization': 'use',

    # Shorten captions (remove redundancy)
    '(Q02: \'Have you seen/heard this element before?\')': '',
    '(Q05: \'Which brand do you think this belongs to?\')': '',
    '(Q29 MaxDiff ranking: \'Which elements are most strongly linked to Škoda?\')': '',
    '(Q30: \'What are the 3 words that come top of mind when thinking of Škoda?\')': '',
    '(Q04 adjective scales': '',

    # Improve specific captions
    'Consumer perception analysis based on Q04 semantic differential scales': 'How people feel about each element - positive vs negative',
    'Which brands do consumers think these elements belong to?': 'Do people think these belong to Škoda or competitors?',
    'Market consistency, consumer associations (Q03), and brand attribution (Q05)': 'Market patterns and what people say about each element',

    # Info boxes - add actionable context
    'This framework categorizes brand elements based on combined analysis': 'We group elements by how well they perform, so you can:',
    'Analysis of elements with high uniqueness but low deployment': 'Find elements that stand out but aren\'t used enough - these are opportunities',

    # Tooltips - make friendlier
    'Elements meeting all high-performance criteria': 'Top performers on all measures',
    'Elements with strong performance in key metrics': 'Strong in at least one key area',
    'Optimization needed': 'Could perform better',
    'Elements with uniqueness ≥25% and usage <40%': 'Distinctive but underused',
}

# Apply replacements
for old, new in replacements.items():
    content = content.replace(old, new)

# Save the improved version
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Copy improvements applied!")
print(f"   - Removed research jargon")
print(f"   - Shortened verbose text")
print(f"   - Added context and value")
print(f"   - Converted passive to active voice")
print(f"   - Made tooltips friendlier")
