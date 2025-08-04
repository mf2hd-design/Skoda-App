import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from PIL import Image
import json
import os
from io import BytesIO

# --- Define Brand Elements (must match the order in your example image) ---
brand_elements = [
    "Electric Green", "Dark Green", "Type", "Symbol",
    "Sonic", "Hacek", "Wordmark", "Facets", "Illustrations", "Vehicle imagery"
]

# --- YOUR MANUALLY VERIFIED FEW-SHOT TRAINING DATA ---
TRAINING_DATA = {
    "ukelroq.jpg": { "Description": "UK Elroq Ad", "Analysis": { "Electric Green": False, "Dark Green": False, "Type": True, "Symbol": False, "Sonic": False, "Hacek": False, "Wordmark": True, "Facets": False, "Illustrations": False, "Vehicle imagery": True }},
    "spainelroq.jpg": { "Description": "Spain Nuevo Elroq Ad", "Analysis": { "Electric Green": False, "Dark Green": False, "Type": True, "Symbol": False, "Sonic": False, "Hacek": False, "Wordmark": False, "Facets": False, "Illustrations": False, "Vehicle imagery": True }},
    "superbhybrid.jpg": { "Description": "Croatia Superb Hybrid Ad", "Analysis": { "Electric Green": True, "Dark Green": False, "Type": True, "Symbol": False, "Sonic": False, "Hacek": False, "Wordmark": True, "Facets": False, "Illustrations": False, "Vehicle imagery": True }}
}

# --- App UI and Logic ---
st.set_page_config(layout="wide")
st.title("Škoda AI Brand Equity & Performance Analyzer")
st.markdown("""
This advanced demo uses **Few-Shot Prompting** to specialize the AI for Škoda's brand,
providing a comprehensive analysis of ad creatives based on detected brand elements and performance metrics.
""")

# --- Configure the AI Model ---
try:
    genai.configure(api_key=st.secrets["google_api_key"])
    model = genai.GenerativeModel('gemini-2.5-flash') 
except Exception as e:
    st.error(f"**Configuration Error:** API Key failed. Please ensure your `.streamlit/secrets.toml` file is correctly set up with `google_api_key`.")
    st.stop()

# --- Helper function to build the few-shot prompt ---
def build_few_shot_prompt(user_image):
    prompt_parts = [
        "You are a highly accurate Škoda brand asset auditor. Analyze the final user-provided image based on the following examples of correct analysis. Respond ONLY with a valid JSON object.",
        "\n--- START OF EXAMPLES ---\n"
    ]
    for filename, data in TRAINING_DATA.items():
        try:
            image_path = os.path.join("training_images", filename)
            if not os.path.exists(image_path):
                st.error(f"**Training Data Error:** Training image not found: `{image_path}`. Please check your 'training_images' folder.")
                return None
            training_image = Image.open(image_path)
            prompt_parts.extend([f"\n--- EXAMPLE FOR '{data['Description']}' ---", training_image, f"CORRECT OUTPUT:\n{json.dumps(data['Analysis'], indent=2)}"])
        except Exception as e:
            st.error(f"**Training Data Error:** Could not load training example `{filename}`. Error: {e}")
            return None
    prompt_parts.extend(["\n--- END OF EXAMPLES ---", "\nNow, analyze the following new image provided by the user and provide the JSON output in the exact same format.", user_image])
    return prompt_parts

# Display the training data used by the AI
with st.expander("Click to see the Few-Shot Training Data used by the AI"):
    st.markdown("The AI learns from these examples to accurately identify Škoda brand elements.")
    for filename, data in TRAINING_DATA.items():
        st.write(f"**Training File:** `{filename}` ({data['Description']})")
        image_path = os.path.join("training_images", filename)
        if os.path.exists(image_path): st.image(image_path, width=200)
        st.json(data['Analysis'])

# --- Flexible Data Input ---
st.header("1. Input Your Ad Creative Data")
input_method = st.radio("Choose your data input method:", ("Upload Images Manually", "Upload CSV/Excel File"), horizontal=True)

ad_data_for_analysis = [] # This will store the final list of dicts for processing

if input_method == "Upload Images Manually":
    uploaded_files = st.file_uploader("Upload your ad creative image files (JPG, PNG)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    if uploaded_files:
        st.subheader("Enter Performance Metrics for Each Creative:")
        cols = st.columns(len(uploaded_files))
        for i, uploaded_file in enumerate(uploaded_files):
            with cols[i]:
                st.image(uploaded_file, caption=uploaded_file.name, width=150)
                spend = st.number_input(f"Spend for {uploaded_file.name} (€)", min_value=0.0, key=f"manual_spend_{i}")
                reach = st.number_input(f"Reach for {uploaded_file.name}", min_value=0, key=f"manual_reach_{i}")
                ad_data_for_analysis.append({'Creative Name': uploaded_file.name, 'Spend': spend, 'Reach': reach, 'file_object': uploaded_file})
    else:
        st.info("Please upload image files to proceed.")

else: # Upload CSV/Excel File
    uploaded_data_file = st.file_uploader("Upload your CSV or Excel file (must contain 'Creative Name', 'Spend', 'Reach', and 'Image_URL' columns)", type=["csv", "xlsx"])
    if uploaded_data_file:
        try:
            if uploaded_data_file.name.endswith('.csv'):
                df_input = pd.read_csv(uploaded_data_file)
            else: # .xlsx
                df_input = pd.read_excel(uploaded_data_file)
            
            # Validate required columns
            required_cols = ['Creative Name', 'Spend', 'Reach', 'Image_URL']
            if not all(col in df_input.columns for col in required_cols):
                st.error(f"**Data Error:** Your file must contain the columns: {', '.join(required_cols)}")
                st.stop()

            # Convert types and handle missing values
            df_input['Spend'] = pd.to_numeric(df_input['Spend'], errors='coerce').fillna(0)
            df_input['Reach'] = pd.to_numeric(df_input['Reach'], errors='coerce').fillna(0)
            df_input.dropna(subset=['Image_URL'], inplace=True) # Drop rows with no image URL

            st.success(f"Successfully loaded {len(df_input)} ads from your file. Ready for AI analysis.")
            st.dataframe(df_input.head()) # Show a preview

            # Prepare data for analysis loop (fetch images from URLs)
            for index, row in df_input.iterrows():
                try:
                    image_url = row['Image_URL']
                    response = requests.get(image_url, timeout=10) # Shorter timeout for URLs
                    response.raise_for_status()
                    image_file_object = BytesIO(response.content)
                    ad_data_for_analysis.append({
                        'Creative Name': row['Creative Name'],
                        'Spend': row['Spend'],
                        'Reach': row['Reach'],
                        'file_object': image_file_object # Pass BytesIO object
                    })
                except Exception as e:
                    st.warning(f"**Image Load Warning:** Could not load image for '{row['Creative Name']}' from URL: {e}. This ad will be skipped.")
        except Exception as e:
            st.error(f"**File Read Error:** Could not read your file. Please ensure it's a valid CSV/Excel and columns are correct. Error: {e}")
            st.stop()
    else:
        st.info("Please upload a CSV or Excel file to proceed.")

# --- Step 3: Run Analysis Button ---
if ad_data_for_analysis and st.button("2. Start AI Analysis on All Ads"):
    st.header("3. Running AI Analysis...")
    with st.spinner("Analyzing creatives with Gemini AI... This may take a moment for multiple images."):
        raw_results_list = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, ad in enumerate(ad_data_for_analysis):
            status_text.text(f"Processing: {ad['Creative Name']} ({i+1}/{len(ad_data_for_analysis)})")
            try:
                user_image = Image.open(ad['file_object'])
                few_shot_prompt = build_few_shot_prompt(user_image)
                if few_shot_prompt is None: continue # Skip if training images are missing
                
                ai_response = model.generate_content(few_shot_prompt, request_options={"timeout": 90})
                response_text = ai_response.text.strip().replace("```json", "").replace("```", "")
                ai_analysis_result = json.loads(response_text)
                
                # Calculate Brand Consistency Score
                detected_count = sum(1 for val in ai_analysis_result.values() if val is True)
                consistency_score = (detected_count / len(brand_elements)) if len(brand_elements) > 0 else 0

                new_row = {
                    'Creative Name': ad['Creative Name'],
                    'Spend': ad['Spend'],
                    'Reach': ad['Reach'],
                    'Brand Consistency Score': consistency_score # Add new score
                }
                new_row.update(ai_analysis_result)
                raw_results_list.append(new_row)
            except Exception as e:
                st.error(f"**AI Analysis Error:** Failed to analyze '{ad['Creative Name']}': {e}")
                continue
            progress_bar.progress((i + 1) / len(ad_data_for_analysis))
        
        if not raw_results_list:
            st.error("AI analysis completed, but no valid results were generated. Please check your inputs and API key.")
            st.stop()
        
        df_raw_results = pd.DataFrame(raw_results_list)
        status_text.success("AI Analysis Complete!")

        # --- Step 3a: AI Audit Results (Per Creative) ---
        st.header("3a. AI Audit Results (Per Creative)")
        st.markdown("Click on each creative to see the specific brand elements the AI identified.")
        for i, row in df_raw_results.iterrows():
            with st.expander(f"**{row['Creative Name']}** - Brand Consistency: {row['Brand Consistency Score']:.1%}"):
                detected = [el for el in brand_elements if row.get(el, False)]
                st.success(f"**Detected Elements:** {', '.join(detected) if detected else 'None'}")
                st.write(f"**Spend:** €{row['Spend']:,.2f} | **Reach:** {row['Reach']:,}")
        
        # --- Step 4: Combined Media & Survey Analysis ---
        st.header("4. Combined Media & Survey Analysis")
        st.markdown("This table synthesizes the AI-driven media audit with simulated survey data, providing a holistic view of brand asset performance and equity.")
        
        total_ads = len(df_raw_results)
        media_metrics = []
        for element in brand_elements:
            if element in df_raw_results.columns:
                filtered_df = df_raw_results[df_raw_results[element] == True]
            else:
                filtered_df = pd.DataFrame(columns=df_raw_results.columns)
            
            media_metrics.append({
                'Element': element, 
                '% used': (len(filtered_df) / total_ads) if total_ads > 0 else 0, 
                'avg reach': filtered_df['Reach'].mean(), 
                'avg spend': filtered_df['Spend'].mean()
            })
        media_df = pd.DataFrame(media_metrics).set_index('Element')

        # --- Create Dummy Survey Data ---
        survey_data = {
            'Element': brand_elements, 
            '% recognised': [0.80, 0.47, 0.78, 0.22, 0.64, 0.52, 0.23, 0.14, 0.74, 0.85], 
            'Positive associations': [0.70, 0.39, 0.29, 0.59, 0.21, 0.35, 0.76, 0.33, 0.13, 0.60], 
            'Negative associations': [0.30, 0.30, 0.51, 0.11, 0.78, 0.46, 0.15, 0.58, 0.39, 0.20], 
            'Attributable to Skoda': [0.48, 0.21, 0.81, 0.22, 0.66, 0.21, 0.72, 0.39, 0.26, 0.90], 
            'Uniqueness': [0.51, 0.29, 0.90, 0.94, 0.53, 0.73, 0.46, 0.54, 0.80, 0.25]
        }
        survey_df = pd.DataFrame(survey_data).set_index('Element')
        
        master_df = pd.concat([media_df.T, survey_df.T])
        master_df = master_df.rename(index={'avg spend': 'avg spend (avg € on ads that include this element)'})
        
        percent_rows = ['% used', '% recognised', 'Positive associations', 'Negative associations', 'Attributable to Skoda', 'Uniqueness']
        currency_rows = ['avg spend (avg € on ads that include this element)']
        number_rows = ['avg reach']

        styler = master_df.fillna(0).style
        styler = styler.background_gradient(cmap='RdYlGn', axis=1)
        
        styler = styler.format("{:,.0f}", subset=(pd.IndexSlice[number_rows], slice(None)))
        styler = styler.format("€{:,.2f}", subset=(pd.IndexSlice[currency_rows], slice(None)))
        styler = styler.format("{:.1%}", subset=(pd.IndexSlice[percent_rows], slice(None)))
        
        st.dataframe(styler)

        # --- Step 5: Brand Equity Matrix ---
        st.header("5. Brand Equity Matrix")
        plot_data = master_df.T.reset_index().rename(columns={'% recognised': 'Recognition (Fame)', 'avg spend (avg € on ads that include this element)': 'avg spend'})
        
        plot_data['avg spend'] = pd.to_numeric(plot_data['avg spend'].astype(str).str.replace('€', '').str.replace(',', ''), errors='coerce').fillna(0)

        fig_matrix = px.scatter(
            plot_data, x="Uniqueness", y="Recognition (Fame)", 
            size="avg spend", color="Positive associations", 
            text="Element", size_max=60, hover_name="Element", 
            color_continuous_scale='RdYlGn', 
            title="Fame vs. Uniqueness (Size by Avg Spend, Color by Positive Associations)"
        )
        fig_matrix.update_traces(textposition='top center')
        fig_matrix.add_vline(x=0.5, line_width=1, line_dash="dash", line_color="grey")
        fig_matrix.add_hline(y=0.5, line_width=1, line_dash="dash", line_color="grey")
        fig_matrix.update_layout(
            xaxis=dict(tickformat=".0%"), 
            yaxis=dict(tickformat=".0%"),
            annotations=[
                dict(x=0.25, y=0.9, text="<b>Core Assets</b><br>(High Fame, Low Uniqueness)", showarrow=False, font=dict(color="grey")),
                dict(x=0.75, y=0.9, text="<b>Differentiating Assets</b><br>(High Fame, High Uniqueness)", showarrow=False, font=dict(color="green")),
                dict(x=0.25, y=0.1, text="<b>Potential Sleepers</b><br>(Low Fame, Low Uniqueness)", showarrow=False, font=dict(color="grey")),
                dict(x=0.75, y=0.1, text="<b>Future Stars</b><br>(Low Fame, High Uniqueness)", showarrow=False, font=dict(color="orange"))
            ]
        )
        st.plotly_chart(fig_matrix, use_container_width=True)

        # --- Step 6: Strategic Recommendations ---
        st.header("6. Strategic Recommendations")
        
        # Calculate average Brand Consistency Score
        avg_consistency = df_raw_results['Brand Consistency Score'].mean()
        
        st.info(f"""
        **Overall Brand Consistency:** On average, your analyzed creatives achieved a **{avg_consistency:.1%}** Brand Consistency Score, meaning they utilized this percentage of your defined core brand elements.

        - **Core Assets (`Wordmark`, `Vehicle imagery`):** These elements show high recognition and attribution, but lower uniqueness. Your investment in them is crucial for basic brand identification. **Strategy:** Protect their consistent usage, but explore creative ways to pair them with more unique elements to avoid visual fatigue.
        - **Differentiating Assets (`Symbol`, `Illustrations`):** These elements boast high uniqueness but currently have low usage and associated spend. **Strategy:** They are powerful, underleveraged assets. Consider A/B testing campaigns that prominently feature these elements to boost brand distinction and memorability.
        - **High Potential (`Electric Green`):** This asset demonstrates strong positive associations and high recognition. **Strategy:** It's a prime candidate for a 'non-negotiable' element in modern, future-facing communications, especially for electric vehicle messaging.
        - **Efficiency Concern (`Sonic`):** This element shows high negative associations in the simulated survey data. **Strategy:** Before further investment, it's critical to investigate *why* this negative sentiment exists. This could involve qualitative research or targeted testing to understand the perception.
        """)