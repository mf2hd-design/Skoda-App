import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from PIL import Image
import json
import os
from io import BytesIO

# --- Define Brand Elements ---
brand_elements = [
    "Electric Green", "Dark Green", "Type", "Symbol",
    "Hacek", "Wordmark", "Facets"
]

# --- YOUR MANUALLY VERIFIED FEW-SHOT TRAINING DATA ---
TRAINING_DATA = {
    "ukelroq.jpg": { "Description": "UK Elroq Ad", "Analysis": { "Electric Green": False, "Dark Green": False, "Type": True, "Symbol": False, "Sonic": False, "Hacek": False, "Wordmark": True, "Facets": False, "Illustrations": False, "Vehicle imagery": True }},
    "spainelroq.jpg": { "Description": "Spain Nuevo Elroq Ad", "Analysis": { "Electric Green": False, "Dark Green": False, "Type": True, "Symbol": False, "Sonic": False, "Hacek": False, "Wordmark": False, "Facets": False, "Illustrations": False, "Vehicle imagery": True }},
    "superbhybrid.jpg": { "Description": "Croatia Superb Hybrid Ad", "Analysis": { "Electric Green": True, "Dark Green": False, "Type": True, "Symbol": False, "Sonic": False, "Hacek": False, "Wordmark": True, "Facets": False, "Illustrations": False, "Vehicle imagery": True }}
}

# --- App UI and Logic ---
st.set_page_config(layout="wide")
st.title("Brand Element Impact Tracker")
st.markdown("This AI-powered tool analyzes ad creatives to measure the usage and equity of key brand elements.")

# --- Configure the AI Model ---
try:
    genai.configure(api_key=st.secrets["google_api_key"])
    model = genai.GenerativeModel('gemini-1.5-flash-latest') 
except Exception as e:
    st.error(f"API Key configuration failed. Check your .streamlit/secrets.toml file. Error: {e}")
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
                st.error(f"Training image not found: {image_path}. Please check your 'training_images' folder.")
                return None
            training_image = Image.open(image_path)
            prompt_parts.extend([f"\n--- EXAMPLE FOR '{data['Description']}' ---", training_image, f"CORRECT OUTPUT:\n{json.dumps(data['Analysis'], indent=2)}"])
        except Exception as e:
            st.error(f"Could not load training example {filename}. Error: {e}")
            return None
    prompt_parts.extend(["\n--- END OF EXAMPLES ---", "\nNow, analyze the following new image provided by the user and provide the JSON output in the exact same format.", user_image])
    return prompt_parts

# --- Helper function to create a downloadable Excel file ---
def to_excel(df):
    output = BytesIO()
    # Use xlsxwriter to create a more polished Excel file if available
    try:
        import xlsxwriter
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
    except ImportError:
        writer = pd.ExcelWriter(output, engine='openpyxl')
    
    df.to_excel(writer, index=True, sheet_name='Analysis')
    writer.close()
    processed_data = output.getvalue()
    return processed_data


# --- Step 1: File Upload ---
st.header("1. Input Your Ad Creative Data")
uploaded_files = st.file_uploader(
    "Upload 1 to 30 of your ad creative files (JPG, PNG)",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    st.header("2. Enter Performance Metrics")
    use_avg_spend = st.checkbox("Apply a single average spend to all creatives?")
    avg_spend = 0
    if use_avg_spend:
        avg_spend = st.number_input("Enter the average spend per creative (€)", min_value=0.0, value=1000.0)

    ad_metrics = []
    cols = st.columns(len(uploaded_files))
    for i, uploaded_file in enumerate(uploaded_files):
        with cols[i]:
            # ----- THIS IS THE CORRECTED LINE -----
            st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
            # ------------------------------------
            spend = st.number_input(
                f"Spend (€)", 
                min_value=0.0, 
                key=f"spend_{i}", 
                value=float(avg_spend) if use_avg_spend else 0.0,
                disabled=use_avg_spend
            )
            ad_metrics.append({'file': uploaded_file, 'spend': spend})

    # --- Step 3: Run Analysis ---
    st.header("3. Run AI Analysis")
    if st.button("Start AI Analysis on All Ads"):
        if 'session_results' not in st.session_state:
            st.session_state.session_results = []
        
        with st.spinner("Running AI Audit using Few-Shot examples..."):
            raw_results_list = []
            for i, ad in enumerate(ad_metrics):
                try:
                    user_image = Image.open(ad['file'])
                    few_shot_prompt = build_few_shot_prompt(user_image)
                    if few_shot_prompt is None: st.stop()
                    ai_response = model.generate_content(few_shot_prompt, request_options={"timeout": 90})
                    response_text = ai_response.text.strip().replace("```json", "").replace("```", "")
                    ai_analysis_result = json.loads(response_text)
                    new_row = {'Creative Name': ad['file'].name, 'Spend': ad['spend']}
                    new_row.update(ai_analysis_result)
                    raw_results_list.append(new_row)
                except Exception as e:
                    st.error(f"AI analysis for {ad['file'].name} failed: {e}")
                    continue
            if not raw_results_list:
                st.error("AI analysis failed for all uploaded images.")
                st.stop()
            
            st.session_state.session_results = pd.DataFrame(raw_results_list)

    if 'session_results' in st.session_state and not st.session_state.session_results.empty:
        df_raw_results = st.session_state.session_results

        st.header("3a. AI Audit Results (Per Creative)")
        for i, row in df_raw_results.iterrows():
            with st.expander(f"**{row['Creative Name']}** - Click to see detected elements"):
                detected = [el for el in brand_elements if row.get(el, False)]
                st.success(f"**Detected Elements:** {', '.join(detected) if detected else 'None'}")
        
        total_ads = len(df_raw_results)
        media_metrics = []
        for element in brand_elements:
            filtered_df = df_raw_results[df_raw_results[element] == True] if element in df_raw_results and element in df_raw_results.columns else pd.DataFrame(columns=df_raw_results.columns)
            media_metrics.append({'Element': element, '% used': (len(filtered_df) / total_ads), 'avg spend': filtered_df['Spend'].mean()})
        media_df = pd.DataFrame(media_metrics).set_index('Element')

        survey_data = {
            'Element': brand_elements, 
            '% recognised': [0.80, 0.47, 0.78, 0.22, 0.52, 0.23, 0.14], 
            'Positive associations': [0.70, 0.39, 0.29, 0.59, 0.35, 0.76, 0.33], 
            'Negative associations': [0.30, 0.30, 0.51, 0.11, 0.46, 0.15, 0.58], 
            'Uniqueness': [0.51, 0.29, 0.90, 0.94, 0.73, 0.46, 0.54]
        }
        survey_df = pd.DataFrame(survey_data).set_index('Element')
        
        master_df = pd.concat([media_df.T, survey_df.T])
        master_df = master_df.rename(index={'avg spend': 'avg spend (avg € on ads that include this element)'})
        
        st.header("4. Combined Media & Survey Analysis")
        st.markdown("This table synthesizes the AI-driven media audit with simulated survey data for the current session.")
        
        styler = master_df.fillna(0).style
        heatmap_rows = ['% recognised', 'Positive associations', 'Negative associations', 'Uniqueness']
        styler = styler.background_gradient(cmap='RdYlGn', axis=1, subset=(pd.IndexSlice[heatmap_rows], slice(None)))
        
        percent_rows = ['% used', '% recognised', 'Positive associations', 'Negative associations', 'Uniqueness']
        styler = styler.format("€{:,.2f}", subset=(pd.IndexSlice[['avg spend (avg € on ads that include this element)']], slice(None)))
        styler = styler.format("{:.1%}", subset=(pd.IndexSlice[percent_rows], slice(None)))
        
        st.dataframe(styler)

        excel_file = to_excel(master_df.fillna(0))
        st.download_button(
            label="📥 Export Analysis to Excel",
            data=excel_file,
            file_name="skoda_brand_analysis_session.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.header("5. Brand Equity Matrix")
        plot_data = master_df.T.reset_index().rename(columns={'% recognised': 'Recognition (Fame)', 'avg spend (avg € on ads that include this element)': 'avg spend'})
        plot_data['avg spend'] = pd.to_numeric(plot_data['avg spend'], errors='coerce').fillna(0)

        fig_matrix = px.scatter(
            plot_data, x="Uniqueness", y="Recognition (Fame)", 
            size="avg spend", color="Positive associations", 
            text="Element", size_max=60, hover_name="Element", 
            color_continuous_scale='RdYlGn', 
            title="Fame vs. Uniqueness (Size by Avg Spend, Color by Positive Associations)"
        )
        fig_matrix.update_layout(legend_title_text='Positive Associations')
        st.plotly_chart(fig_matrix, use_container_width=True)