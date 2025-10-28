# FILE: app.py
# (Final version with new title and tagline)

import streamlit as st
import pandas as pd
import re
# Import the functions from your logic file
from analyzer_logic import (
    load_all_models, 
    extract_aspect_sentiments
)

# --- Page Configuration ---
st.set_page_config(
    page_title="AspectLens", # <-- Changed browser tab title
    page_icon="🔎",        # <-- Changed browser tab icon
    layout="wide" 
)

# --- ABOUT SECTION IN SIDEBAR ---
st.sidebar.title("🛠️ About This Project")
st.sidebar.markdown("""
This app is an end-to-end Natural Language Processing (NLP) project that performs **Aspect-Based Sentiment Analysis**.

---

### Key Components:

* **Frontend:** The UI is built with **Streamlit**.

* **Core Model:** We use a custom **Bidirectional LSTM** (a deep learning, recurrent neural network) built with **TensorFlow (Keras)**.

* **Training Data:** The model was trained on the **Flipkart Product Review Dataset**. We specifically trained it on the positive and negative reviews, removing the ambiguous "neutral" class.

* **Data Balancing:** We used **Scikit-learn's** `class_weight` utility during training to force the model to pay equal attention to the (much rarer) negative reviews, which fixed our model's bias.

* **Aspect Extraction:** **spaCy** is used for linguistic preprocessing. It tokenizes the review, identifies the Parts-of-Speech (like nouns), and finds the base form (lemma) of each word.

* **Aspect Clustering:** We use **NLTK (WordNet)** to measure the *semantic similarity* between aspects. This allows the app to group related words (e.g., 'battery' and 'life') and average their scores.
""")

# --- Sample Review ---
SAMPLE_REVIEW = """
I have been using this phone for a week. The camera quality is absolutely amazing, especially in low light. 
The pictures are so vibrant.
However, the battery life is a huge disappointment. It barely lasts a full day with moderate usage, and the charging is slow. 
The screen is bright and vibrant, making videos look great. But I must say, the battery is a major drawback.
"""

# --- Load models ---
with st.spinner("Loading NLP models... This will run once."):
    nlp, model, tokenizer = load_all_models()

# --- UI Layout ---

# *** NEW TITLE ***
st.title("🔎 AspectLens") 

# *** NEW SUBTITLE (TAGLINE) ***
st.markdown("Uncovering the sentiment behind every feature. Check the sidebar for project details!")


# --- Create two main columns for a cleaner side-by-side look ---
col1, col2 = st.columns(2)

with col1:
    # --- Input Area ---
    with st.container(border=True): # Use a simple border
        st.subheader("Enter Your Review")
        review_text = st.text_area("Review Text:", height=300, value=SAMPLE_REVIEW, label_visibility="collapsed")
        analyze_button = st.button("Analyze Review", type="primary", use_container_width=True)

with col2:
    # --- Results Area ---
    with st.container(border=True): # Use a simple border
        st.subheader("Detailed Aspect Analysis")
        
        # This is a placeholder that will be filled when the button is pressed
        results_container = st.empty() 
        results_container.markdown("Click 'Analyze Review' to see the results here.")

        if analyze_button:
            if not review_text.strip():
                results_container.error("Please enter a review to analyze.")
            else:
                with st.spinner("Analyzing..."):
                    clean_review = re.sub(r'\s+', ' ', review_text).strip()
                    doc = nlp(clean_review)
                    results = extract_aspect_sentiments(doc, model, tokenizer)
                    
                    if not results:
                        results_container.warning("No specific aspects were detected.")
                    else:
                        # Display results in the clean table
                        df = pd.DataFrame(results)
                        
                        # Apply color formatting
                        def style_sentiment(val):
                            if val == "Positive":
                                return "color: #2a9d8f; font-weight: bold;"
                            elif val == "Negative":
                                return "color: #e76f51; font-weight: bold;"
                            else:
                                return "color: #e9c46a;"
                        
                        # Clear the placeholder and show the table
                        results_container.dataframe(
                            df.style.applymap(style_sentiment, subset=['Sentiment']),
                            use_container_width=True
                        )