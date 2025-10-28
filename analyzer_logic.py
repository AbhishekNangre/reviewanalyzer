# FILE: analyzer_logic.py
# (This file is correct and doesn't need changes from the last step)

import spacy
import json
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
from nltk.corpus import wordnet
from collections import defaultdict
import os
import streamlit as st 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

@st.cache_resource
def load_all_models():
    """Loads all models and resources, and caches them."""
    print("Loading models... This will only run once.")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("spaCy model not loaded. Please run: python -m spacy download en_core_web_sm")
        nlp = None

    model = load_model('product_sentiment_BINARY_model.keras')
    with open('product_BINARY_tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    
    print("All models loaded.")
    return nlp, model, tokenizer

# --- Global constants and helper functions ---
MAX_LEN = 150
noise_filter = {'phone', 'week', 'day', 'usage', 'video', 'light'}

def preprocess_text_for_keras(text, tokenizer):
    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

def predict_sentiment_binary(text, model, tokenizer):
    if not text.strip():
        return 0.5 
    processed_seq = preprocess_text_for_keras(text, tokenizer)
    prediction = model.predict(processed_seq, verbose=0)[0][0]
    return prediction

# --- FUNCTION get_overall_sentiment REMOVED ---

def extract_aspect_sentiments(doc, model, tokenizer):
    """The main aspect extraction and clustering logic."""
    aspect_sentiments = defaultdict(list)
    
    all_aspects = []
    for token in doc:
        if token.pos_ == 'NOUN' and not token.is_stop and not token.is_punct:
            if token.lemma_ not in noise_filter:
                all_aspects.append(token.lemma_)
    unique_aspects = set(all_aspects)

    for aspect in unique_aspects:
        for sent in doc.sents: 
            if aspect in [token.lemma_ for token in sent]:
                sentence_text = sent.text
                score = predict_sentiment_binary(sentence_text, model, tokenizer)
                aspect_sentiments[aspect].append(score)

    # --- Clustering (WordNet) ---
    clustered_results = defaultdict(list)
    processed_aspects = set()
    all_aspect_keys = list(aspect_sentiments.keys())

    for i in range(len(all_aspect_keys)):
        aspect1_name = all_aspect_keys[i]
        if aspect1_name in processed_aspects: continue
        
        aspect1_synset = wordnet.synsets(aspect1_name, pos=wordnet.NOUN)
        if not aspect1_synset:
            clustered_results[aspect1_name].extend(aspect_sentiments[aspect1_name])
            processed_aspects.add(aspect1_name)
            continue
        
        aspect1_synset = aspect1_synset[0]
        cluster_name = aspect1_name
        clustered_results[cluster_name].extend(aspect_sentiments[aspect1_name])
        processed_aspects.add(aspect1_name)

        for j in range(i + 1, len(all_aspect_keys)):
            aspect2_name = all_aspect_keys[j]
            if aspect2_name in processed_aspects: continue
            
            aspect2_synset = wordnet.synsets(aspect2_name, pos=wordnet.NOUN)
            if not aspect2_synset: continue
            aspect2_synset = aspect2_synset[0]
            
            similarity = aspect1_synset.path_similarity(aspect2_synset)
            
            if similarity and similarity > 0.3: 
                clustered_results[cluster_name].extend(aspect_sentiments[aspect2_name])
                processed_aspects.add(aspect2_name)

    # --- Averaging ---
    final_results = [] 
    for aspect, scores in clustered_results.items():
        if scores:
            avg_score = np.mean(scores)
            sentiment = "Positive" if avg_score > 0.6 else "Negative" if avg_score < 0.4 else "Neutral"
            final_results.append({
                "Aspect": aspect.capitalize(),
                "Sentiment": sentiment,
                "Score": f"{avg_score:.2f}",
                "Mentions": len(scores)
            })

    return final_results