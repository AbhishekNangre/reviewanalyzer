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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

print("Loading models... This may take a moment.")

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None

# *** CHANGED: Load the new BINARY model ***
model = load_model('product_sentiment_BINARY_model.keras')

# *** CHANGED: Load the new BINARY tokenizer ***
with open('product_BINARY_tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

print("All models loaded.")

MAX_LEN = 150      

def preprocess_text_for_keras(text):
    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

def predict_sentiment_binary(text):
    """
    Uses our binary model to predict sentiment.
    Returns a single score from 0.0 (Neg) to 1.0 (Pos).
    """
    if not text.strip():
        return 0.5 # Default to Neutral
        
    processed_seq = preprocess_text_for_keras(text)
    
    # *** CHANGED: Prediction is now a single value (e.g., 0.95 or 0.02) ***
    prediction = model.predict(processed_seq, verbose=0)[0][0]
    return prediction

def extract_aspect_sentiments(doc):
    aspect_sentiments = defaultdict(list)
    noise_filter = {'phone', 'week', 'day', 'usage', 'video', 'light'}
    
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
                
                # *** CHANGED: Use the new binary prediction function ***
                score = predict_sentiment_binary(sentence_text)
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
    final_results = {}
    for aspect, scores in clustered_results.items():
        if scores:
            avg_score = np.mean(scores)
            # *** CHANGED: Simple thresholds for binary score ***
            sentiment = "Positive" if avg_score > 0.6 else "Negative" if avg_score < 0.4 else "Neutral"
            final_results[aspect] = (sentiment, np.mean(scores), len(scores))

    return final_results

# --- Main Execution ---
if __name__ == "__main__":
    if not nlp:
        print("spaCy model not loaded. Exiting.")
        exit()
        
    sample_review = """
    I have been using this phone for a week. The camera quality is absolutely amazing, especially in low light. 
    The pictures are so vibrant.
    However, the battery life is a huge disappointment. It barely lasts a full day with moderate usage, and the charging is slow. 
    The screen is bright and vibrant, making videos look great. But I must say, the battery is a major drawback.
    """
    
    clean_review = re.sub(r'\s+', ' ', sample_review).strip()
    print("\n--- Smart Review Analyzer ---")
    print(f"\nAnalyzing Review:\n'{clean_review}'")
    print("\n" + "="*50 + "\n")

    doc = nlp(clean_review)
    results = extract_aspect_sentiments(doc)
    
    print("Aspect-Based Sentiment Results:")
    if not results:
        print("No specific aspects with sentiment were detected.")
    else:
        for aspect, (sentiment, score, count) in results.items():
            print(f"  -> Aspect: {aspect.capitalize()}")
            print(f"     Sentiment: {sentiment} (Score: {score:.2f}, Mentions: {count})")