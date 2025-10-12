# review_analyzer.py

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import spacy
import re
from collections import Counter

# Load the spaCy model once at the start
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Please run the setup.py script first.")
    nlp = None

def analyze_sentiment_per_sentence(text):
    sia = SentimentIntensityAnalyzer()
    sentences = sent_tokenize(text)
    results = []
    
    for sentence in sentences:
        score = sia.polarity_scores(sentence)['compound']
        if score >= 0.05:
            sentiment = "Positive"
        elif score <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        results.append((sentence, sentiment))
    return results

def extract_key_aspects_with_spacy(text, top_n=5):
    if not nlp:
        return ["spaCy model is not loaded."]
        
    doc = nlp(text.lower())
    
    aspects = []
    for token in doc:
        if not token.is_stop and not token.is_punct and token.pos_ in ("NOUN", "ADJ"):
            aspects.append(token.lemma_)
    
    freq_counter = Counter(aspects)
    return [word for word, freq in freq_counter.most_common(top_n)]

if __name__ == "__main__":
    
    sample_review = """
    I have been using this phone for a week. The camera quality is absolutely amazing, especially in low light. 
    However, the battery life is a huge disappointment. It barely lasts a full day with moderate usage, and the charging is slow. 
    The screen is bright and vibrant, making videos look great. But I must say, the battery is a major drawback.
    """
    
    clean_review = re.sub(r'\s+', ' ', sample_review).strip()

    print("\n--- Smart Review Analyzer ---")
    print(f"\nAnalyzing Review:\n'{clean_review}'")
    print("\n" + "="*50 + "\n")

    print("1. Sentiment Analysis (Sentence by Sentence):")
    sentiment_results = analyze_sentiment_per_sentence(clean_review)
    for sentence, sentiment in sentiment_results:
        print(f"   [{sentiment:^8}] {sentence}")
    print("\n")

    print("2. Key Aspects Mentioned (Nouns & Adjectives):")
    keywords = extract_key_aspects_with_spacy(clean_review)
    print(f"   -> {', '.join(keywords)}\n")