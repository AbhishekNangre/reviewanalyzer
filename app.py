# setup.py

import nltk
import spacy.cli

def download_nltk_packages():
    """Downloads all necessary data packages from NLTK."""
    print("--- Starting NLTK Data Downloads ---")
    packages = [
        'punkt',         # For tokenizing sentences
        'stopwords',     # For filtering common words
        'vader_lexicon', # For sentiment analysis
        'averaged_perceptron_tagger' # For Part-of-Speech tagging
    ]
    for package in packages:
        try:
            print(f"-> Downloading '{package}'...")
            nltk.download(package)
        except Exception as e:
            print(f"    Error downloading {package}: {e}")
    print("--- NLTK downloads complete! ---\n")

def download_spacy_model():
    """Downloads the necessary model for spaCy."""
    print("--- Starting spaCy Model Download ---")
    try:
        spacy.cli.download("en_core_web_sm")
        print("\n✅ spaCy model download complete!")
    except Exception as e:
        print(f"\n❌ An error occurred during spaCy download: {e}")

if __name__ == "__main__":
    download_nltk_packages()
    download_spacy_model()
    print("\n--- All setup is complete! Your project is now ready to run. ---")