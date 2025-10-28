# 🔎 AspectLens: Aspect-Based Sentiment Analysis

AspectLens is an end-to-end Natural Language Processing (NLP) application that analyzes product reviews to uncover sentiment for specific *features* or *aspects*.

Instead of just giving a single "Positive" or "Negative" score for an entire review, AspectLens tells you *why* a review is good or bad. For example, it can determine that a phone's "camera" was "Positive" while its "battery" was "Negative," all from the same block of text.



---

## 🚀 How It Works: The NLP Pipeline

This project is split into two main phases: an "offline" training phase to build the model and a "live" analysis phase that powers the Streamlit app.

### 1. Offline Phase: Training the "Brain"
This is the work done once to create our custom sentiment model.

**[CSV Icon] Flipkart Product Review Dataset** (from Kaggle)
&nbsp;&nbsp;&nbsp;&nbsp;**&darr;**
**1. Pre-process Data** (**Pandas**, **Scikit-learn**)
*(Loaded 200k+ reviews, dropped "Neutral" class to create a clear binary [Pos/Neg] task, calculated `class_weight` to fix the heavy "Positive" bias)*
&nbsp;&S;**&darr;**
**2. Tokenize Text** (**Keras Tokenizer**)
*(Converted all review text into integer sequences for the model)*
&nbsp;&nbsp;&nbsp;&nbsp;**&darr;**
**3. Train Deep Learning Model** (**TensorFlow / Keras**)
*(Fed the data into a **Bidirectional LSTM** model, which learned the sentiment patterns)*
&nbsp;&nbsp;&nbsp;&nbsp;**&darr;**
**[Brain Icon] Saved Files**
* `product_sentiment_BINARY_model.keras` (The 97.25% accurate model)
* `product_BINARY_tokenizer.json` (The word-to-index vocabulary)

### 2. Live Phase: The AspectLens App
This is what happens every time you click "Analyze" in the app.

**[User Icon] User Enters Review** (**Streamlit UI**)
&nbsp;&nbsp;&nbsp;&nbsp;**&darr;**
**1. Extract Aspects** (**spaCy**)
*(Parse the review, identify all nouns like "battery," "screen," "camera")*
&nbsp;&nbsp;&nbsp;&nbsp;**&darr;**
**2. Cluster Aspects** (**NLTK / WordNet**)
*(Group semantically similar aspects, e.g., "picture" and "camera," using `path_similarity`)*
&nbsp;&nbsp;&nbsp;&nbsp;**&darr;**
**3. Score Sentiment** (**Custom LSTM Model**)
*(Load the saved `.keras` model to predict the sentiment (0.0 to 1.0) for the sentence containing each aspect)*
&nbsp;&nbsp;&nbsp;&S;**&darr;**
**[Chart Icon] Display Final Report** (**Streamlit Table**)
*(Aggregate all scores, label them (Pos/Neg), and show the color-coded table)*

---

## 🛠️ Tech Stack & Key Concepts

This project integrates a full stack of modern data science and NLP tools:

* **Deep Learning:** **TensorFlow (Keras)** for building, training, and evaluating the Bidirectional LSTM.
* **Web Framework:** **Streamlit** for building the interactive web UI.
* **Linguistic Parsing:** **spaCy** for robust and efficient text preprocessing, part-of-speech tagging (to find nouns), and lemmatization.
* **Semantic Analysis:** **NLTK (WordNet)** for measuring semantic similarity between aspect nouns, allowing for intelligent clustering.
* **Data Handling:** **Pandas** for loading and cleaning the initial CSV dataset.
* **Model Balancing:** **Scikit-learn** for its `class_weight` utility, which was critical in fixing our imbalanced dataset and training an unbiased model.
* **Core Language:** **Python 3**

---

## ⚡ How to Run This Project Locally

You can run this app on your local machine by following these steps.

### 1. Clone the Repository
```
git clone https://github.com/AbhishekNangre/reviewanalyzer.git
cd reviewanalyzer
```


2. Set Up a Virtual Environment (Recommended)
Bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3. Install Dependencies
This project requires several packages. You can install them all using pip.

Bash

pip install streamlit tensorflow pandas scikit-learn spacy nltk
4. Download NLP Model Data
You only need to do this once. Run Python in your terminal and execute the following commands to download the necessary spaCy and NLTK models.

Python (run this py file to download)
  ```
import spacy
import nltk

# Download the spaCy model
spacy.cli.download("en_core_web_sm")

# Download the NLTK WordNet data
nltk.download("wordnet")
```

5. Run the Streamlit App
You're all set! From your terminal, run the following command:

```
streamlit run app.py
```
Your browser will automatically open, and you'll be able to use the app.
