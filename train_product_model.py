import pandas as pd
import numpy as np
import re
import json
import warnings
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings('ignore')
print("--- Step 1: Loading and Preprocessing BINARY Data ---")

try:
    df = pd.read_csv("Dataset-SA.csv", encoding='latin1')
except Exception:
    df = pd.read_csv("Dataset-SA.csv")

df = df[['Summary', 'Sentiment']].copy()
df = df.dropna()

# *** CHANGED: Remove the 'neutral' class ***
df = df[df['Sentiment'] != 'neutral']

# *** CHANGED: New binary label mapping ***
label_mapping = {'negative': 0, 'positive': 1}
df['label'] = df['Sentiment'].map(label_mapping)

print("Label Distribution:")
print(df['Sentiment'].value_counts())

vocab_size = 15000 
max_len = 150     
oov_token = "<OOV>" 

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(df['Summary'])

sequences = tokenizer.texts_to_sequences(df['Summary'])
padded_seq = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

print(f"\nData shape: {padded_seq.shape}")

print("\n--- Step 2: Splitting Data & Calculating Class Weights ---") 
X = padded_seq
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y 
)

# Re-calculate class weights for the 2 classes
class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)
class_weights_dict = dict(enumerate(class_weights))

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Class Weights: {class_weights_dict}") 

print("\n--- Step 3: Building the BINARY LSTM Model ---")

model = Sequential([
    Embedding(vocab_size, 64, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    # *** CHANGED: Output layer is 1 neuron with 'sigmoid' ***
    Dense(1, activation='sigmoid') 
])

# *** CHANGED: Loss is 'binary_crossentropy' ***
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print(model.summary())

print("\n--- Step 4: Training the Model (with Class Weights) ---") 

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

history = model.fit(
    X_train, y_train,
    epochs=10, 
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights_dict, 
    verbose=1
)

print("\n--- Step 5: Evaluating the Model ---")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

print("\n--- Step 6: Saving Model and Tokenizer ---")

# *** CHANGED: Save to a new model file ***
model.save('product_sentiment_BINARY_model.keras') 
print("Model saved as 'product_sentiment_BINARY_model.keras'")

tokenizer_json = tokenizer.to_json()
# *** CHANGED: Save to a new tokenizer file ***
with open('product_BINARY_tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))
print("Tokenizer saved as 'product_BINARY_tokenizer.json'")