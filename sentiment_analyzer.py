# sentiment_analyzer.py

import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download NLTK resources
nltk.download('stopwords')

# Load dataset
with open("IMDB Dataset.csv", encoding="utf-8", errors="replace") as f:
    df = pd.read_csv(f)
# or encoding='ISO-8859-1'
# Ensure your CSV file has 'review' and 'sentiment' columns
df = df[['review', 'sentiment']].dropna()

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    stop_words = stopwords.words('english')
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df['clean_review'] = df['review'].apply(clean_text)

# Label encoding
df['label'] = df['sentiment'].apply(lambda x: 1 if x == "positive" else 0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['clean_review'], df['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Model Evaluation
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

# Sample Predictions
sample_size = min(5, len(X_test))
sample_reviews = X_test.sample(sample_size, random_state=1)
sample_preds = model.predict(tfidf.transform(sample_reviews))

for review, pred in zip(sample_reviews, sample_preds):
    print(f"Review: \"{review[:80]}...\"\nPrediction: {'POSITIVE' if pred == 1 else 'NEGATIVE'}\n")

# Test custom input
while True:
    user_input = input("Enter a movie review (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    cleaned = clean_text(user_input)
    prediction = model.predict(tfidf.transform([cleaned]))[0]
    print(f"Prediction: {'POSITIVE' if prediction == 1 else 'NEGATIVE'}\n")
