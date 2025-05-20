# Task 4: Sentiment Analysis using NLP

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download NLTK data
nltk.download("stopwords")
nltk.download("wordnet")

# Load dataset
df = pd.read_csv("amazon_reviews.csv")  # Assume the file contains 'Review' and 'Sentiment'

# Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df["Cleaned_Review"] = df["Review"].apply(clean_text)

# Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["Cleaned_Review"])
y = df["Sentiment"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluation
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
