#Importing and downloading all the necessary libraries that are needed for the project
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
# remove negations from stopwords because negations are important part of the review
negations = {"not", "no", "nor", "never"}
stop_words = stop_words - negations

#First step is loading the dataset
file_path = "DatasetFULL.csv"
df = pd.read_csv(file_path, encoding="latin-1")

#checking whether the dataset is correctly imported and looking at its columns in order to choose relevant ones
print("Dataset loaded!")
print(df.head())
print("\nColumns:")
print(df.columns)

#choosing the important columns that we might use in future
useful_columns = [
    "ProductId",
    "ProfileName",
    "HelpfulnessNumerator",
    "HelpfulnessDenominator",
    "Score",
    "Summary",
    "Text"
]
df = df[useful_columns].copy()

# Fill missing text fields
df["Summary"] = df["Summary"].fillna("")
df["Text"] = df["Text"].fillna("")
# Remove rows with missing score
df.dropna(subset=["Score"], inplace=True)
print("\nAfter selecting useful columns:")
print(df.head())

# Creating sentiment labels
def score_to_sentiment(score):
    if score in [1, 2]:
        return "negative"
    elif score == 3:
        return "neutral"
    else:
        return "positive"
df["sentiment"] = df["Score"].apply(score_to_sentiment)
print("\nSentiment label examples:")
print(df[["Score", "sentiment"]].head(10))
print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

# Combining text and summary for better model learning and cleaning the text in order to remove unnecessary characters
# Cleaning is done in order to avoid overwhelming the model with some data that is not relevant for its purpose
df["full_text"] = df["Summary"] + " " + df["Text"]
def clean_text(text):
    text = str(text).lower()
    # Transforming negations like don't become do not
    text = re.sub(r"n't", " not", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Replace punctuation with spaces
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    # Tokenize and remove stopwords except negations
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)
df["clean_text"] = df["full_text"].apply(clean_text)
print("\nCleaned text examples:")
print(df[["full_text", "clean_text"]].head())
# Now we come to the training part where we will use 20% of the dataset for testing and other 80% for learning
X = df["clean_text"]
y = df["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain size:", len(X_train))
print("Test size:", len(X_test))

# Converting textual data into numerical vectors so that machine learning model can process it
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF shapes:")
print("Train:", X_train_tfidf.shape)
print("Test :", X_test_tfidf.shape)

# Training a Logistic Regression model on TF-IDF features
# class_weight is used to handle class imbalance
model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)
model.fit(X_train_tfidf, y_train)

# Generating predictions for the test dataset
y_pred = model.predict(X_test_tfidf)

# Initializing VADER sentiment analyzer as a rule-based baseline method
# Used for comparison with the machine learning model
analyzer = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]

    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

# Evaluating model performance using accuracy, precision, recall and confusion matrix
# This provides insight into how well the model performs on different sentiment classes

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Saving trained model and vectorizer for future use
# This allows the system to make predictions without retraining
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully.")

# Function that processes a new customer review and predicts sentiment
# Includes text cleaning and transformation using the trained vectorizer
def predict_sentiment(new_review):
    cleaned_review = clean_text(new_review)
    review_vector = vectorizer.transform([cleaned_review])
    prediction = model.predict(review_vector)[0]
    return prediction, cleaned_review


# Assigning complaint priority based on sentiment and presence of strong complaint keywords
# High priority indicates serious issues that require immediate attention
priority_keywords = {
    "refund", "broken", "worst", "awful", "terrible", "disappointed",
    "horrible", "waste", "never", "not worth", "bad", "poor", "stale",
    "damaged", "disgusting", "returned", "complaint"
}

def assign_priority(text, ml_sentiment, vader_sent):
    text_lower = text.lower()

    has_strong_complaint = any(keyword in text_lower for keyword in priority_keywords)

    if ml_sentiment == "negative" and has_strong_complaint:
        return "high"
    elif ml_sentiment == "negative" or vader_sent == "negative":
        return "medium"
    else:
        return "low"

# Determining whether the issue should be handled by a human agent or automated system
# Based on priority and presence of critical issue indicators
def assign_intervention(priority, text, sentiment):
    text_lower = text.lower()

    critical_keywords = {
        "refund", "broken", "wrong", "damaged", "complaint",
        "never again", "return", "cancel", "not working"
    }

    has_critical_issue = any(word in text_lower for word in critical_keywords)

    if priority == "high" or has_critical_issue:
        return "agent"
    elif sentiment == "negative":
        return "agent"
    else:
        return "automated"

# Classifying the complaint into predefined categories using keyword matching
# This helps in understanding the type of customer issue
category_keywords = {
    "taste_quality": {
        "taste", "flavor", "smell", "stale", "spoiled", "rotten",
        "awful", "disgusting", "bland", "salty", "sweet", "bitter"
    },
    "delivery_shipping": {
        "delivery", "shipping", "late", "slow", "arrived", "courier",
        "delay", "delayed"
    },
    "packaging": {
        "package", "packaging", "bag", "box", "damaged", "broken seal",
        "opened", "leaking", "holes", "stains"
    },
    "wrong_item": {
        "wrong", "different", "ordered", "received", "instead",
        "another item", "something else"
    },
    "price_value": {
        "price", "expensive", "cheap", "worth", "value", "cost",
        "overpriced"
    },
    "customer_service": {
        "service", "support", "seller", "response", "refund",
        "return", "cancel", "complaint"
    }
}

def assign_category(text):
    text_lower = text.lower()

    category_scores = {}

    for category, keywords in category_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        category_scores[category] = score

    best_category = max(category_scores, key=category_scores.get)

    if category_scores[best_category] == 0:
        return "general"

    return best_category

# Interactive part where user can input a custom review
# System outputs sentiment, priority, handling type and category
user_review = input("\nEnter a new customer review for sentiment prediction:\n")

predicted_sentiment, cleaned_input = predict_sentiment(user_review)

print("\n--- PREDICTION RESULT ---")
print("Original review:", user_review)
print("Cleaned review :", cleaned_input)
print("Predicted sentiment:", predicted_sentiment)
vader_result = vader_sentiment(user_review)
print("VADER sentiment:", vader_result)
predicted_priority = assign_priority(user_review, predicted_sentiment, vader_result)
print("Predicted priority:", predicted_priority)
predicted_intervention = assign_intervention(
    predicted_priority,
    user_review,
    predicted_sentiment
)
print("Handling type:", predicted_intervention)
predicted_category = assign_category(user_review)
print("Complaint category:", predicted_category)