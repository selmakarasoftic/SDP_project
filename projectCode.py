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
negations = {"not", "no", "nor", "never"}
stop_words = stop_words - negations


file_path = "DatasetFULL.csv"
df = pd.read_csv(file_path, encoding="latin-1")

print("Dataset loaded!")
print(df.head())

print("\nColumns:")
print(df.columns)

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

df["ProfileName"] = df["ProfileName"].fillna("")
df["Summary"] = df["Summary"].fillna("")
df["Text"] = df["Text"].fillna("")
df.dropna(subset=["Score"], inplace=True)

print("\nAfter selecting useful columns:")
print(df.head())

raw_sample_reviews = df[["ProfileName", "Score", "Summary", "Text"]].head(10)
raw_sample_reviews.to_csv("raw_sample_reviews.csv", index=False)

print("\nRaw sample reviews saved successfully:")
print("- raw_sample_reviews.csv")
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


female_names = {
    "mary", "patricia", "jennifer", "linda", "elizabeth", "barbara", "susan",
    "jessica", "sarah", "karen", "nancy", "lisa", "betty", "margaret",
    "sandra", "ashley", "kimberly", "emily", "donna", "michelle", "carol",
    "amanda", "melissa", "deborah", "stephanie", "rebecca", "laura",
    "sharon", "cynthia", "kathleen", "amy", "angela", "helen", "anna",
    "brenda", "pamela", "nicole", "samantha", "katherine", "emma",
    "olivia", "sophia", "isabella", "mia", "charlotte", "amelia",
    "selma", "amina", "lejla", "lamija", "ajla", "sara", "hana",
    "nejra", "dzejla", "merima", "azra", "maja", "ivana", "ana"
}

male_names = {
    "james", "john", "robert", "michael", "william", "david", "richard",
    "joseph", "thomas", "christopher", "charles", "daniel", "matthew",
    "anthony", "mark", "donald", "steven", "paul", "andrew", "joshua",
    "kenneth", "kevin", "brian", "george", "edward", "ronald", "timothy",
    "jason", "jeffrey", "ryan", "jacob", "gary", "nicholas", "eric",
    "jonathan", "stephen", "larry", "justin", "scott", "brandon",
    "benjamin", "samuel", "omar", "amer", "tarik", "edin", "emir",
    "adnan", "haris", "mirza", "kenan", "faris", "amar", "malik",
    "nedim", "dino", "alen", "amir", "enes"
}


def predict_gender(profile_name):
    name = str(profile_name).lower().strip()

    if name == "" or name in ["nan", "none", "unknown"]:
        return "unknown"

    name = re.sub(f"[{re.escape(string.punctuation)}]", " ", name)
    name = re.sub(r"\d+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    if name == "":
        return "unknown"

    first_name = name.split()[0]

    if first_name in female_names:
        return "female"
    elif first_name in male_names:
        return "male"
    else:
        return "unknown"


df["PredictedGender"] = df["ProfileName"].apply(predict_gender)

print("\nPredicted gender distribution:")
print(df["PredictedGender"].value_counts())

df["full_text"] = df["Summary"] + " " + df["Text"]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)


df["clean_text"] = df["full_text"].apply(clean_text)

print("\nCleaned text examples:")
print(df[["full_text", "clean_text"]].head())

cleaned_sample_reviews = df[["full_text", "clean_text"]].head(10)
cleaned_sample_reviews.to_csv("cleaned_sample_reviews.csv", index=False)

print("\nCleaned sample reviews saved successfully:")
print("- cleaned_sample_reviews.csv")
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

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

analyzer = SentimentIntensityAnalyzer()

custom_vader_dictionary = {
    "refund": -2.8,
    "broken": -3.0,
    "damaged": -2.7,
    "spoiled": -3.0,
    "stale": -2.4,
    "rotten": -3.0,
    "disgusting": -3.2,
    "awful": -3.0,
    "terrible": -3.1,
    "horrible": -3.1,
    "waste": -2.5,
    "worst": -3.2,
    "complaint": -2.0,
    "late": -1.6,
    "delayed": -1.8,
    "leaking": -2.2,
    "overpriced": -1.8,

    "fresh": 2.0,
    "tasty": 2.1,
    "delicious": 2.8,
    "excellent": 3.0,
    "perfect": 2.7,
    "recommend": 2.0,
    "satisfied": 2.3,

    "average": 0.0,
    "okay": 0.0,
    "ok": 0.0,
    "fine": 0.0,
    "normal": 0.0,
    "standard": 0.0,
    "expected": 0.0,
    "arrived": 0.0,
    "received": 0.0,
    "product": 0.0,
    "item": 0.0,
    "package": 0.0,
    "packaging": 0.0,
    "delivery": 0.0,
    "shipping": 0.0
}

analyzer.lexicon.update(custom_vader_dictionary)

neutral_phrases = {
    "nothing special",
    "as expected",
    "product arrived",
    "received the product",
    "it was okay",
    "it is okay",
    "it was fine",
    "it is fine",
    "not bad",
    "not great",
    "not bad but not great",
    "average product",
    "normal product",
    "standard product",
    "neither good nor bad"
}

VADER_POSITIVE_THRESHOLD = 0.10
VADER_NEGATIVE_THRESHOLD = -0.10


def contains_phrase(text, phrase_set):
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in phrase_set)


def vader_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound_score = scores["compound"]
    text_lower = text.lower()

    if "not bad but not great" in text_lower:
        return "neutral"

    has_neutral_phrase = contains_phrase(text, neutral_phrases)

    if has_neutral_phrase and -0.40 < compound_score < 0.40:
        return "neutral"

    if compound_score >= VADER_POSITIVE_THRESHOLD:
        return "positive"
    elif compound_score <= VADER_NEGATIVE_THRESHOLD:
        return "negative"
    else:
        return "neutral"


def vader_sentiment_with_score(text):
    scores = analyzer.polarity_scores(text)
    sentiment = vader_sentiment(text)
    return sentiment, scores["compound"]


emotion_keywords = {
    "anger": {
        "angry", "mad", "furious", "annoyed", "upset", "irritated",
        "terrible", "awful", "horrible", "worst", "disgusting"
    },
    "frustration": {
        "frustrated", "disappointed", "waste", "not working", "broken",
        "damaged", "late", "delayed", "refund", "complaint", "never again"
    },
    "satisfaction": {
        "happy", "satisfied", "great", "excellent", "perfect", "fresh",
        "delicious", "tasty", "love", "recommend", "good"
    },
    "neutral": {
        "okay", "ok", "fine", "average", "normal", "standard",
        "as expected", "nothing special", "not bad but not great"
    }
}


def detect_emotion(text, ml_sentiment, vader_sent):
    text_lower = text.lower()

    if contains_phrase(text_lower, neutral_phrases):
        return "neutral"

    emotion_scores = {}

    for emotion, keywords in emotion_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        emotion_scores[emotion] = score

    best_emotion = max(emotion_scores, key=emotion_scores.get)

    if emotion_scores[best_emotion] > 0:
        return best_emotion

    if ml_sentiment == "negative" or vader_sent == "negative":
        return "frustration"
    elif ml_sentiment == "positive" or vader_sent == "positive":
        return "satisfaction"
    else:
        return "neutral"


def final_sentiment_decision(text, ml_sentiment, vader_sentiment_result):
    text_lower = text.lower()

    if "not bad but not great" in text_lower:
        return "neutral"

    has_neutral_phrase = contains_phrase(text_lower, neutral_phrases)

    strong_negative_terms = {
        "refund", "broken", "damaged", "spoiled", "stale", "rotten",
        "disgusting", "awful", "terrible", "horrible", "worst",
        "complaint", "not worth", "never again"
    }

    strong_positive_terms = {
        "excellent", "perfect", "delicious", "fresh", "recommend",
        "satisfied", "love", "great"
    }

    has_strong_negative = any(term in text_lower for term in strong_negative_terms)
    has_strong_positive = any(term in text_lower for term in strong_positive_terms)

    if has_neutral_phrase and not has_strong_negative and not has_strong_positive:
        return "neutral"

    if ml_sentiment == vader_sentiment_result:
        return ml_sentiment

    if vader_sentiment_result == "neutral":
        return ml_sentiment

    if ml_sentiment == "neutral":
        return vader_sentiment_result

    return ml_sentiment


priority_keywords = {
    "refund", "broken", "worst", "awful", "terrible", "disappointed",
    "horrible", "waste", "never", "not worth", "bad", "poor", "stale",
    "damaged", "disgusting", "returned", "complaint"
}


def assign_priority(text, final_sentiment, vader_sent):
    text_lower = text.lower()
    has_strong_complaint = any(keyword in text_lower for keyword in priority_keywords)

    if final_sentiment == "negative" and has_strong_complaint:
        return "high"
    elif final_sentiment == "negative" or vader_sent == "negative":
        return "medium"
    else:
        return "low"


def assign_intervention(priority, text, final_sentiment):
    text_lower = text.lower()

    critical_keywords = {
        "refund", "broken", "wrong", "damaged", "complaint",
        "never again", "return", "cancel", "not working"
    }

    has_critical_issue = any(word in text_lower for word in critical_keywords)

    if priority == "high" or has_critical_issue:
        return "agent"
    elif final_sentiment == "negative":
        return "agent"
    else:
        return "automated"


category_keywords = {
    "taste_quality": {
        "taste", "flavor", "smell", "stale", "spoiled", "rotten",
        "awful", "disgusting", "bland", "salty", "sweet", "bitter", "fresh"
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
        "another item", "something else", "wrong item"
    },
    "price_value": {
        "price", "expensive", "cheap", "worth", "value", "cost",
        "overpriced", "not worth"
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


def predict_sentiment(new_review):
    cleaned_review = clean_text(new_review)
    review_vector = vectorizer.transform([cleaned_review])
    prediction = model.predict(review_vector)[0]
    return prediction, cleaned_review


def analyze_review(review_text, profile_name=""):
    predicted_sentiment, cleaned_input = predict_sentiment(review_text)

    vader_result, vader_compound = vader_sentiment_with_score(review_text)

    final_sentiment = final_sentiment_decision(
        review_text,
        predicted_sentiment,
        vader_result
    )

    predicted_emotion = detect_emotion(
        review_text,
        predicted_sentiment,
        vader_result
    )

    predicted_priority = assign_priority(
        review_text,
        final_sentiment,
        vader_result
    )

    predicted_intervention = assign_intervention(
        predicted_priority,
        review_text,
        final_sentiment
    )

    predicted_category = assign_category(review_text)
    predicted_gender = predict_gender(profile_name)

    return {
        "Original Review": review_text,
        "Profile Name": profile_name,
        "Predicted Gender": predicted_gender,
        "Cleaned Review": cleaned_input,
        "ML Sentiment": predicted_sentiment,
        "VADER Sentiment": vader_result,
        "VADER Compound Score": round(vader_compound, 3),
        "Final Sentiment": final_sentiment,
        "Detected Emotion": predicted_emotion,
        "Priority": predicted_priority,
        "Handling Type": predicted_intervention,
        "Complaint Category": predicted_category
    }


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
print(matrix)

with open("classification_report.txt", "w") as file:
    file.write("MODEL EVALUATION RESULTS\n")
    file.write("========================\n\n")
    file.write(f"Accuracy: {accuracy}\n\n")
    file.write("Classification Report:\n")
    file.write(report)

labels = ["negative", "neutral", "positive"]

confusion_df = pd.DataFrame(
    matrix,
    index=[f"actual_{label}" for label in labels],
    columns=[f"predicted_{label}" for label in labels]
)

confusion_df.to_csv("confusion_matrix.csv")

sample_results = pd.DataFrame({
    "Review": X_test.values[:50],
    "Actual Sentiment": y_test.values[:50],
    "Predicted Sentiment": y_pred[:50]
})

sample_results.to_csv("sample_predictions.csv", index=False)

gender_distribution = df["PredictedGender"].value_counts().reset_index()
gender_distribution.columns = ["Predicted Gender", "Count"]
gender_distribution.to_csv("gender_distribution.csv", index=False)

gender_sentiment_summary = pd.crosstab(
    df["PredictedGender"],
    df["sentiment"]
)

gender_sentiment_summary.to_csv("gender_sentiment_summary.csv")

print("\nEvaluation files saved successfully:")
print("- classification_report.txt")
print("- confusion_matrix.csv")
print("- sample_predictions.csv")
print("- gender_distribution.csv")
print("- gender_sentiment_summary.csv")

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully.")


#user_review = input("\nEnter a new customer review for sentiment prediction:\n")
#user_profile_name = input("Enter profile name if available, or press Enter to skip:\n")
#result = analyze_review(user_review, user_profile_name)
#print("\n--- PREDICTION RESULT ---")
#print("Original review:", result["Original Review"])
#print("Profile name:", result["Profile Name"])
#print("Predicted gender:", result["Predicted Gender"])
#print("Cleaned review :", result["Cleaned Review"])
#print("ML predicted sentiment:", result["ML Sentiment"])
#print("VADER sentiment:", result["VADER Sentiment"])
#print("VADER compound score:", result["VADER Compound Score"])
#print("Final sentiment decision:", result["Final Sentiment"])
#print("Detected emotion/tone:", result["Detected Emotion"])
#print("Predicted priority:", result["Priority"])
#print("Handling type:", result["Handling Type"])
#print("Complaint category:", result["Complaint Category"])