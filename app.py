import streamlit as st
import pandas as pd
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
negations = {"not", "no", "nor", "never"}
stop_words = stop_words - negations


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


@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer


def analyze_review(review_text, model, vectorizer, profile_name=""):
    cleaned_review = clean_text(review_text)
    review_vector = vectorizer.transform([cleaned_review])

    ml_sentiment = model.predict(review_vector)[0]
    vader_result, vader_compound = vader_sentiment_with_score(review_text)

    final_sentiment = final_sentiment_decision(
        review_text,
        ml_sentiment,
        vader_result
    )

    emotion = detect_emotion(review_text, ml_sentiment, vader_result)

    priority = assign_priority(review_text, final_sentiment, vader_result)
    handling_type = assign_intervention(priority, review_text, final_sentiment)
    category = assign_category(review_text)
    gender = predict_gender(profile_name)

    return {
        "Original Review": review_text,
        "Profile Name": profile_name,
        "Predicted Gender": gender,
        "Cleaned Review": cleaned_review,
        "ML Sentiment": ml_sentiment,
        "VADER Sentiment": vader_result,
        "VADER Compound Score": round(vader_compound, 3),
        "Final Sentiment": final_sentiment,
        "Detected Emotion": emotion,
        "Priority": priority,
        "Handling Type": handling_type,
        "Complaint Category": category
    }


st.set_page_config(
    page_title="Customer Review Analysis",
    page_icon="📝",
    layout="wide"
)

st.title("Customer Review Analysis Using NLP and ML")

st.write(
    "This application analyzes customer reviews using machine learning, "
    "VADER sentiment analysis, thresholding, custom dictionaries, and rule-based logic."
)

st.info(
    "Gender prediction is used only as an optional aggregate insight based on the provided profile name. "
    "It is not used for priority, category, or human intervention decisions."
)

try:
    model, vectorizer = load_model()
except FileNotFoundError:
    st.error(
        "Model files were not found. First run projectCode.py to create "
        "sentiment_model.pkl and tfidf_vectorizer.pkl."
    )
    st.stop()


tab1, tab2, tab3 = st.tabs([
    "Single Review Analysis",
    "Batch Review Analysis",
    "Model Evaluation"
])


with tab1:
    st.subheader("Single Review Analysis")

    example_reviews = {
        "Neutral example": "The product arrived on time. It was okay, nothing special.",
        "Neutral mixed example": "The item was average, not bad but not great.",
        "Positive example": "The food was fresh and delicious. I would recommend it.",
        "Negative example": "The product was terrible and I am very disappointed.",
        "High priority complaint": "The package was damaged and I want a refund.",
        "Delivery issue": "The delivery was late and the package arrived damaged.",
        "Packaging issue": "The box was opened and the product was leaking.",
        "Wrong item issue": "I received the wrong item instead of what I ordered.",
        "Price value issue": "The product is overpriced and not worth the money."
    }

    selected_example = st.selectbox(
        "Choose an example review or write your own:",
        ["Write my own"] + list(example_reviews.keys())
    )

    default_review = ""

    if selected_example != "Write my own":
        default_review = example_reviews[selected_example]

    profile_name = st.text_input(
        "Profile name (optional):",
        placeholder="Example: Selma"
    )

    review = st.text_area(
        "Enter customer review:",
        value=default_review,
        height=150,
        placeholder="Example: The product arrived on time. It was okay, nothing special."
    )

    if st.button("Analyze Review"):
        if review.strip() == "":
            st.warning("Please enter a review first.")
        else:
            result = analyze_review(review, model, vectorizer, profile_name)

            st.subheader("Analysis Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("ML Sentiment", result["ML Sentiment"])
                st.metric("Final Sentiment", result["Final Sentiment"])
                st.metric("Predicted Gender", result["Predicted Gender"])

            with col2:
                st.metric("VADER Sentiment", result["VADER Sentiment"])
                st.metric("VADER Score", result["VADER Compound Score"])

            with col3:
                st.metric("Emotion", result["Detected Emotion"])
                st.metric("Priority", result["Priority"])

            st.metric("Handling Type", result["Handling Type"])
            st.metric("Complaint Category", result["Complaint Category"])

            st.subheader("Cleaned Review")
            st.write(result["Cleaned Review"])

            st.subheader("Full Result")
            st.dataframe(pd.DataFrame([result]))


with tab2:
    st.subheader("Batch Review Analysis")

    uploaded_file = st.file_uploader(
        "Upload a CSV file with customer reviews:",
        type=["csv"]
    )

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        st.subheader("Raw Sample Reviews")
        st.caption(
            "This table shows raw examples before preprocessing and classification."
        )

        sample_columns = []

        if "ProfileName" in batch_df.columns:
            sample_columns.append("ProfileName")

        if "Score" in batch_df.columns:
            sample_columns.append("Score")

        if "Text" in batch_df.columns:
            sample_columns.append("Text")

        if len(sample_columns) > 0:
            st.dataframe(batch_df[sample_columns].head(10))
        else:
            st.dataframe(batch_df.head(10))

        if "Text" not in batch_df.columns:
            st.error("The uploaded CSV file must contain a column named 'Text'.")
        else:
            if st.button("Analyze Uploaded Reviews"):
                results = []

                for _, row in batch_df.iterrows():
                    review_text = row.get("Text", "")
                    profile_name = row.get("ProfileName", "")
                    result = analyze_review(review_text, model, vectorizer, profile_name)
                    results.append(result)

                results_df = pd.DataFrame(results)

                st.subheader("Batch Analysis Results")
                st.dataframe(results_df)

                st.subheader("Result Summary")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("Final Sentiment Distribution")
                    st.bar_chart(results_df["Final Sentiment"].value_counts())

                    st.write("Priority Distribution")
                    st.bar_chart(results_df["Priority"].value_counts())

                    st.write("Gender Distribution")
                    st.bar_chart(results_df["Predicted Gender"].value_counts())

                with col2:
                    st.write("Emotion Distribution")
                    st.bar_chart(results_df["Detected Emotion"].value_counts())

                    st.write("Complaint Category Distribution")
                    st.bar_chart(results_df["Complaint Category"].value_counts())

                    st.write("Handling Type Distribution")
                    st.bar_chart(results_df["Handling Type"].value_counts())

                st.subheader("Gender-Based Aggregate Insight")

                gender_sentiment = pd.crosstab(
                    results_df["Predicted Gender"],
                    results_df["Final Sentiment"]
                )

                gender_priority = pd.crosstab(
                    results_df["Predicted Gender"],
                    results_df["Priority"]
                )

                st.write("Sentiment by Predicted Gender")
                st.dataframe(gender_sentiment)

                st.write("Priority by Predicted Gender")
                st.dataframe(gender_priority)

                csv_data = results_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Analysis Results as CSV",
                    data=csv_data,
                    file_name="customer_review_analysis_results.csv",
                    mime="text/csv"
                )


with tab3:
    st.subheader("Model Evaluation Results")

    st.write(
        "This section displays the saved evaluation outputs generated after training the model."
    )

    try:
        with open("classification_report.txt", "r") as file:
            report_text = file.read()

        st.text_area(
            "Classification Report",
            value=report_text,
            height=300
        )

    except FileNotFoundError:
        st.warning(
            "classification_report.txt was not found. Run projectCode.py first to generate it."
        )

    try:
        confusion_matrix_df = pd.read_csv("confusion_matrix.csv", index_col=0)

        st.write("Confusion Matrix")
        st.dataframe(confusion_matrix_df)

        st.write("Confusion Matrix Chart")
        st.bar_chart(confusion_matrix_df)

    except FileNotFoundError:
        st.warning(
            "confusion_matrix.csv was not found. Run projectCode.py first to generate it."
        )

    try:
        sample_predictions_df = pd.read_csv("sample_predictions.csv")

        st.write("Sample Predictions")
        st.dataframe(sample_predictions_df)

    except FileNotFoundError:
        st.warning(
            "sample_predictions.csv was not found. Run projectCode.py first to generate it."
        )

    try:
        gender_distribution_df = pd.read_csv("gender_distribution.csv")

        st.write("Gender Distribution from Training Dataset")
        st.dataframe(gender_distribution_df)
        st.bar_chart(
            gender_distribution_df.set_index("Predicted Gender")["Count"]
        )

    except FileNotFoundError:
        st.warning(
            "gender_distribution.csv was not found. Run projectCode.py first to generate it."
        )

    try:
        gender_sentiment_summary_df = pd.read_csv(
            "gender_sentiment_summary.csv",
            index_col=0
        )

        st.write("Training Dataset Sentiment by Predicted Gender")
        st.dataframe(gender_sentiment_summary_df)

    except FileNotFoundError:
        st.warning(
            "gender_sentiment_summary.csv was not found. Run projectCode.py first to generate it."
        )