import streamlit as st
import nltk
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="AI Sentiment Intelligence",
    page_icon="🧠",
    layout="wide"
)

# ======================================
# NLTK (CACHED)
# ======================================
@st.cache_resource
def download_nltk():
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")

download_nltk()

# ======================================
# DATA + MODEL (CACHED, FAST)
# ======================================
@st.cache_resource
def load_everything():
    df = pd.read_csv("datanew.csv")

    # ---- SAFE IMPUTATION ----
    if "Up Votes" in df:
        df["Up Votes"] = SimpleImputer(strategy="mean").fit_transform(df[["Up Votes"]])
    if "Down Votes" in df:
        df["Down Votes"] = SimpleImputer(strategy="median").fit_transform(df[["Down Votes"]])
    if "Place of Review" in df:
        df["Place of Review"] = SimpleImputer(strategy="most_frequent") \
            .fit_transform(df[["Place of Review"]]).flatten()

    df.fillna("", inplace=True)

    # ---- SENTIMENT LABEL ----
    def infer_sentiment(r):
        if r >= 3.5:
            return 2   # Positive
        elif r <= 2.5:
            return 0   # Negative
        return 1       # Neutral

    df["Sentiment"] = df["Ratings"].apply(infer_sentiment)

    # ---- NLP ----
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def preprocess(text):
        text = re.sub(r"[^\w\s]", "", text.lower())
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
        return " ".join(tokens)

    df["Processed Review"] = df["Review text"].apply(preprocess)

    # ---- VECTORIZATION ----
    vectorizer = TfidfVectorizer(max_features=1200)
    X = vectorizer.fit_transform(df["Processed Review"])
    y = df["Sentiment"]

    # ---- MODEL ----
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    preds = model.predict(X)
    f1 = f1_score(y, preds, average="weighted")

    return df, model, vectorizer, preprocess, f1

df, model, vectorizer, preprocess, f1_score_value = load_everything()

# ======================================
# HEADER
# ======================================
st.markdown("""
# 🧠 AI Sentiment Intelligence Platform  
## By: Zainul Abedeen
### Fast • Stable • Enterprise-Grade NLP Dashboard
""")
st.markdown("---")

# ======================================
# TABS
# ======================================
tab1, tab2, tab3 = st.tabs([
    "🔮 Prediction Engine",
    "📊 Model Insights",
    "🗃 Dataset Explorer"
])

# ======================================
# TAB 1 — PREDICTION + BATCH
# ======================================
with tab1:
    col1, col2 = st.columns([2, 1])

    # ---------- SINGLE PREDICTION ----------
    with col1:
        review = st.text_area(
            "Enter Customer Review",
            height=160,
            placeholder="Type or paste a customer review..."
        )

        if st.button("🚀 Analyze Sentiment", use_container_width=True) and review.strip():
            cleaned = preprocess(review)
            vect = vectorizer.transform([cleaned])

            pred = model.predict(vect)[0]
            probs = model.predict_proba(vect)[0]

            label_map = {2: "Positive 😊", 1: "Neutral 😐", 0: "Negative ☹️"}
            st.success(f"### Predicted Sentiment: **{label_map[pred]}**")

            st.subheader("🔍 Prediction Confidence")
            st.progress(float(np.max(probs)))

            st.write({
                "Negative": round(float(probs[0]), 3),
                "Neutral": round(float(probs[1]), 3),
                "Positive": round(float(probs[2]), 3)
            })

            # ---- PER-SENTENCE (BIG & READABLE) ----
            st.subheader("🧠 Per-Sentence Sentiment")
            for s in re.split(r"[.!?]", review):
                if s.strip():
                    p = model.predict(vectorizer.transform([preprocess(s)]))[0]
                    color = ["#ff4d4d", "#ffaa00", "#00e676"][p]
                    st.markdown(
                        f"<div style='font-size:20px;padding:10px;"
                        f"border-radius:12px;background:{color};"
                        f"color:black;margin-bottom:6px'>{s.strip()}</div>",
                        unsafe_allow_html=True
                    )

            with st.expander("🧪 NLP Preprocessing Pipeline"):
                st.code(cleaned)

    # ---------- METRICS ----------
    with col2:
        st.metric("📈 Training F1 Score", round(f1_score_value, 4))
        st.metric("📚 Total Reviews", len(df))
        st.metric("🧠 Vocabulary Size", len(vectorizer.vocabulary_))

    st.markdown("---")

    # ---------- BATCH PREDICTION ----------
    st.subheader("📊 Batch Prediction (CSV Upload)")
    uploaded = st.file_uploader("Upload CSV with 'Review text' column", type=["csv"])

    if uploaded:
        batch = pd.read_csv(uploaded)
        batch["Processed"] = batch["Review text"].fillna("").apply(preprocess)
        Xb = vectorizer.transform(batch["Processed"])
        batch["Predicted Sentiment"] = model.predict(Xb)

        label_map = {2: "Positive", 1: "Neutral", 0: "Negative"}
        batch["Predicted Sentiment"] = batch["Predicted Sentiment"].map(label_map)

        st.dataframe(batch, use_container_width=True)

        st.download_button(
            "⬇ Download Predictions",
            batch.to_csv(index=False),
            "batch_sentiment_predictions.csv"
        )

# ======================================
# TAB 2 — MODEL INSIGHTS
# ======================================
with tab2:
    st.subheader("🌲 Random Forest — Top Influential Words")

    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]
    features = np.array(vectorizer.get_feature_names_out())[indices]

    fig, ax = plt.subplots()
    ax.barh(features, importances[indices])
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 Words Driving Sentiment")
    st.pyplot(fig)

    st.subheader("📊 Sentiment Distribution")
    st.bar_chart(df["Sentiment"].value_counts())

# ======================================
# TAB 3 — DATASET EXPLORER
# ======================================
with tab3:
    st.subheader("🔎 Dataset Preview")
    st.dataframe(df.head(300), use_container_width=True)

    st.subheader("📈 Ratings vs Sentiment")
    st.scatter_chart(df[["Ratings", "Sentiment"]])


