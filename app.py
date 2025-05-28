# test change for git
import nltk

def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
        except LookupError:
            nltk.download(resource)

download_nltk_resources()

import os
import re
import fitz  # PyMuPDF
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.metrics import precision_score, recall_score

@st.cache_data
def extract_text(file):
    if file.type == 'application/pdf':
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])
        return text
    elif file.type == 'text/plain':
        return file.read().decode('utf-8')
    return ""

def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalnum() and w not in stop_words]
    return tokens

def chunk_text(text, chunk_size=100):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current = []
    for sent in sentences:
        current.extend(nltk.word_tokenize(sent))
        if len(current) >= chunk_size:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks

def analyze_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0.1:
        return 'Positive', 'green'
    elif sentiment < -0.1:
        return 'Negative', 'red'
    else:
        return 'Neutral', 'gray'

def summarize_text(text):
    try:
        blob = TextBlob(text)
        return " ".join(blob.sentences[:3])
    except:
        return "Summary Unavailable"

st.title("📝 Chat-with-Documents: Classical IR Edition")

uploaded_file = st.file_uploader("Upload PDF or Text File", type=["pdf", "txt"])
if uploaded_file:
    raw_text = extract_text(uploaded_file)
    chunks = chunk_text(raw_text)
    preprocessed_chunks = [" ".join(preprocess(chunk)) for chunk in chunks]

    st.write(f"**Total Chunks:** {len(chunks)}")

    query = st.text_input("Ask a question:")
    method = st.radio("Choose Retrieval Method:", ["TF-IDF", "BM25"])

    if query:
        query_tokens = preprocess(query)

        # TF-IDF
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_chunks)
        query_vec = tfidf_vectorizer.transform([" ".join(query_tokens)])
        tfidf_scores = (tfidf_matrix @ query_vec.T).toarray().flatten()

        # BM25
        bm25 = BM25Okapi([chunk.split() for chunk in preprocessed_chunks])
        bm25_scores = bm25.get_scores(query_tokens)

        # Ranking
        if method == "TF-IDF":
            top_indices = tfidf_scores.argsort()[::-1][:5]
            scores = tfidf_scores
        else:
            top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:5]
            scores = bm25_scores

        st.subheader("🔍 Top Results")
        for idx in top_indices:
            text_chunk = chunks[idx]
            score = scores[idx]
            sentiment, color = analyze_sentiment(text_chunk)
            highlighted = re.sub(rf"({'|'.join(query_tokens)})", r"**\1**", text_chunk, flags=re.IGNORECASE)
            st.markdown(f"<span style='color:{color}'><b>[{sentiment}]</b></span> ({round(score, 3)}):<br>{highlighted}", unsafe_allow_html=True)

        st.subheader("🧾 Document Summary")
        summary = summarize_text(raw_text)
        st.info(summary)

        st.subheader("⚖️ TF-IDF vs BM25 Scores")
        compare_df = pd.DataFrame({
            "Chunk": [chunks[i][:50] + '...' for i in range(len(chunks))],
            "TF-IDF": tfidf_scores,
            "BM25": bm25_scores
        }).sort_values(by="TF-IDF", ascending=False)
        st.dataframe(compare_df.head(10))
