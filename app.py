# --- NLTK Setup for Streamlit ---
import nltk
nltk.download('punkt_tab')
import os
import ssl

# Handle SSL certificate issues for NLTK in hosted environments
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Set up persistent NLTK data directory
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download required NLTK data if not present
for resource in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_dir)

# --- Imports ---
import re
import fitz  # PyMuPDF
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# --- Functions ---
@st.cache_resource
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
        return " ".join(str(s) for s in blob.sentences[:3])
    except:
        return "Summary Unavailable"

# --- Streamlit App ---
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
            # Escape special characters in query_tokens for safe regex
            escaped_tokens = [re.escape(token) for token in query_tokens]
            highlighted = re.sub(rf"({'|'.join(escaped_tokens)})", r"**\1**", text_chunk, flags=re.IGNORECASE)
            st.markdown(
                f"<span style='color:{color}'><b>[{sentiment}]</b></span> ({round(score, 3)}):<br>{highlighted}",
                unsafe_allow_html=True
            )

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
