# --- NLTK Setup for Streamlit ---
import nltk
import os
import ssl
import streamlit as st

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
nltk_resources = ['punkt', 'stopwords', 'wordnet', 'punkt_tab', 'vader_lexicon']
for resource in nltk_resources:
    try:
        nltk.data.find(resource)
    except LookupError:
        try:
            nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
        except:
            pass

# --- Imports ---
import re
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import io
from docx import Document
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import time

# --- Configuration ---
st.set_page_config(
    page_title="Chat with Documents",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
@st.cache_resource
def load_stopwords():
    try:
        return set(stopwords.words('english'))
    except:
        return set()

@st.cache_resource
def load_lemmatizer():
    return WordNetLemmatizer()

@st.cache_resource
def load_sentiment_analyzer():
    try:
        return SentimentIntensityAnalyzer()
    except:
        return None

# --- Text Processing Functions ---
def extract_text_from_file(file):
    """Extract text from uploaded file (PDF, TXT, or DOCX)"""
    try:
        if file.type == 'application/pdf':
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text = ""
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            doc.close()
            return text
        
        elif file.type == 'text/plain':
            return file.read().decode('utf-8')
        
        elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            doc = Document(io.BytesIO(file.read()))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        
        else:
            st.warning(f"Unsupported file type: {file.type}")
            return ""
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return ""

def preprocess_text(text):
    """Preprocess text for search and analysis"""
    lemmatizer = load_lemmatizer()
    stop_words = load_stopwords()
    
    # Clean and tokenize
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(w) for w in tokens 
              if w.isalnum() and w not in stop_words and len(w) > 2]
    
    return tokens

def chunk_text_advanced(text, chunk_size=200, overlap=50):
    """Create overlapping chunks from text"""
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        if current_length + len(words) > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Keep overlap
            overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_words + words
            current_length = len(current_chunk)
        else:
            current_chunk.extend(words)
            current_length += len(words)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def analyze_sentiment_advanced(text):
    """Analyze sentiment using multiple methods"""
    try:
        # NLTK VADER
        sia = load_sentiment_analyzer()
        if sia:
            vader_scores = sia.polarity_scores(text)
            vader_compound = vader_scores['compound']
        else:
            vader_compound = 0
        
        # TextBlob
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        
        # Combine scores
        combined_score = (vader_compound + textblob_polarity) / 2
        
        if combined_score > 0.1:
            return 'Positive', 'green', combined_score
        elif combined_score < -0.1:
            return 'Negative', 'red', combined_score
        else:
            return 'Neutral', 'orange', combined_score
    except:
        return 'Neutral', 'orange', 0

def generate_complete_summary(text, max_sentences=5):
    """Generate a comprehensive summary of the document"""
    try:
        # Split into sentences
        sentences = nltk.sent_tokenize(text)
        
        if len(sentences) <= max_sentences:
            return " ".join(sentences)
        
        # Simple extractive summarization using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate sentence scores
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        # Get top sentences
        top_indices = sentence_scores.argsort()[-max_sentences:][::-1]
        top_indices.sort()  # Keep original order
        
        summary = " ".join([sentences[i] for i in top_indices])
        return summary
    except:
        return "Unable to generate summary. Please check the document content."

def find_keyword_lines(text, keywords):
    """Find lines containing specific keywords"""
    lines = text.split('\n')
    keyword_lines = []
    
    for i, line in enumerate(lines, 1):
        for keyword in keywords:
            if keyword.lower() in line.lower():
                keyword_lines.append({
                    'line_number': i,
                    'line_text': line.strip(),
                    'keyword': keyword
                })
                break
    
    return keyword_lines

def highlight_keywords(text, keywords):
    """Highlight keywords in text"""
    highlighted_text = text
    for keyword in keywords:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        highlighted_text = pattern.sub(f"**{keyword.upper()}**", highlighted_text)
    
    return highlighted_text

# --- Search Functions ---
def search_with_tfidf(chunks, query, top_k=5):
    """Search using TF-IDF"""
    preprocessed_chunks = [" ".join(preprocess_text(chunk)) for chunk in chunks]
    query_processed = " ".join(preprocess_text(query))
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_chunks)
    query_vec = vectorizer.transform([query_processed])
    
    scores = (tfidf_matrix @ query_vec.T).toarray().flatten()
    top_indices = scores.argsort()[::-1][:top_k]
    
    return [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]

def search_with_bm25(chunks, query, top_k=5):
    """Search using BM25"""
    preprocessed_chunks = [preprocess_text(chunk) for chunk in chunks]
    query_tokens = preprocess_text(query)
    
    bm25 = BM25Okapi(preprocessed_chunks)
    scores = bm25.get_scores(query_tokens)
    
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    return [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]

# --- Visualization Functions ---
def create_sentiment_chart(sentiment_data):
    """Create sentiment analysis chart"""
    sentiment_counts = Counter([item[0] for item in sentiment_data])
    
    colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'orange'}
    
    fig = px.pie(
        values=list(sentiment_counts.values()),
        names=list(sentiment_counts.keys()),
        color=list(sentiment_counts.keys()),
        color_discrete_map=colors,
        title="Document Sentiment Distribution"
    )
    
    return fig

def create_keyword_frequency_chart(text, top_n=10):
    """Create keyword frequency chart"""
    tokens = preprocess_text(text)
    word_freq = Counter(tokens)
    
    top_words = dict(word_freq.most_common(top_n))
    
    fig = px.bar(
        x=list(top_words.keys()),
        y=list(top_words.values()),
        title=f"Top {top_n} Most Frequent Words",
        labels={'x': 'Words', 'y': 'Frequency'}
    )
    
    return fig

# --- Main Streamlit App ---
def main():
    st.title("📚 Enhanced Chat-with-Documents")
    st.markdown("Upload your document and interact with it using advanced search and analysis features!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Document", 
            type=["pdf", "txt", "docx"],
            help="Supported formats: PDF, TXT, DOCX"
        )
        
        if uploaded_file:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Processing options
            st.subheader("Processing Options")
            chunk_size = st.slider("Chunk Size", 100, 500, 200)
            overlap = st.slider("Chunk Overlap", 0, 100, 50)
            max_results = st.slider("Max Search Results", 3, 10, 5)
    
    # Main content
    if uploaded_file:
        # Extract text
        with st.spinner("Extracting text from document..."):
            raw_text = extract_text_from_file(uploaded_file)
        
        if not raw_text.strip():
            st.error("No text could be extracted from the document.")
            return
        
        # Create chunks
        chunks = chunk_text_advanced(raw_text, chunk_size, overlap)
        
        # Display document stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Characters", len(raw_text))
        with col2:
            st.metric("Total Words", len(raw_text.split()))
        with col3:
            st.metric("Total Chunks", len(chunks))
        with col4:
            st.metric("Avg Chunk Length", f"{np.mean([len(chunk.split()) for chunk in chunks]):.0f} words")
        
        # Document Summary Section
        st.header("📄 Complete Document Summary")
        with st.spinner("Generating comprehensive summary..."):
            summary = generate_complete_summary(raw_text, max_sentences=5)
            summary_sentiment, summary_color, summary_score = analyze_sentiment_advanced(summary)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text_area("Document Summary", summary, height=150)
        with col2:
            st.markdown(f"**Overall Sentiment:**")
            st.markdown(f"<span style='color:{summary_color}; font-weight:bold'>{summary_sentiment}</span>", unsafe_allow_html=True)
            st.markdown(f"**Score:** {summary_score:.3f}")
        
        # Sentiment Analysis Section
        st.header("😊 Sentiment Analysis")
        
        with st.spinner("Analyzing sentiment across document..."):
            chunk_sentiments = []
            for chunk in chunks:
                sentiment, color, score = analyze_sentiment_advanced(chunk)
                chunk_sentiments.append((sentiment, color, score))
        
        # Display sentiment chart
        col1, col2 = st.columns(2)
        with col1:
            sentiment_chart = create_sentiment_chart(chunk_sentiments)
            st.plotly_chart(sentiment_chart, use_container_width=True)
        
        with col2:
            # Sentiment statistics
            positive_count = sum(1 for s in chunk_sentiments if s[0] == 'Positive')
            negative_count = sum(1 for s in chunk_sentiments if s[0] == 'Negative')
            neutral_count = sum(1 for s in chunk_sentiments if s[0] == 'Neutral')
            
            st.subheader("Sentiment Statistics")
            st.write(f"🟢 Positive: {positive_count} chunks ({positive_count/len(chunks)*100:.1f}%)")
            st.write(f"🔴 Negative: {negative_count} chunks ({negative_count/len(chunks)*100:.1f}%)")
            st.write(f"🟠 Neutral: {neutral_count} chunks ({neutral_count/len(chunks)*100:.1f}%)")
        
        # Search Section
        st.header("🔍 Search & Chat")
        
        # Search input
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Ask a question or search for keywords:", placeholder="What is this document about?")
        with col2:
            search_method = st.selectbox("Search Method", ["TF-IDF", "BM25"])
        
        if query:
            # Perform search
            with st.spinner("Searching..."):
                if search_method == "TF-IDF":
                    results = search_with_tfidf(chunks, query, max_results)
                else:
                    results = search_with_bm25(chunks, query, max_results)
            
            if results:
                st.subheader(f"🎯 Top {len(results)} Results")
                
                for i, (chunk_idx, score) in enumerate(results, 1):
                    chunk_text = chunks[chunk_idx]
                    sentiment, color, sent_score = analyze_sentiment_advanced(chunk_text)
                    
                    # Highlight keywords
                    query_words = query.split()
                    highlighted_chunk = highlight_keywords(chunk_text, query_words)
                    
                    # Display result
                    with st.expander(f"Result {i} - Score: {score:.3f} | Sentiment: {sentiment}"):
                        st.markdown(f"**Sentiment:** <span style='color:{color}'>{sentiment}</span> ({sent_score:.3f})", unsafe_allow_html=True)
                        st.markdown(highlighted_chunk)
            else:
                st.warning("No relevant results found. Try different keywords.")
            
            # Find specific lines with keywords
            st.subheader("📍 Lines with Keywords")
            keyword_lines = find_keyword_lines(raw_text, query.split())
            
            if keyword_lines:
                for line_info in keyword_lines[:10]:  # Show max 10 lines
                    st.markdown(f"**Line {line_info['line_number']}:** {highlight_keywords(line_info['line_text'], [line_info['keyword']])}")
            else:
                st.info("No specific lines found with the searched keywords.")
        
        # Analytics Section
        st.header("📊 Document Analytics")
        
        # Word frequency chart
        freq_chart = create_keyword_frequency_chart(raw_text)
        st.plotly_chart(freq_chart, use_container_width=True)
        
        # Additional insights
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Reading Statistics")
            avg_reading_speed = 200  # words per minute
            total_words = len(raw_text.split())
            reading_time = total_words / avg_reading_speed
            st.write(f"**Estimated Reading Time:** {reading_time:.1f} minutes")
            st.write(f"**Average Sentence Length:** {np.mean([len(s.split()) for s in nltk.sent_tokenize(raw_text)]):.1f} words")
        
        with col2:
            st.subheader("🔤 Text Complexity")
            sentences = nltk.sent_tokenize(raw_text)
            words = raw_text.split()
            st.write(f"**Total Sentences:** {len(sentences)}")
            st.write(f"**Average Words per Sentence:** {len(words)/len(sentences):.1f}")
            st.write(f"**Vocabulary Richness:** {len(set(words))/len(words):.3f}")
    
    else:
        st.info("👆 Please upload a document to get started!")
        
        # Show app features
        st.header("🚀 App Features")
        
        features = [
            "📄 **Complete Document Summary** - Get comprehensive summaries of your documents",
            "😊 **Advanced Sentiment Analysis** - Analyze emotions with color-coded results (Green=Positive, Orange=Neutral, Red=Negative)",
            "🔍 **Intelligent Search** - Find relevant content using TF-IDF or BM25 algorithms",
            "📍 **Keyword Line Detection** - Locate specific lines containing your search terms",
            "📊 **Document Analytics** - Visualize word frequencies and document statistics",
            "🎯 **Multiple File Formats** - Support for PDF, TXT, and DOCX files",
            "⚙️ **Customizable Processing** - Adjust chunk sizes and search parameters"
        ]
        
        for feature in features:
            st.markdown(feature)

if __name__ == "__main__":
    main()