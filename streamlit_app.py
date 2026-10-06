import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download NLTK resources automatically
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Page Configuration
st.set_page_config(page_title="TalentMatch - CV Matcher", layout="wide", page_icon="📄")

st.title("📄 TalentMatch - Resume & Job Description Matcher")
st.write("Enter the Job Description and Resume text below to calculate the match percentage.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Job Description")
    job_desc = st.text_area("Paste Job Description here:", height=220, placeholder="e.g., Looking for a Data Engineer skilled in Python, SQL...")

with col2:
    st.subheader("📝 Resume / CV Content")
    cv_text = st.text_area("Paste Resume content here:", height=220, placeholder="e.g., 3 years of experience in data analysis using Python...")

@st.cache_resource
def load_vectorizer():
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        from sklearn.feature_extraction.text import TfidfVectorizer
        return TfidfVectorizer()

vectorizer = load_vectorizer()

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Calculate Match Rate 🚀", use_container_width=True):
    if job_desc.strip() and cv_text.strip():
        tfidf_matrix = vectorizer.fit_transform([job_desc, cv_text])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100

        st.markdown("---")
        st.subheader("📊 Matching Analysis Results:")

        if score >= 70:
            st.success(f"🎯 Match Rate: {score:.2f}% (Excellent Match)")
        elif score >= 40:
            st.info(f"⚡ Match Rate: {score:.2f}% (Moderate Match)")
        else:
            st.warning(f"⚠️ Match Rate: {score:.2f}% (Low Match)")

        st.progress(int(score))
    else:
        st.error("Please enter both Job Description and Resume content first!")
