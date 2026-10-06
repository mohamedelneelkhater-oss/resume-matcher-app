import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from pypdf import PdfReader
import docx

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
st.write("Enter the Job Description and upload or paste the Resume to calculate the match percentage.")

st.markdown("---")

# Function to extract text from PDF/DOCX
def extract_text_from_file(file):
    text = ""
    if file.name.endswith('.pdf'):
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    elif file.name.endswith('.docx'):
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file.name.endswith('.txt'):
        text = str(file.read(), 'utf-8')
    return text

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Job Description")
    job_desc = st.text_area("Paste Job Description here:", height=260, placeholder="e.g., Looking for a Data Engineer skilled in Python, SQL...")

with col2:
    st.subheader("📝 Resume / CV Content")
    
    # Upload file directly from Desktop
    uploaded_file = st.file_uploader("Upload CV from Desktop (PDF, DOCX, TXT):", type=['pdf', 'docx', 'txt'])
    
    uploaded_text = ""
    if uploaded_file is not None:
        try:
            uploaded_text = extract_text_from_file(uploaded_file)
            st.success("✅ File uploaded and text extracted successfully!")
        except Exception as e:
            st.error("Error reading file. Please paste text manually or try another file.")

    cv_text = st.text_area("Or edit/paste Resume text here:", value=uploaded_text, height=180, placeholder="CV text will appear here automatically after upload...")

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Calculate Match Rate 🚀", use_container_width=True):
    if job_desc.strip() and cv_text.strip():
        # Initialize a clean vectorizer to prevent pruning errors
        vectorizer = TfidfVectorizer(stop_words='english')
        
        try:
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
        except ValueError:
            st.error("Could not process text. Please ensure both fields contain meaningful words/text.")
    else:
        st.error("Please enter both Job Description and Resume content first!")
