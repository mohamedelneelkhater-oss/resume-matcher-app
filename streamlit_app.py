import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import docx
import nltk

# تنزيل بيانات NLTK المطلوبة للتقطيع التلقائي
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
# ---------------------------------------------------------
# 1. إعدادات الصفحة وتحميل موارد NLTK
# ---------------------------------------------------------
st.set_page_config(page_title="نظام مطابقة السيرة الذاتية مع الوظيفة", layout="wide")

@st.cache_resource
def load_nltk_resources():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

load_nltk_resources()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ---------------------------------------------------------
# 2. تحميل النموذج
# ---------------------------------------------------------
@st.cache_resource
def load_tfidf_model():
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        return pickle.load(f)

try:
    tfidf = load_tfidf_model()
except Exception as e:
    st.error("تعذر تحميل ملف النموذج `tfidf_vectorizer.pkl`. يرجى التاكد من وجوده في المستودع.")
    st.stop()

# ---------------------------------------------------------
# 3. دوال قراءة الملفات والمعالجة المسبقة
# ---------------------------------------------------------
def extract_text_from_file(uploaded_file):
    text = ""
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type == 'pdf':
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
    elif file_type == 'docx':
        doc = docx.Document(uploaded_file)
        for paragraph in doc.paragraphs:
            text += paragraph.text + " "
    elif file_type == 'txt':
        text = uploaded_file.read().decode('utf-8')
        
    return text

def advanced_clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    tokens = word_tokenize(text)
    cleaned = [lemmatizer.lemmatize(token) for token in tokens if lemmatizer.lemmatize(token) not in stop_words and len(lemmatizer.lemmatize(token)) > 2]
    return " ".join(cleaned)

# ---------------------------------------------------------
# 4. واجهة المستخدم (UI)
# ---------------------------------------------------------
st.title("📄 نظام مطابقة السيرة الذاتية مع الوصف الوظيفي")
st.write("قم بإرفاق السيرة الذاتية وإدخال الوصف الوظيفي لحساب نسبة المطابقة.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 الوصف الوظيفي (Job Description)")
    job_desc_input = st.text_area("أدخل تفاصيل ومتطلبات الوظيفة هنا:", height=220, placeholder="مثال: المطلوب مهندس بيانات يجيد Python و SQL...")

with col2:
    st.subheader("📑 السيرة الذاتية (CV)")
    
    # خيار إرفاق ملف من الجهاز
    uploaded_file = st.file_uploader("ارفق ملف السيرة الذاتية (PDF, DOCX, TXT):", type=['pdf', 'docx', 'txt'])
    
    # مربع نص اختياري للنسخ المباشر
    cv_text_input = st.text_area("أو أدخل نص السيرة الذاتية يدويًا:", height=100, placeholder="مثال: خبرة 3 سنوات في تحليل البيانات واستخدام Python...")

if st.button("🚀 حساب نسبة المطابقة", use_container_width=True):
    cv_text = ""
    
    # سحب النص من الملف المرفق أو مربع النص
    if uploaded_file is not None:
        cv_text = extract_text_from_file(uploaded_file)
    elif cv_text_input.strip():
        cv_text = cv_text_input.strip()

    if not job_desc_input.strip() or not cv_text.strip():
        st.error("يرجى إدخال الوصف الوظيفي وإرفاق/كتابة السيرة الذاتية أولاً!")
    else:
        # معالجة النصوص
        clean_job = advanced_clean_text(job_desc_input)
        clean_cv = advanced_clean_text(cv_text)

        # التحويل عبر TF-IDF
        vectors = tfidf.transform([clean_job, clean_cv])
        
        # حساب نسبة التشابه
        similarity_score = cosine_similarity(vectors[0], vectors[1])[0][0] * 100
        
        st.success(f"🎯 **نسبة المطابقة بين السيرة الذاتية والوظيفة:** `{similarity_score:.2f}%`")
        
