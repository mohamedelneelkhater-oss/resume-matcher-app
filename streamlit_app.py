import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# إعدادات الصفحة
st.set_page_config(page_title="TalentMatch - CV Matcher", layout="wide", page_icon="📄")

st.title("📄 نظام مطابقة السير الذاتية بالوظائف (TalentMatch)")
st.write("قم بإدخال الوصف الوظيفي والسيرة الذاتية لحساب نسبة المطابقة ومقارنة الكفاءة.")

st.markdown("---")

# تقسيم الشاشة إلى عمودين مدخلات
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 الوصف الوظيفي (Job Description)")
    job_desc = st.text_area("أدخل تفاصيل ومتطلبات الوظيفة هنا:", height=220, placeholder="مثال: المطلوب مهندس بيانات يجيد Python و SQL...")

with col2:
    st.subheader("📝 السيرة الذاتية (CV Content)")
    cv_text = st.text_area("أدخل نص السيرة الذاتية هنا:", height=220, placeholder="مثال: خبرة 3 سنوات في تحليل البيانات واستخدام Python...")

# تحميل نموذج TF-IDF إن وجد أو إنشاؤه مباشرة
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

# زر حساب المطابقة
if st.button("حساب نسبة المطابقة 🚀", use_container_width=True):
    if job_desc.strip() and cv_text.strip():
        # استخراج المتجهات وحساب التشابه
        tfidf_matrix = vectorizer.fit_transform([job_desc, cv_text])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
        
        st.markdown("---")
        st.subheader("📊 نتيجة التحليل والمطابقة:")
        
        # عرض النتيجة
        if score >= 70:
            st.success(f"🎯 نسبة المطابقة ممتازة: {score:.2f}%")
        elif score >= 40:
            st.info(f"⚡ نسبة المطابقة متوسطة: {score:.2f}%")
        else:
            st.warning(f"⚠️ نسبة المطابقة ضعيفة: {score:.2f}%")
            
        st.progress(int(score))
    else:
        st.error("يرجى إدخال كل من الوصف الوظيفي والسيرة الذاتية أولاً!")
