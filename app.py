import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)


def get_match_score(resume_text, jd_text):
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_clean, jd_clean])

    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    match_percent = round(similarity * 100, 2)

    feature_names = vectorizer.get_feature_names_out()
    resume_vector = vectors[0].toarray()[0]
    jd_vector = vectors[1].toarray()[0]

    common_words = [
        word for i, word in enumerate(feature_names)
        if resume_vector[i] > 0 and jd_vector[i] > 0
    ]

    jd_words = set(jd_clean.split())
    resume_words = set(resume_clean.split())
    missing_keywords = list(jd_words - resume_words)

    return match_percent, sorted(common_words), sorted(missing_keywords)[:15]


st.set_page_config(page_title="Resume - JD Matcher", page_icon="📄", layout="centered")

st.title("📄 Resume — Job Description Matcher")
st.write("Paste your resume text and a job description below to see how well they match.")

col1, col2 = st.columns(2)

with col1:
    resume_input = st.text_area("Resume text", height=300, placeholder="Paste your resume content here...")

with col2:
    jd_input = st.text_area("Job description", height=300, placeholder="Paste the job description here...")

if st.button("Check Match Score", type="primary"):
    if resume_input.strip() and jd_input.strip():
        score, common, missing = get_match_score(resume_input, jd_input)

        st.subheader(f"Match Score: {score}%")
        st.progress(min(int(score), 100))

        if score >= 70:
            st.success("Strong match! Your resume aligns well with this job.")
        elif score >= 40:
            st.warning("Moderate match. Consider adding more relevant keywords.")
        else:
            st.error("Low match. Your resume may need significant tailoring for this role.")

        st.subheader("Matching keywords found")
        st.write(", ".join(common) if common else "No common keywords found.")

        st.subheader("Keywords in JD missing from your resume")
        st.write(", ".join(missing) if missing else "Great! No major keywords missing.")
    else:
        st.warning("Please paste both resume text and job description before checking.")

st.caption("Built with Python, scikit-learn (TF-IDF + cosine similarity), and Streamlit.")
