import streamlit as st
import re
import nltk
import io
import pdfplumber
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


def extract_pdf_text(uploaded_file):
    text = ""
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


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


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume–JD Matcher",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

/* Reset Streamlit defaults */
html, body, [data-testid="stAppViewContainer"] {
    background: #0b0f1a !important;
}
[data-testid="stAppViewContainer"] > .main {
    background: #0b0f1a;
}
[data-testid="stHeader"] { background: transparent !important; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }

/* Global font */
* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

/* ── Landing hero ── */
.hero-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 88vh;
    text-align: center;
    padding: 2rem 1rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border-radius: 999px;
    padding: 5px 16px;
    margin-bottom: 1.6rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2.4rem, 6vw, 4rem);
    font-weight: 700;
    line-height: 1.1;
    color: #f0f4ff;
    margin: 0 0 1rem;
    letter-spacing: -0.02em;
}
.hero-title span {
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1.05rem;
    color: #94a3b8;
    max-width: 480px;
    line-height: 1.7;
    margin: 0 0 2.6rem;
}
.hero-stats {
    display: flex;
    gap: 2.5rem;
    margin-bottom: 2.8rem;
    flex-wrap: wrap;
    justify-content: center;
}
.stat-item { text-align: center; }
.stat-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #818cf8;
    line-height: 1;
}
.stat-label { font-size: 0.72rem; color: #64748b; letter-spacing: 0.05em; margin-top: 4px; }
.hero-cta-hint { font-size: 0.78rem; color: #475569; margin-top: 0.6rem; }

/* ── Section header ── */
.section-header {
    text-align: center;
    margin: 2rem 0 1.6rem;
}
.section-header h2 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    color: #e2e8f0;
    font-weight: 700;
    margin: 0 0 0.4rem;
}
.section-header p { font-size: 0.88rem; color: #64748b; margin: 0; }

/* ── Upload cards ── */
.upload-card {
    background: #131929;
    border: 1px solid #1e2d47;
    border-radius: 14px;
    padding: 1.4rem 1.4rem 1rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.upload-card:hover { border-color: #3b4f6e; }
.card-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #818cf8;
    margin-bottom: 0.6rem;
}
.card-title { font-size: 1rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.3rem; }
.card-desc { font-size: 0.8rem; color: #64748b; margin-bottom: 1rem; }

/* ── Results ── */
.result-card {
    background: #131929;
    border: 1px solid #1e2d47;
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1.5rem;
    text-align: center;
}
.score-ring {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 4rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.score-label { font-size: 0.82rem; color: #64748b; letter-spacing: 0.06em; text-transform: uppercase; }
.verdict-pill {
    display: inline-block;
    border-radius: 999px;
    padding: 6px 20px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-top: 1rem;
}

.kw-section {
    background: #131929;
    border: 1px solid #1e2d47;
    border-radius: 14px;
    padding: 1.4rem;
    margin-top: 1rem;
}
.kw-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
}
.kw-green { color: #34d399; }
.kw-amber { color: #fbbf24; }
.chip-wrap { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.chip {
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.75rem;
    font-weight: 500;
}
.chip-green { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.25); }
.chip-amber { background: rgba(251,191,36,0.10); color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }

/* ── Streamlit widget overrides ── */
.stTextArea textarea {
    background: #0e1420 !important;
    border: 1px solid #1e2d47 !important;
    border-radius: 10px !important;
    color: #cbd5e1 !important;
    font-size: 0.85rem !important;
    resize: vertical;
}
.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.18) !important;
}
[data-testid="stFileUploader"] {
    background: #0e1420 !important;
    border: 1px dashed #2a3a56 !important;
    border-radius: 10px !important;
    padding: 0.6rem !important;
}
[data-testid="stFileUploader"]:hover { border-color: #6366f1 !important; }
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 2.2rem !important;
    width: 100%;
    transition: opacity 0.2s, transform 0.1s !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #6366f1, #c084fc) !important;
    border-radius: 999px !important;
}
[data-testid="stProgress"] {
    background: #1e2d47 !important;
    border-radius: 999px !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: #0e1420 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    border-radius: 7px !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: #1e2d47 !important;
    color: #a5b4fc !important;
}
.stRadio > div { flex-direction: row !important; gap: 1rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "landing"


# ══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "landing":

    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-badge">🎯 Smart Resume Analysis</div>
        <h1 class="hero-title">Resume–JD<br><span>Match Score</span></h1>
        <p class="hero-sub">
            Instantly see how well your resume aligns with any job description.
            Upload a PDF or paste your text — get a score, matched keywords, and gaps to fix.
        </p>
        <div class="hero-stats">
            <div class="stat-item">
                <div class="stat-num">TF-IDF</div>
                <div class="stat-label">Algorithm</div>
            </div>
            <div class="stat-item">
                <div class="stat-num">PDF</div>
                <div class="stat-label">Upload support</div>
            </div>
            <div class="stat-item">
                <div class="stat-num">~2s</div>
                <div class="stat-label">Analysis time</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        if st.button("🚀  Analyse My Resume", use_container_width=True):
            st.session_state.page = "app"
            st.rerun()

    st.markdown('<p class="hero-cta-hint" style="text-align:center">No sign-up required · runs locally</p>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP PAGE
# ══════════════════════════════════════════════════════════════════════════════
else:

    # Back button
    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← Back"):
            st.session_state.page = "landing"
            st.rerun()

    st.markdown("""
    <div class="section-header">
        <h2>📄 Resume–JD Matcher</h2>
        <p>Paste text or upload a PDF for your resume, then add the job description</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Resume input ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="upload-card">
        <div class="card-label">Step 1</div>
        <div class="card-title">Your Resume</div>
        <div class="card-desc">Upload a PDF <em>or</em> paste plain text below</div>
    </div>
    """, unsafe_allow_html=True)

    resume_mode = st.radio("Resume input method", ["📎 Upload PDF", "✏️ Paste text"], horizontal=True, label_visibility="collapsed")

    resume_text = ""
    if resume_mode == "📎 Upload PDF":
        resume_file = st.file_uploader("Upload resume PDF", type=["pdf"], label_visibility="collapsed", key="resume_pdf")
        if resume_file:
            with st.spinner("Extracting text from PDF…"):
                try:
                    resume_text = extract_pdf_text(resume_file)
                    st.success(f"✅ Extracted {len(resume_text.split())} words from your resume")
                    with st.expander("Preview extracted text"):
                        st.text(resume_text[:800] + ("…" if len(resume_text) > 800 else ""))
                except Exception as e:
                    st.error(f"Could not read PDF: {e}")
    else:
        resume_text = st.text_area("Resume text", height=220, placeholder="Paste your resume content here…", label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── JD input ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="upload-card">
        <div class="card-label">Step 2</div>
        <div class="card-title">Job Description</div>
        <div class="card-desc">Paste the full job description from the listing</div>
    </div>
    """, unsafe_allow_html=True)

    jd_text = st.text_area("Job description", height=220, placeholder="Paste the job description here…", label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Analyse button ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run = st.button("🔍  Analyse Match", use_container_width=True)

    # ── Results ───────────────────────────────────────────────────────────────
    if run:
        if resume_text.strip() and jd_text.strip():
            with st.spinner("Analysing…"):
                score, common, missing = get_match_score(resume_text, jd_text)

            # Score colour
            if score >= 70:
                score_color = "#34d399"
                verdict_text = "Strong match"
                verdict_bg = "rgba(52,211,153,0.12)"
                verdict_border = "rgba(52,211,153,0.35)"
                verdict_color = "#34d399"
            elif score >= 40:
                score_color = "#fbbf24"
                verdict_text = "Moderate match — consider adding keywords"
                verdict_bg = "rgba(251,191,36,0.10)"
                verdict_border = "rgba(251,191,36,0.35)"
                verdict_color = "#fbbf24"
            else:
                score_color = "#f87171"
                verdict_text = "Low match — your resume needs tailoring"
                verdict_bg = "rgba(248,113,113,0.12)"
                verdict_border = "rgba(248,113,113,0.35)"
                verdict_color = "#f87171"

            st.markdown(f"""
            <div class="result-card">
                <div class="score-ring" style="color:{score_color}">{score}%</div>
                <div class="score-label">Match Score</div>
                <div class="verdict-pill" style="background:{verdict_bg};border:1px solid {verdict_border};color:{verdict_color}">
                    {verdict_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(min(int(score), 100))

            # Keyword columns
            col_a, col_b = st.columns(2)

            with col_a:
                chips = "".join(f'<span class="chip chip-green">{w}</span>' for w in common) if common else '<span style="color:#475569;font-size:0.82rem">None found</span>'
                st.markdown(f"""
                <div class="kw-section">
                    <div class="kw-title kw-green">✅ Matched Keywords ({len(common)})</div>
                    <div class="chip-wrap">{chips}</div>
                </div>
                """, unsafe_allow_html=True)

            with col_b:
                chips = "".join(f'<span class="chip chip-amber">{w}</span>' for w in missing) if missing else '<span style="color:#475569;font-size:0.82rem">None — great coverage!</span>'
                st.markdown(f"""
                <div class="kw-section">
                    <div class="kw-title kw-amber">⚠️ Missing Keywords ({len(missing)})</div>
                    <div class="chip-wrap">{chips}</div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.warning("Please provide both your resume and a job description before analysing.")
