import re
import spacy
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text
from docx import Document
from pathlib import Path

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Streamlit App Title
st.title("📂 Fit Mave")

st.write("🔍 Upload a **job description** and multiple **resumes** to find the best matches based on similarity scores.")

# Upload job description
job_desc_file = st.file_uploader("📜 Upload Job Description (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

# Upload multiple resumes
resume_files = st.file_uploader("📄 Upload Resumes (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

def extract_text_from_file(uploaded_file):
    """Extract text from PDF, DOCX, or TXT files with error handling."""
    try:
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        if file_extension == ".pdf":
            text = extract_text(uploaded_file)
        elif file_extension == ".docx":
            doc = Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif file_extension == ".txt":
            text = uploaded_file.read().decode("utf-8")
        else:
            return None, "Unsupported file format"
        
        # Check for empty or corrupted files
        if not text.strip() or len(text) < 50:
            return None, "File appears empty or corrupted"
        
        return text, None  # No error
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

def preprocess_text(text):
    """Preprocess text using spaCy."""
    doc = nlp(text.lower())
    return " ".join(token.lemma_ for token in doc if token.is_alpha and not token.is_stop)

def calculate_similarity(job_desc, resumes):
    """Calculate cosine similarity using TF-IDF."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    job_desc_vector = vectorizer.fit_transform([job_desc])
    resume_vectors = vectorizer.transform(resumes)
    return cosine_similarity(job_desc_vector, resume_vectors).flatten()

if st.button("🚀 Analyze Resumes"):
    if not job_desc_file or not resume_files:
        st.error("⚠️ Please upload both a job description and at least one resume.")
    else:
        with st.spinner("Processing... Please wait! ⏳"):
            resumes = {}
            skipped_files = []

            # Extract job description text
            job_text, error = extract_text_from_file(job_desc_file)
            if error:
                st.error(f"⚠️ Error processing Job Description: {error}")
            else:
                job_text_processed = preprocess_text(job_text)

                # Process resumes
                for resume_file in resume_files:
                    text, error = extract_text_from_file(resume_file)
                    if error:
                        skipped_files.append((resume_file.name, error))
                        continue  # Skip corrupted/unreadable files

                    resumes[resume_file.name] = {
                        "raw_text": text,
                        "processed_text": preprocess_text(text)
                    }

                if not resumes:
                    st.warning("⚠️ No valid resumes processed.")
                else:
                    # Compute similarity scores
                    resume_texts = [data["processed_text"] for data in resumes.values()]
                    scores = calculate_similarity(job_text_processed, resume_texts)

                    # Convert results to a DataFrame
                    results_df = pd.DataFrame([
                        {
                            "Resume": name,
                            "Match Score": round(score, 2),
                            "Status": "✅ Recommended" if score >= 0.25 else "❌ Rejected"
                        }
                        for (name, data), score in zip(resumes.items(), scores)
                    ]).sort_values(by="Match Score", ascending=False)

                    # Display Results
                    st.subheader("📊 Resume Matching Results")
                    st.dataframe(results_df)

                    # Skipped Files Report
                    if skipped_files:
                        st.subheader("⚠️ Skipped Files")
                        for file, reason in skipped_files:
                            st.write(f"❌ `{file}` - {reason}")

st.sidebar.markdown("👨‍💻 **Developed by Awaiz Kazi**")
