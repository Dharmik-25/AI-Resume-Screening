import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Load spaCy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

# Extract key skills using NER
def extract_skills(text):
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PERSON", "PRODUCT", "EVENT"]]
    return list(set(skills))

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    
    job_desc_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_desc_vector], resume_vectors).flatten()
    
    return cosine_similarities

# Streamlit UI
st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("🚀 AI Resume Screening & Candidate Ranking System")

# Job Description input
st.header("📝 Enter Job Description")
job_description = st.text_area("Enter the job description", height=150)

# File uploader
st.header("📂 Upload Resumes (PDFs Only)")
uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("📊 Ranking Resumes...")

    resume_texts = []
    skill_sets = []
    
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resume_texts.append(text)
        skills = extract_skills(text)
        skill_sets.append(", ".join(skills))

    # Rank resumes
    scores = rank_resumes(job_description, resume_texts)

    # Create Results DataFrame
    results = pd.DataFrame({
        "Resume Name": [file.name for file in uploaded_files],
        "Match Score": scores,
        "Extracted Skills": skill_sets
    })

    # Sort results
    results = results.sort_values(by="Match Score", ascending=False)

    # Display results
    st.dataframe(results)

    # Visualization - Bar Chart
    st.subheader("📊 Resume Ranking Visualization")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=results["Match Score"], y=results["Resume Name"], palette="viridis", ax=ax)
    ax.set_xlabel("Match Score")
    ax.set_ylabel("Resume Name")
    st.pyplot(fig)

    # Download results as CSV
    st.subheader("📥 Download Ranked Resumes")
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="ranked_resumes.csv", mime="text/csv")



   