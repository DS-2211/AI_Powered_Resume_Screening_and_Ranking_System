import streamlit as st
import pdfplumber
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += re.sub(r'[^a-zA-Z0-9\s]', '', page_text.lower()) + "\n"
    
    return text.strip() if text else None

def normalize_similarity_scores(similarity_scores):
    max_score = max(similarity_scores) if max(similarity_scores) > 0 else 1
    return similarity_scores / max_score

def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=50000,
        ngram_range=(1, 2),
        analyzer='word',
        sublinear_tf=True,
        norm='l2'
    )
    
    vectors = vectorizer.fit_transform(documents).toarray()
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    cosine_similarities = normalize_similarity_scores(cosine_similarities)
    
    return sorted(zip(resumes, cosine_similarities), key=lambda x: x[1], reverse=True)

st.title("AI Resume Screening & Candidate Ranking System")

job_description = st.text_area("Enter the job description")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if st.button("Rank Resumes"):
    if not job_description:
        st.error("Please enter a job description.")
    
    elif not uploaded_files:
        st.error("Please upload at least one resume.")
    
    else:
        resumes_texts, file_names = [], []
        
        for uploaded_file in uploaded_files:
            text = extract_text_from_pdf(uploaded_file)
            if text:
                resumes_texts.append(text)
                file_names.append(uploaded_file.name)
            else:
                st.warning(f"Could not extract text from {uploaded_file.name}")
        
        if resumes_texts:
            ranked_resumes = rank_resumes(job_description, resumes_texts)
            results_df = pd.DataFrame(ranked_resumes, columns=["Resume Text", "Similarity Score"])
            results_df.insert(0, "File Name", file_names)
            
            st.header("Ranked Resumes")
            st.dataframe(results_df[["File Name", "Similarity Score"]])
            
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Rankings as CSV", data=csv, file_name="resume_rankings.csv", mime="text/csv")
        
        else:
            st.error("No valid resumes were processed.")
