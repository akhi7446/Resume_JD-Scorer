import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Ensure protocol buffers are correctly implemented
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Function to calculate the match percentage between two texts
def calculate_match_percentage(text1, text2):
    vectorizer = CountVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(vectors)[0][1]
    return cosine_sim * 100

# Function to extract key terms from text
def extract_key_terms(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    return set(filtered_words)

# Streamlit UI
st.title('Resume to Job Description Matcher')

# Text areas for user input
resume_text = st.text_area("Paste Your Resume Here")
jd_text = st.text_area("Paste Job Description Here")

if st.button('Match'):
    if resume_text and jd_text:
        # Calculate the match percentage
        match_percentage = calculate_match_percentage(resume_text, jd_text)
        st.write(f"Match Percentage: {match_percentage:.2f}%")

        # Extracting key terms from JD and checking against the resume
        jd_terms = extract_key_terms(jd_text)
        resume_terms = extract_key_terms(resume_text)
        missing_terms = jd_terms - resume_terms

        # Display feedback based on match percentage
        if match_percentage >= 70:
            st.success("Good chances of getting your resume shortlisted.")
        elif 40 <= match_percentage < 70:
            st.warning("Good match but can be improved.")
            if missing_terms:
                st.info(f"Consider adding these key terms from the job description to your resume:\n {', '.join(missing_terms)}")
        elif match_percentage < 40:
            st.error("Poor match.")
            if missing_terms:
                st.info(f"Your resume is missing these key terms from the job description:\n {', '.join(missing_terms)}")
    else:
        st.warning("Please enter both Resume and Job Description.")
