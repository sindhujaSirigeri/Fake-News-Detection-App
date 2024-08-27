# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import h5py
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load necessary NLP resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialise the lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load tokeniser, TF-IDF vectoriser, scaler, and model
with open('fake-news-predictor/data/processed/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('fake-news-predictor/data/processed/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('fake-news-predictor/data/splits/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

model_path = 'fake-news-predictor/models/logistic_regression_model.pkl'
model = joblib.load(model_path)

# Title of the app
st.title("Fake News Classification using Machine Learning")

st.header("How to Use This App")
st.write("""
- **Step 1:** Enter the title of the news article in the provided text box. This step is optional.
- **Step 2:** Paste or type the body of the news article into the text area below.
- **Step 3:** Click on the **Predict** button to see whether the article is likely to be classified as Fake or Real News.
""")

# Side panel with data information
st.sidebar.header("About This Project")
st.sidebar.write("""
This app uses Natural Language Processing (NLP) and Machine Learning to classify news articles as fake or real. The model was trained on a large dataset of news articles and aims to help users discern the credibility of online content.
""")

st.sidebar.header("Model Performance")
st.sidebar.write("""
The model used for predictions has the following performance metrics:
- **Accuracy:** 95%
- **Precision:** 92%
- **Recall:** 90%
""")

st.sidebar.header("Dataset Information")
st.sidebar.write("This project uses the WELFake dataset, which is designed for fake news classification. The dataset consists of 72,134 news articles, with 35,028 labeled as real news and 37,106 labeled as fake news. The data was sourced from four popular news datasets—Kaggle, McIntire, Reuters, and BuzzFeed Political—to provide a diverse set of text data and prevent overfitting in machine learning models.")
st.sidebar.write("Dataset from Kaggle: [Fake News Classification](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)")
st.sidebar.write("The dataset was published in the following paper:\n- **IEEE Transactions on Computational Social Systems**: pp. 1-13 (doi: [10.1109/TCSS.2021.3068519](https://doi.org/10.1109/TCSS.2021.3068519)).")


# Define the preprocess_input function
def preprocess_input(title, text):
    # Step 1: Handle missing title
    if not title.strip():
        title = 'No Title'

    # Step 2: Combine title and text
    combined_text = title + " " + text

    # Step 3: Remove stopwords
    def remove_stopwords(text):
        word_tokens = text.split()
        filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
        return ' '.join(filtered_text)

    cleaned_text = remove_stopwords(combined_text)

    # Step 4: Remove numerics and special characters
    def clean_text(text):
        text = re.sub(r'[^A-Za-z\s]', '', text)  # Keep only letters and spaces
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text

    cleaned_text = clean_text(cleaned_text)

    # Step 5: Lowercase before lemmatization
    cleaned_text = cleaned_text.lower()

    # Step 6: Lemmatization
    def lemmatize_text(text):
        return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    lemmatized_text = lemmatize_text(cleaned_text)

    # Step 7: Vectorization
    vectorized_input = vectorizer.transform([lemmatized_text])
    
    # Step 8: Combine vectorized input with additional features
    original_length = len(lemmatized_text.split())  # Count words in the text
    log_transformed_length = np.log1p(original_length)
    additional_features = np.array([[log_transformed_length]])
    combined_input = np.hstack([vectorized_input.toarray(), additional_features])

    # Step 9: Scaling
    scaled_input = scaler.transform(combined_input)

    return scaled_input

# Input fields for the user to enter the title and text
title = st.text_input("Enter the title of the news article:")
text = st.text_area("Enter the text of the news article:")

# Processing the input when the "Predict" button is clicked
if st.button("Predict"):
    if not text.strip():
        st.error("Text is a required field.")
    else:
        # Process user input
        processed_input = preprocess_input(title, text)

        # Make prediction
        prediction = model.predict(processed_input)

        # Interpret the prediction and display styled output
        if prediction[0] == 0:
            st.markdown("<div style='padding:10px; border-radius:5px; background-color:#FFCCCC; color:#900000;'>"
                        "The model predicts this is likely <strong>Fake News</strong>.</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<div style='padding:10px; border-radius:5px; background-color:#CCFFCC; color:#006600;'>"
                        "The model predicts this is likely <strong>Real News</strong>.</div>",
                        unsafe_allow_html=True)

st.header("Understanding the Output")
st.write("""
- **Fake News:** The model predicts that the content entered is likely to be fabricated or misleading.
- **Real News:** The model predicts that the content entered is likely to be factual and accurate.
""")

# Commented out testing section for future reference
# Test the function with different inputs
# test_title_1 = "Breaking News"
# test_text_1 = "This is a TEST article on RUNNING dogs."
# final_sequence_1 = preprocess_input(test_title_1, test_text_1)
# st.write(f"Final Scaled Input 1: {final_sequence_1}")

# test_title_2 = "Faster Foxes"
# test_text_2 = "The FOXES were RUNNING faster than the other ANIMALS."
# final_sequence_2 = preprocess_input(test_title_2, test_text_2)
# st.write(f"Final Scaled Input 2: {final_sequence_2}")
