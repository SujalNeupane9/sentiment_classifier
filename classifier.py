import re
import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer

# Load Sentence Transformer model
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

# Load CatBoostClassifier model
with open('model/catboost_model.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Define sentiment labels
sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}

# Define Streamlit app
def app():
    # Set page title
    st.set_page_config(page_title='Sentiment Analysis App')

    # Set app title
    st.title('Sentiment Analysis App')

    # Prompt user for input
    user_input = st.text_input('Enter a review:')
    
    # Check if user has entered anything
    if user_input:
        # Preprocess input text
        user_input = re.sub(r'[^\w\s]', '', user_input.lower())
        
        # Generate embeddings for input text
        input_embedding = model.encode(user_input)

        # Predict sentiment using loaded classifier
        sentiment_prediction = classifier.predict(input_embedding)[0]

        # Map predicted sentiment to label
        sentiment_label = sentiment_labels[sentiment_prediction]

        # Display result to user
        st.write(f'The sentiment of your review is {sentiment_label}.')

if __name__ == "__main__":
    app()
