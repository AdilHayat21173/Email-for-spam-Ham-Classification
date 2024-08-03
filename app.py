import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

# Load the trained model
model = load_model('lstm_model.keras')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = joblib.load(handle)

# Define parameters
max_sequence_length = 100

# Streamlit app title
st.title('Email Classification')

# Text input
text_input = st.text_area("Enter the email content:")

# Predict button
if st.button('Classify'):
    if text_input:
        # Preprocess the input text
        sequences = tokenizer.texts_to_sequences([text_input])
        padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
        
        # Make prediction
        prediction = model.predict(padded_sequences)
        predicted_class = np.argmax(prediction, axis=1)
        class_labels = ['not spam', 'spam']  # Modify this based on your dataset's labels

        # Display the result
        st.write(f"Predicted Class: {class_labels[predicted_class[0]]}")
    else:
        st.write("Please enter the email content.")
