import tensorflow
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN
from tensorflow.keras.models import load_model


(x_train, y_train),(x_test,y_test) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

model = load_model('imdb_rnn_model_final.h5')

def preprocess_text(text):
    words = text.lower().split()
    encoded_text = [word_index.get(word, 2) + 3 for word in words]  # 2 is the index for unknown words
    padded_text = pad_sequences([encoded_text], maxlen=500)
    return padded_text

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    print(preprocessed_input)

    model_output = model.predict(preprocessed_input)
    print(model_output[0][0])

    if model_output[0][0] > 0.5:
        sentiment = 'positive'
    else:
        sentiment = 'negative'
    return sentiment, model_output[0][0]

st.title("IMDB Movie Review Sentiment Analysis")

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)
    model_output = model.predict(preprocessed_input)
    if model_output[0][0] > 0.5:
        sentiment = 'positive'
    else:
        sentiment = 'negative'
    st.write(f"Sentiment: {sentiment}, Score: {model_output[0][0]:.2f}")
else:
    st.write("Please write a movie review to classify")
