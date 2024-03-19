# import subprocess

# # Install required libraries from requirements.txt
# subprocess.call(['pip', 'install', '-r', 'requirements.txt'])


import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from PAL import preprocess_test_data, predict_accident_level
import numpy as np 

# Load the pre-trained model
model = load_model('model.h5')
label_to_index = {'I':0, 'II':1 ,'III': 2,'IV':3,'V':4}
MAX_LENGTH=100
VOCAB_SIZE = 10000

# Load the tokenizer
tokenizer = Tokenizer(num_words=VOCAB_SIZE)


def preprocess_text(text):
  # Lowercase text
  text = text.lower()
  # Remove punctuation
  text = ''.join([c for c in text if c.isalpha() or c.isspace()])
  # Tokenize text (split into words)
  tokens = text.split()

  # Create a tokenizer (fit on training data later)
  tokenizer = Tokenizer(num_words=VOCAB_SIZE)  


  return tokens  # Return the list of tokens 


def main():
    st.title('Accident Level Predictor')

    # Text input for user to enter description
    input_text = st.text_input('Enter description of the accident')

    if st.button('Predict Accident Level'):
        # Preprocess input text
        padded_input = preprocess_test_data([input_text], tokenizer)
        processed_input = preprocess_text(input_text)
        tokenized_input = tokenizer.texts_to_sequences([processed_input])
        padded_input = pad_sequences(tokenized_input, maxlen=MAX_LENGTH)

        # Predict accident level
        predictions = model.predict(padded_input)
        predicted_class_index = np.argmax(predictions)
        predicted_label = None
        for label, index in label_to_index.items():
            if index == predicted_class_index:
                predicted_label = label
                break

        # Display predicted accident level
        st.write(f'Predicted Accident Level: {predicted_class_index,predicted_label}')


if __name__ == '__main__':
    main()