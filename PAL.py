# -*- coding: utf-8 -*-
"""Capstone Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SWQ8Vb6iANexNW61I1QH0GuBs47qLt5l
"""

# !pip install tensorflow==2.12 -q
# !pip install --upgrade tensorflow-probability==0.17.0 -q
# !pip install missingno -q
# !pip install catboost -q
# !pip install holidays -q
# !pip install bokeh matplotlib plotly -q
# !pip install hvplot -q
# !pip install transformers -q
# !pip install nlpaug -q
# !pip install sacremoses -q

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random, re
import time
import warnings
import missingno as mno
import holoviews as hv
from holoviews import opts
import bokeh
import holidays
# %matplotlib inline

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve### metrics
import datetime
import pathlib
import io
import os
from numpy import random
import gensim.downloader as api
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (Dense,Flatten,SimpleRNN,InputLayer,Conv1D,Bidirectional,GRU,LSTM,BatchNormalization,Dropout,Input, Embedding,TextVectorization)
from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy,TopKCategoricalAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
# from google.colab import drive
# from google.colab import files
from tensorboard.plugins import projector
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Hyperparameters 
MAX_LENGTH = 100  # Maximum sequence length
VOCAB_SIZE = 10000  # Number of words to consider in the vocabulary
EMBEDDING_DIM = 128  # Embedding dimension for words
LSTM_UNITS = 64  # Number of units in LSTM layers



# Preprocessing Function (fixed to return integer indices)
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

def preprocess_test_data(tokens, tokenizer):
    """
    Preprocesses test data by converting text sequences to integer sequences and padding them.

    Parameters:
    - X_test (list of str): List of text sequences to preprocess.
    - tokenizer: Tokenizer object used to convert text to sequences.
    - MAX_LENGTH (int): Maximum length for padding sequences.

    Returns:
    - X_test_encoded (numpy.ndarray): Padded integer sequences representing the preprocessed test data.
    """
    MAX_LENGTH = 100
    # Convert text sequences to integer sequences
    text_sequences = tokenizer.texts_to_sequences(tokens)
    # Pad sequences
    padded_input = pad_sequences(text_sequences, maxlen=MAX_LENGTH)
    padded_input = np.array(padded_input)
    
    return padded_input

def predict_accident_level(padded_input, loaded_model, label_to_index):
    """
    Predicts the accident level for a given padded input using a loaded model.

    Parameters:
    - padded_input (numpy.ndarray): Padded input data for prediction.
    - loaded_model: Pre-trained model loaded for prediction.
    - label_to_index (dict): Mapping from index to label for decoding predictions.

    Returns:
    - predicted_accident_level (str): Predicted accident level.
    """
    # Make predictions using the loaded model
    label_to_index = {'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4}
    predictions = loaded_model.predict(padded_input)
    predicted_class_index = np.argmax(predictions)

    # Get the corresponding label using the dictionary
    predicted_label = None
    for label, index in label_to_index.items():
        if index == predicted_class_index:
            predicted_label = label
            break
    
    return predicted_class_index,predicted_label





def predict_accident_from_text(input_text, tokenizer, loaded_model):
    """
    Predicts the accident level for a given input text using a loaded model.

    Parameters:
    - input_text (str): Input text to predict the accident level.
    - tokenizer: Tokenizer object used to convert text to sequences.
    - MAX_LENGTH (int): Maximum length for padding sequences.
    - loaded_model: Pre-trained model loaded for prediction.
    - label_to_index (dict): Mapping from index to label for decoding predictions.

    Returns:
    - predicted_accident_level (str): Predicted accident level.
    """
    # Preprocess the input text
    padded_input = preprocess_test_data([input_text], tokenizer, MAX_LENGTH)
    
    # Predict the accident level
    predicted_class_index,predicted_label = predict_accident_level(padded_input, loaded_model, label_to_index)
    
    return predicted_class_index,predicted_label





