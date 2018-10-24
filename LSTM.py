# Keras
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, CuDNNLSTM, SimpleRNN
from keras.layers.embeddings import Embedding


# NLTK
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import SnowballStemmer

import numpy as np
import pandas as pd

import os
os.environ['MKL_NUM_THREADS'] = '12'
os.environ['GOTO_NUM_THREADS'] = '12'
os.environ['OMP_NUM_THREADS'] = '12'
os.environ['openmp'] = 'True'


file = '/home/tianyi/Desktop/yelp/yelp_dataset/yelp_academic_dataset_review.csv'

df = pd.read_csv(file, usecols=['stars', 'text'], error_bad_lines=False)

# df = df.iloc[1:500]

df = df.dropna()
df = df[df.text.apply(lambda x: x != "")]
df = df[df.stars.apply(lambda x: x != "")]

df.head(50)

df.groupby('stars').count()

labels = df['stars'].map(lambda x: 1 if int(x) > 3 else 0)

K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=12, inter_op_parallelism_threads=12)))

vocabulary_size = 1000
tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(df['text'])

sequences = tokenizer.texts_to_sequences(df['text'])
data = pad_sequences(sequences, maxlen=50)

print(data.shape)

# RNN
model_rnn = Sequential()
model_rnn.add(Embedding(vocabulary_size, 100, input_length=50))
model_rnn.add(SimpleRNN(100))
model_rnn.add(Dense(1, activation='sigmoid'))
model_rnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model_rnn.fit(data, np.array(labels), validation_split=0.4, epochs=2)


# RNN + LSTM
model_lstm = Sequential()
model_lstm.add(Embedding(vocabulary_size, 100, input_length=50))
model_lstm.add(CuDNNLSTM(100))
model_lstm.add(Dense(1, activation='sigmoid'))
model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model_lstm.fit(data, np.array(labels), validation_split=0.4, epochs=2)


# RNN + CNN + LSTM
def create_conv_model():
    model_conv = Sequential()
    model_conv.add(Embedding(vocabulary_size, 100, input_length=50))
    model_conv.add(Dropout(0.2))
    model_conv.add(Conv1D(64, 5, activation='relu'))
    model_conv.add(MaxPooling1D(pool_size=4))
    model_conv.add(CuDNNLSTM(100))
    model_conv.add(Dense(1, activation='sigmoid'))
    model_conv.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_conv

model_conv = create_conv_model()
model_conv.fit(data, np.array(labels), validation_split=0.4, epochs = 2)
