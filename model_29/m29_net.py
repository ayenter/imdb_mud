#TF_CPP_MIN_LOG_LEVEL=1 python keras_lstm_1.py --ssh 

# LSTM and CNN for sequence classification in the IMDB dataset


# -+-+-+-+-+-+-+- GLOBAL VARIABLES -+-+-+-+-+-+-+-

import os
global_model_version = 29
global_batch_size = 32
global_top_words = 5000
global_max_review_length = 500
global_dir_name = os.path.dirname(os.path.realpath(__file__))
global_embedding_vecor_length = 32


# -+-+-+-+-+-+-+- IMPORTS -+-+-+-+-+-+-+-
import sys
sys.path.append('..')
from master import run_model
import time
import numpy as np
import matplotlib
import argparse
import keras
import csv
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Merge, Input, Reshape, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras.regularizers import l2


def build_model(top_words, embedding_vecor_length, max_review_length, show_summaries=False):
	input_layer = Embedding(top_words, embedding_vecor_length, input_length=max_review_length)

	# --- 3 ---
	branch_3 = Sequential()
	branch_3.add(input_layer)
	branch_3.add(Conv1D(filters=32, kernel_size=3, padding='same', kernel_regularizer=l2(.01)))
	branch_3.add(Activation('relu'))

	branch_3_1 = Sequential()
	branch_3_1.add(branch_3)
	branch_3_1.add(MaxPooling1D(pool_size=500))

	branch_3_2 = Sequential()
	branch_3_2.add(branch_3)
	branch_3_2.add(MaxPooling1D(pool_size=2))
	branch_3_2.add(Dropout(0.5))
	branch_3_2.add(BatchNormalization())
	branch_3_2.add(LSTM(100))

	# --- 4 ---
	branch_4 = Sequential()
	branch_4.add(input_layer)
	branch_4.add(Conv1D(filters=32, kernel_size=4, padding='same', kernel_regularizer=l2(.01)))
	branch_4.add(Activation('relu'))

	branch_4_1 = Sequential()
	branch_4_1.add(branch_4)
	branch_4_1.add(MaxPooling1D(pool_size=500))

	branch_4_2 = Sequential()
	branch_4_2.add(branch_4)
	branch_4_2.add(MaxPooling1D(pool_size=2))
	branch_4_2.add(Dropout(0.5))
	branch_4_2.add(BatchNormalization())
	branch_4_2.add(LSTM(100))

	# --- 5 ---
	branch_5 = Sequential()
	branch_5.add(input_layer)
	branch_5.add(Conv1D(filters=32, kernel_size=5, padding='same', kernel_regularizer=l2(.01)))
	branch_5.add(Activation('relu'))

	branch_5_1 = Sequential()
	branch_5_1.add(branch_5)
	branch_5_1.add(MaxPooling1D(pool_size=500))

	branch_5_2 = Sequential()
	branch_5_2.add(branch_5)
	branch_5_2.add(MaxPooling1D(pool_size=2))
	branch_5_2.add(Dropout(0.5))
	branch_5_2.add(BatchNormalization())
	branch_5_2.add(LSTM(100))


	model_1 = Sequential()
	model_1.add(Merge([branch_3_1,branch_4_1,branch_5_1], mode='concat'))
	model_1.add(Flatten())
	model_1.add(Dropout(0.5))
	model_1.add(Dense(300, activation='sigmoid'))

	model = Sequential()
	model.add(Merge([branch_3_1,branch_4_1,branch_5_1, branch_3_2, branch_4_2, branch_5_2], mode='concat'))
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	if show_summaries:
		print(branch_3.summary())
		print(branch_4.summary())
		print(branch_5.summary())
		print(model.summary())

	return model


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
parser = argparse.ArgumentParser(description='Sentiment LSTM running through Keras on IMDb movie reviews')
parser.add_argument('-s', dest="show_summaries", action="store_true", default=False, help="Show network summaries")
parser.add_argument('num_epochs', action="store", default=3, help="Number of Epochs", type=int)
inputs = parser.parse_args()
run_model(build_model(global_top_words, global_embedding_vecor_length, global_max_review_length, inputs.show_summaries), global_model_version, global_batch_size, inputs.num_epochs, global_top_words, global_max_review_length, global_dir_name)







