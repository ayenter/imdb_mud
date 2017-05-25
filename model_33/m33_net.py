#TF_CPP_MIN_LOG_LEVEL=1 python keras_lstm_1.py --ssh 

# LSTM and CNN for sequence classification in the IMDB dataset


# -+-+-+-+-+-+-+- GLOBAL VARIABLES -+-+-+-+-+-+-+-

import os
global_model_version = 33
global_batch_size = 32
global_top_words = 5000
global_max_review_length = 500
global_dir_name = os.path.dirname(os.path.realpath(__file__))
global_embedding_vecor_length = 32

global_model_description = "conv(3/4/5x32)[l2(0.01)] -> relu -> maxpool(2) -> dropout(0.5) -> batchnorm -> lstm(50) -> merge(concat) -> dense(1)  [32 batch size]"


# -+-+-+-+-+-+-+- IMPORTS -+-+-+-+-+-+-+-
import sys
sys.path.append('..')
from master import run_model, generate_read_me
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
	model = Sequential()
	input_layer = Embedding(top_words, embedding_vecor_length, input_length=max_review_length)
	model.add(input_layer)
	model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(Dense(250, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	if show_summaries:
		print(model.summary())

	return model


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
parser = argparse.ArgumentParser(description='Sentiment LSTM running through Keras on IMDb movie reviews')
parser.add_argument('-s', dest="show_summaries", action="store_true", default=False, help="Show network summaries")
parser.add_argument('num_epochs', action="store", default=3, help="Number of Epochs", type=int)
inputs = parser.parse_args()
generate_read_me(global_model_version, global_dir_name, global_model_description)
run_model(build_model(global_top_words, global_embedding_vecor_length, global_max_review_length, inputs.show_summaries), global_model_version, global_batch_size, inputs.num_epochs, global_top_words, global_max_review_length, global_dir_name)







