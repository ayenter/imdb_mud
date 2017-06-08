#TF_CPP_MIN_LOG_LEVEL=1 python keras_lstm_1.py --ssh 

# LSTM and CNN for sequence classification in the IMDB dataset


# -+-+-+-+-+-+-+- GLOBAL VARIABLES -+-+-+-+-+-+-+-

import os
global_model_version = 67
global_batch_size = 128
global_top_words = 5000
global_max_review_length = 500
global_dir_name = os.path.dirname(os.path.realpath(__file__))
global_embedding_vecor_length = 32

global_model_description = "conv(3/5/7/9x128)[l2(0.01)] -> relu -> maxpool(2) -> dropout(0.5) -> batchnorm -> lstm(128) -> merge(concat) -> dense(1)  [ 128 batch size ]"


# -+-+-+-+-+-+-+- IMPORTS -+-+-+-+-+-+-+-
import sys
sys.path.append('..')
from master import run_model, generate_read_me, get_text_data, load_word2vec
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
	branch_3.add(Conv1D(filters=128, kernel_size=3, padding='same', kernel_regularizer=l2(.01)))
	branch_3.add(Activation('relu'))
	branch_3.add(MaxPooling1D(pool_size=2))
	branch_3.add(Dropout(0.5))
	branch_3.add(BatchNormalization())
	branch_3.add(LSTM(128))

	# --- 5 ---
	branch_5 = Sequential()
	branch_5.add(input_layer)
	branch_5.add(Conv1D(filters=128, kernel_size=5, padding='same', kernel_regularizer=l2(.01)))
	branch_5.add(Activation('relu'))
	branch_5.add(MaxPooling1D(pool_size=2))
	branch_5.add(Dropout(0.5))
	branch_5.add(BatchNormalization())
	branch_5.add(LSTM(128))

	# --- 7 ---
	branch_7 = Sequential()
	branch_7.add(input_layer)
	branch_7.add(Conv1D(filters=128, kernel_size=7, padding='same', kernel_regularizer=l2(.01)))
	branch_7.add(Activation('relu'))
	branch_7.add(MaxPooling1D(pool_size=2))
	branch_7.add(Dropout(0.5))
	branch_7.add(BatchNormalization())
	branch_7.add(LSTM(128))

	# --- 9 ---
	branch_9 = Sequential()
	branch_9.add(input_layer)
	branch_9.add(Conv1D(filters=128, kernel_size=9, padding='same', kernel_regularizer=l2(.01)))
	branch_9.add(Activation('relu'))
	branch_9.add(MaxPooling1D(pool_size=2))
	branch_9.add(Dropout(0.5))
	branch_9.add(BatchNormalization())
	branch_9.add(LSTM(128))


	model = Sequential()
	model.add(Merge([branch_3,branch_5,branch_7,branch_9], mode='concat'))
	model.add(Dense(1, activation='sigmoid'))
	opt = keras.optimizers.RMSprop(lr=0.01, decay=0.08)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

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

generate_read_me(global_model_version, global_dir_name, global_model_description)
run_model(build_model(global_top_words, global_embedding_vecor_length, global_max_review_length, inputs.show_summaries), global_model_version, global_batch_size, inputs.num_epochs, global_top_words, global_max_review_length, global_dir_name)







