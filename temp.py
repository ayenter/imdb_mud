import time
import numpy as np
import matplotlib
import argparse
import os
import keras
import csv
import progressbar
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Merge, Input, Reshape, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras.regularizers import l2
from keras.preprocessing.text import Tokenizer

def get_text_data(path='../data'):
	print("Grabbing Data from [train/pos, train/neg, test/pos, test/neg]")
	# sets of data
	X_train = []
	y_train = []
	X_test = []
	y_test = []
	train_pos = os.path.join(path,"train","pos")
	train_neg = os.path.join(path,"train","neg")
	test_pos = os.path.join(path,"test","pos")
	test_neg = os.path.join(path,"test","neg")
	all_dict = [{'dir':train_pos, 'X_set':X_train, 'y_set':y_train, 'label':1},
				{'dir':train_neg, 'X_set':X_train, 'y_set':y_train, 'label':0},
				{'dir':test_pos, 'X_set':X_test, 'y_set':y_test, 'label':1},
				{'dir':test_neg, 'X_set':X_test, 'y_set':y_test, 'label':0}]
	for the_dict in all_dict:
		bar = progressbar.ProgressBar()
		for fname in bar(sorted(os.listdir(the_dict['dir']))):
			with open(os.path.join(the_dict['dir'],fname), "rb") as f:
				the_dict['X_set'].append(f.read().replace("<br />",""))
				the_dict['y_set'].append(the_dict['label'])
	return [X_train, y_train, X_test, y_test]