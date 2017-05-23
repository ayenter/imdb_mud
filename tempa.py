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

def load_word2vec(path='../data/GoogleNews-vectors-negative300.bin', binary=True):
	print("Loading Vectors...")
	model = KeyedVectors.load_word2vec_format(path, binary=binary)
	print("Vectors Loaded")
	print("")
	return model