#TF_CPP_MIN_LOG_LEVEL=1 python keras_lstm_1.py --ssh 

# LSTM and CNN for sequence classification in the IMDB dataset


# -+-+-+-+-+-+-+- IMPORTS -+-+-+-+-+-+-+-

import time
import numpy as np
import matplotlib
import argparse
import os
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


# -+-+-+-+-+-+-+- CALLBACK -+-+-+-+-+-+-+-

class BatchHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.loss = []
    def on_batch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.loss.append(logs.get('loss'))


# -+-+-+-+-+-+-+- FUNCTIONS -+-+-+-+-+-+-+-

def print_time(start, end):
	hours, rem = divmod(end-start, 3600)
	minutes, seconds = divmod(rem, 60)
	print("{:0>2}h {:0>2}m {:05.2f}s ".format(int(hours),int(minutes),seconds))

def plot_epochs(history, batch_history):
	epochs = len(history['acc'])
	total_batches = len(batch_history['acc'])
	batches = total_batches/epochs
	x = np.arange(1,epochs+1)
	batch_x = np.arange(1,total_batches+1)/float(batches)
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()

	ax1.plot(batch_x, batch_history['acc'], 'g:')
	ax1.plot(x, history['acc'], 'g--', marker='o')
	ax1.plot(x, history['val_acc'], 'g-', marker='o')

	ax2.plot(batch_x, batch_history['loss'], 'r:')
	ax2.plot(x, history['loss'], 'r--', marker='o')
	ax2.plot(x, history['val_loss'], 'r-', marker='o')

	ax1.set_xlabel('Epochs')
	ax1.set_ylabel('Accuracy', color='g')
	ax2.set_ylabel('Loss', color='r')

	plt.margins(.05,.1)
	plt.show()


# -+-+-+-+-+-+-+- ARGUMENTS -+-+-+-+-+-+-+-

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
parser = argparse.ArgumentParser(description='Sentiment LSTM running through Keras on IMDb movie reviews')
parser.add_argument('--ssh', dest="ssh", action="store_true", default=False, help="Change matplotlib back-end for ssh")
inputs = parser.parse_args()


# -+-+-+-+-+-+-+- DATA PREPROCESSING -+-+-+-+-+-+-+-

print("PREPROCESSING DATA")
# fix random seed for reproducibility
np.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
print("")


# -+-+-+-+-+-+-+- BUILDING MODEL -+-+-+-+-+-+-+-

print("BUILDING MODEL")
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
print("")


# -+-+-+-+-+-+-+- TRAINING MODEL -+-+-+-+-+-+-+-

print("TRAINING MODEL")
batch_hist = BatchHistory()
start_time = time.time()
hist = model.fit(np.vstack((X_train,X_test)), np.hstack((y_train,y_test)), validation_split=0.5, epochs=3, batch_size=64, callbacks=[batch_hist])
end_time = time.time()
print_time(start_time, end_time)
batch_history = {}
batch_history.update({'loss':np.asarray(batch_hist.loss)[::10]})
batch_history.update({'acc':np.asarray(batch_hist.acc)[::10]})
print("")


# -+-+-+-+-+-+-+- EVALUATION AND PLOTTING -+-+-+-+-+-+-+-

print("EVALUATION AND PLOTTING")
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# import plt incase of 
if inputs.ssh:
	matplotlib.use('GTK')
import matplotlib.pyplot as plt

plot_epochs(hist.history, batch_history)
