#TF_CPP_MIN_LOG_LEVEL=1 python keras_lstm_1.py --ssh 

# LSTM and CNN for sequence classification in the IMDB dataset


# -+-+-+-+-+-+-+- IMPORTS -+-+-+-+-+-+-+-

import time
import numpy as np
import matplotlib
import argparse
import os
import keras
import csv
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Merge, Input, Reshape
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import plot_model
import matplotlib.pyplot as plt


# -+-+-+-+-+-+-+- CALLBACK -+-+-+-+-+-+-+-

class ExtraHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.batch_acc = []
        self.batch_loss = []
        self.epoch_val_acc = []
        self.epoch_data = []
    def on_batch_end(self, batch, logs={}):
        self.batch_acc.append(logs.get('acc'))
        self.batch_loss.append(logs.get('loss'))
    def on_epoch_end(self, epoch, logs={}):
    	self.epoch_val_acc.append(logs.get('val_acc'))
    	self.epoch_data.append("Epoch: " + str(epoch) + " ==>   loss: " + str(logs.get('loss')) + "  -  acc: " + str(logs.get('acc')) + "  -  val_loss: " + str(logs.get('val_loss')) + "  -  val_acc: " + str(logs.get('val_acc')) + "\n")



# -+-+-+-+-+-+-+- FUNCTIONS -+-+-+-+-+-+-+-

def get_new_name(name):
	if os.path.isfile(name):
		temp = 1
		no_ext = os.path.splitext(name)[0]
		ext = os.path.splitext(name)[1]
		while os.path.isfile(no_ext+str(temp)+ext):
			temp+=1
		name = no_ext+str(temp)+ext
	return name

def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def get_version():
	dir_name = os.path.dirname(os.path.realpath(__file__)).split('/')[-1]
	if dir_name.find('_') != -1 and is_int(dir_name[dir_name.find('_')+1:]):
		return dir_name[dir_name.find('_')+1:]
	else:
		return "V"

def print_time(start, end):
	hours, rem = divmod(end-start, 3600)
	minutes, seconds = divmod(rem, 60)
	print("{:0>2}h {:0>2}m {:05.2f}s ".format(int(hours),int(minutes),seconds))

def plot_epochs(history, batch_history, graph_name):
	epochs = len(history['acc'])
	total_batches = len(batch_history['acc'])
	batches = total_batches/epochs
	x = np.arange(1,epochs+1)
	batch_x = np.arange(1,total_batches+1)/float(batches)
	fig, ax1 = plt.subplots(figsize=(10,5))
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
	plt.savefig(graph_name)
	plt.show()


# -+-+-+-+-+-+-+- ARGUMENTS -+-+-+-+-+-+-+-

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
parser = argparse.ArgumentParser(description='Sentiment LSTM running through Keras on IMDb movie reviews')
parser.add_argument('--ssh', dest="ssh", action="store_true", default=False, help="Change matplotlib back-end for ssh")
parser.add_argument('num_epochs', action="store", default=3, help="Number of Epochs", type=int)
inputs = parser.parse_args()


# -+-+-+-+-+-+-+- SET MODEL VERSION AND NAME -+-+-+-+-+-+-+-

theversion = get_version()
dir_name = os.path.dirname(os.path.realpath(__file__))
version_name = "m" + theversion + "_e" + str(inputs.num_epochs) + "_"
diagram_name = get_new_name(os.path.join(dir_name, "m" + theversion + "_" + "diagram.png"))
graph_name = get_new_name(os.path.join(dir_name, version_name + "graph.png"))
data_name = get_new_name(os.path.join(dir_name, version_name + "data.txt"))
avgs_name = os.path.join(dir_name, "averages.csv")

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
embedding_vecor_length = 32

input_layer = Embedding(top_words, embedding_vecor_length, input_length=max_review_length)

branch_3 = Sequential()
branch_3.add(input_layer)
branch_3.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
branch_3.add(MaxPooling1D(pool_size=2))
branch_3.add(LSTM(100))

branch_4 = Sequential()
branch_4.add(input_layer)
branch_4.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
branch_4.add(MaxPooling1D(pool_size=2))
branch_4.add(LSTM(100))

branch_5 = Sequential()
branch_5.add(input_layer)
branch_5.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
branch_5.add(MaxPooling1D(pool_size=2))
branch_5.add(LSTM(100))

model = Sequential()
model.add(Merge([branch_3,branch_4,branch_5], mode='concat'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("")


# -+-+-+-+-+-+-+- TRAINING MODEL -+-+-+-+-+-+-+-

print("RUNNING MODEL")
extra_hist = ExtraHistory()
start_time = time.time()
hist = model.fit(np.vstack((X_train,X_test)), np.hstack((y_train,y_test)), validation_split=0.5, epochs=inputs.num_epochs, batch_size=64, callbacks=[extra_hist])
end_time = time.time()
print_time(start_time, end_time)
print("")


# -+-+-+-+-+-+-+- RESULTS -+-+-+-+-+-+-+-

print("RESULTS")
# setup for conveying results
avg_acc = sum(extra_hist.epoch_val_acc) / float(len(extra_hist.epoch_val_acc))
batch_history = {}
batch_history.update({'loss':np.asarray(extra_hist.batch_loss)[::10]})
batch_history.update({'acc':np.asarray(extra_hist.batch_acc)[::10]})
# print avg acc
print( "Average Accuracy for " + str(inputs.num_epochs) + " epochs :  " + str(avg_acc*100) + "%" )
print("")


# -+-+-+-+-+-+-+- SAVING RESULTS -+-+-+-+-+-+-+-

print("SAVING MODEL AND RESULTS")
#  -> average
print("Saving avg to " + avgs_name)
with open(data_name, "a") as f:
	writer = csv.writer(f)
	writer.writerow([avg_acc])
#  -> diagram
print("Saving model diagram to " + diagram_name)
plot_model(model, to_file=diagram_name)
#  -> data
print("Saving model results data to " + data_name)
with open(data_name, "wb") as f:
	f.writelines(extra_hist.epoch_data)
	f.write("Average Accuracy for " + str(inputs.num_epochs) + " epochs :  " + str(avg_acc))
#  -> graph
print("Saving model results graph to " + graph_name)
plot_epochs(hist.history, batch_history, graph_name)
