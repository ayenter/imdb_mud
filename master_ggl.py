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
from gensim.models import KeyedVectors



# -+-+-+-+-+-+-+- CALLBACK -+-+-+-+-+-+-+-

class ExtraHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.batch_data = {'loss':[], 'acc':[]}
        self.epoch_data = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
    def on_batch_end(self, batch, logs={}):
        self.batch_data['loss'].append(logs.get('loss'))
        self.batch_data['acc'].append(logs.get('acc'))
    def on_epoch_end(self, epoch, logs={}):
    	self.epoch_data['loss'].append(logs.get('loss'))
    	self.epoch_data['acc'].append(logs.get('acc'))
    	self.epoch_data['val_loss'].append(logs.get('val_loss'))
    	self.epoch_data['val_acc'].append(logs.get('val_acc'))



# -+-+-+-+-+-+-+- FUNCTIONS -+-+-+-+-+-+-+-

def load_word2vec(path='../data/GoogleNews-vectors-negative300.bin', binary=True):
	print("Loading Vectors...")
	model = KeyedVectors.load_word2vec_format(path, binary=binary)
	print("Vectors Loaded")
	print("")
	return model

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

def generate_read_me(model_version, dir_name, model_description):
	mv = str(model_version)
	with open(os.path.join(dir_name,"README.md"), "wb") as f:
		f.write('# Model '+ mv)
		f.write('\n')
		f.write('## '+ model_description)
		f.write('\n')
		f.write("![diagram](https://github.com/ayenter/imdb_mud/blob/master/model_"+mv+"/m"+mv+"_diagram.png)")
		f.write('\n')
		f.write("![graph](https://github.com/ayenter/imdb_mud/blob/master/model_"+mv+"/m"+mv+"_r1_e10_graph.png)")


def get_run_version(name):
	run_v = 0
	if os.path.isfile(name):
		data_info = []
		with open(name, "rb") as f:
			reader = csv.reader(f)
			for row in reader:
				data_info.append(row)
		if len(data_info)>1:
			run_v = int(data_info[-1][1])
	else:
		with open(name, "wb") as f:
			writer = csv.writer(f)
			writer.writerow(['model', 'run', '#epochs', 'metric', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10',])
	return run_v+1

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
	ax1.legend(['Batch', 'Train', 'Valid'], loc='lower left')
	ax2.legend(['Batch', 'Train', 'Valid'], loc='upper left')
	plt.savefig(graph_name)
	plt.show()


def run_model(model, model_version, batch_size, num_epochs, top_words, max_review_length, dir_name):

	# -+-+-+-+-+-+-+- ARGUMENTS -+-+-+-+-+-+-+-

	os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


	# -+-+-+-+-+-+-+- SET MODEL VERSION AND NAME -+-+-+-+-+-+-+-

	model_version = str(model_version) #get_version()
	version_name = "m" + model_version + "_"
	epoch_name = "e" + str(num_epochs) + "_"
	diagram_name = os.path.join(dir_name, version_name + "diagram.png")
	data_name = os.path.join(dir_name, version_name + "data.csv")

	run_version = get_run_version(data_name)

	graph_name = os.path.join(dir_name, version_name + "r" + str(run_version) + "_" + epoch_name + "graph.png")


	# -+-+-+-+-+-+-+- DATA PREPROCESSING -+-+-+-+-+-+-+-

	print("")
	print(" >< = >< = >< = >< = >< = >< = >< = >< = >< = >< = >< = >< = >< = >< = >< = >< ")
	print("")
	print("PREPROCESSING DATA")
	# fix random seed for reproducibility
	np.random.seed(7)
	# load the dataset but only keep the top n words, zero the rest
	X_train,y_train,X_test,y_test = get_text_data()
	tokenizer = Tokenizer(nb_words=top_words)
	tokenizer.fit_on_texts(X_train+X_test)
	seq_X_train = tokenizer.texts_to_sequences(X_train)
	seq_X_test = tokenizer.texts_to_sequences(X_test)

	data_X_train = sequence.pad_sequences(seq_X_train, maxlen=max_review_length)
	data_X_test = sequence.pad_sequences(seq_X_test, maxlen=max_review_length)

	word2vec = load_word2vec()

	emb_matrix = np.zeros((len(tokenizer.word_index)+1, 300))
	for w,i in tokenizer.word_index.items():
		if w in word2vec:
			emb_matrix[i] = word2vec[w]


	train_indices = np.arange(data_X_train.shape[0])
	test_indices = np.arange(data_X_test.shape[0])

	np.random.shuffle(train_indices)
	np.random.shuffle(test_indices)

	X_train = data_X_train[train_indices]
	X_test = data_X_test[test_indices]

	y_train = np.asarray(y_train)[train_indices]
	y_test = np.asarray(y_test)[test_indices]
	print("")


	# -+-+-+-+-+-+-+- TRAINING MODEL -+-+-+-+-+-+-+-

	print("RUNNING MODEL")
	extra_hist = ExtraHistory()
	start_time = time.time()
	hist = model.fit(np.vstack((X_train,X_test)), np.hstack((y_train,y_test)), validation_split=0.5, epochs=num_epochs, batch_size=batch_size, callbacks=[extra_hist])
	end_time = time.time()
	print_time(start_time, end_time)
	print("")


	# -+-+-+-+-+-+-+- RESULTS -+-+-+-+-+-+-+-

	print("RESULTS")
	# setup for conveying results
	skip_step = int((float(len(X_train))/batch_size)/39.0625)
	batch_history = {}
	batch_history.update({'loss':np.asarray(extra_hist.batch_data['loss'])[::skip_step]})
	batch_history.update({'acc':np.asarray(extra_hist.batch_data['acc'])[::skip_step]})
	# print val_acc stats
	val_acc = extra_hist.epoch_data['val_acc']
	print("MODEL: " + str(model_version) + "  |  RUN: " + str(run_version) + "  |  #EPOCHS: " + str(num_epochs))
	print("----- ACCURACY -----")
	print("avg: " + str(np.mean(val_acc)))
	print("max: " + str(np.max(val_acc)))
	print("min: " + str(np.min(val_acc)))
	print("")


	# -+-+-+-+-+-+-+- SAVING RESULTS -+-+-+-+-+-+-+-

	print("SAVING MODEL AND RESULTS")
	#  -> data
	print("Saving data to " + data_name)
	with open(data_name, "a") as f:
		writer = csv.writer(f)
		writer.writerow([model_version, run_version, num_epochs, 'loss'] + extra_hist.epoch_data['loss'])
		writer.writerow([model_version, run_version, num_epochs, 'acc'] + extra_hist.epoch_data['acc'])
		writer.writerow([model_version, run_version, num_epochs, 'val_loss'] + extra_hist.epoch_data['val_loss'])
		writer.writerow([model_version, run_version, num_epochs, 'val_acc'] + extra_hist.epoch_data['val_acc'])

	#  -> diagram
	print("Saving model diagram to " + diagram_name)
	plot_model(model, to_file=diagram_name)

	#  -> graph
	print("Saving model results graph to " + graph_name)
	plot_epochs(hist.history, batch_history, graph_name)
	print(" >< = >< = >< = >< = >< = >< = >< = >< = >< = >< = >< = >< = >< = >< = >< = >< ")
	print("")
