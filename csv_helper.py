import os, csv
import numpy as np
import matplotlib.pyplot as plt

def get_csv():
	lst = []
	full_csv = []
	for root, dirs, files in os.walk("."):
		for file in files:
			if file.endswith(".csv"):
				lst.append(os.path.join(root, file))
	for fname in lst:
		with open(fname,"rb") as f:
			reader = csv.reader(f)
			for row in reader:
				if row[3] == 'val_acc':
					temp = []
					for i in range(14):
						if i != 3:
							if i<len(row):
								temp.append(float(row[i]))
							else:
								temp.append(0)
					temp2 = [max(temp[3:])]
					full_csv.append(temp)
	return np.asarray(full_csv)

def plot_csv():
	full_csv = np.asarray(get_csv())
	data = full_csv[:,3:]
	v_maxes = np.amax(data, axis=1)
	i_maxes = np.arange(len(data))
	maxes = np.swapaxes(np.vstack((i_maxes,v_maxes)),0,1)
	top_idx = maxes[maxes[:,1].argsort()][::-1][:8][:,0].astype('int')
	data = data[top_idx]

	for d in data:
		temp = []
		for i in d:
			if i!=0:
				if i>.80:
					temp.append(i)
				else:
					temp.append(.80)
		plt.plot(temp)

	plt.legend(full_csv[top_idx][:,0], loc='lower right')
	plt.show()


plot_csv()

"""
full_csv = np.asarray(get_csv())
	i_max = full_csv.argmax(axis=0)[3:]
	v_max = np.amax(full_csv,axis=0)[3:]
	m_max = np.asarray([full_csv[i][0] for i in i_max]).astype('int')
	stats = np.swapaxes(np.vstack((i_max,v_max)),0,1)
"""