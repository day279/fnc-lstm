from single_dyn_lstm import Single_Directional_LSTM
import json
import numpy as np
import tensorflow as tf

def read_file(filename):
	with open(filename) as data_file:
		data = json.load(data_file)
	features = [np.array(item["sequence"]) for item in data]
	labels = [item["label"] for item in data]

	features = np.array(features)
	features = np.reshape(features, [-1,5])
	labels = np.array(labels)
	labels = np.reshape(labels, [-1,1])

	return features, labels

if __name__ == "__main__":
	data, labels = read_file('dummy_data.json')
	
	# Target log path
	logs_path = './logs'
	writer = tf.summary.FileWriter(logs_path)

	lstm = Single_Directional_LSTM(writer)
	lstm.train(data, labels)