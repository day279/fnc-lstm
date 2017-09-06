from single_dyn_lstm import Single_Directional_LSTM
import json
import numpy as np
import tensorflow as tf

def read_file(filename, feature_size):
	with open(filename) as data_file:
		data = json.load(data_file)

	labels = [item["label"] for item in data]
	features = [np.array(item["sequence"]) for item in data]

	labels = np.reshape(labels, [-1,1])
	features = np.reshape(features, [len(labels), -1, feature_size])

	return features, labels

if __name__ == "__main__":
	feature_size = 3
	data, labels = read_file('dummy_data_fixed.json', feature_size)
	max_seq_len = max([len(item) for item in data])

	# Target log path
	logs_path = './logs'
	writer = tf.summary.FileWriter(logs_path)

	lstm = Single_Directional_LSTM(writer, feature_size, max_seq_len)
	lstm.train(data, labels)