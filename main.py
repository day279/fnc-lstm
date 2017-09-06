from single_dyn_lstm import Single_Directional_LSTM
import json
import numpy as np
import tensorflow as tf

def read_file(filename, feature_size):
	with open(filename) as data_file:
		data = json.load(data_file)
	features = [np.array(item["sequence"]) for item in data]
	labels = [item["label"] for item in data]

	labels = np.reshape(labels, [-1,1])

	#print("\nFirst:")
	#print(features)
	features = np.reshape(features, [len(labels), -1, feature_size])
	#print("\nSecond:")
	#print(features)

	return features, labels

if __name__ == "__main__":
	feature_size = 3
	data, labels = read_file('dummy_data_fixed.json', feature_size)

	# Target log path
	logs_path = './logs'
	writer = tf.summary.FileWriter(logs_path)

	lstm = Single_Directional_LSTM(writer, feature_size)
	lstm.train(data, labels)