import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import random

class Single_Directional_LSTM(object):
	def __init__(self, writer, feature_size):
		self.writer = writer
		self.hidden_size = 10
		self.output_size = 2
		self.feature_size = feature_size
		self.learning_rate = 0.001
		self.training_iters = 5
		self.batch_size = 1
		self.sent_len = 5
		self.max_seq_len = 6

	def variable_summaries(self, var):
		with tf.name_scope("summaries"):
			mean = tf.reduce_mean(var)
		tf.summary.scalar("mean", mean)
		with tf.name_scope("stddev"):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar("stddev", stddev)
		tf.summary.scalar("max", tf.reduce_max(var))
		tf.summary.scalar("min", tf.reduce_min(var))
		tf.summary.histogram("histogram", var)


	def build_model(self):
		x = tf.placeholder(tf.float32, [self.batch_size, self.max_seq_len, self.feature_size], name="input_features")
		y = tf.placeholder(tf.float32, [self.batch_size, self.output_size], name="gold_labels")

		weights = tf.Variable(tf.random_normal([self.hidden_size, self.output_size], dtype=tf.float32), name="weights")
		self.variable_summaries(weights)
		biases = tf.Variable(tf.random_normal([self.output_size], dtype=tf.float32), name="biases")
		self.variable_summaries(biases)

		lstm = rnn.BasicLSTMCell(self.hidden_size)

		# Split into single-step tensors along axis=1 with shape [self.batch_size, self.feature_size]
		#print(x.get_shape())
		#print(tf.shape(x))
		#x_split = tf.unstack(x, num=seq_len, axis=1)
		x_split = tf.split(x, self.max_seq_len, 1)
		x_split = [tf.squeeze(t, [1]) for t in x_split]


		# TODO change back to dynamic_rnn
		#self.outputs, self.states = tf.nn.dynamic_rnn(lstm, input_features, initial_state=self.initial_state)
		# x_split must be list of 2-D tensors [batch_size, input_size]
		outputs, states = rnn.static_rnn(lstm, x_split, dtype=tf.float32)
		tf.summary.histogram("outputs", outputs)
		tf.summary.histogram("states", states)

		return (x, y, tf.matmul(outputs[-1], weights) + biases)

	def train(self, data, gold):
		num_examples = len(gold)
		#data = np.reshape(data, [num_examples, -1, self.feature_size])

		x, y, pred = self.build_model()

		with tf.name_scope("cross_entropy"):
			cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

		with tf.name_scope("loss_optimizer"):
			optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(cost)

		correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))

		with tf.name_scope("accuracy"):
			accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		tf.summary.scalar("cross_entropy_scl", cost)
		tf.summary.scalar("training_accuracy", accuracy)

		summarize_all = tf.summary.merge_all()

		init = tf.global_variables_initializer()

		with tf.Session() as session:
			session.run(init)
			step = 0

			self.writer.add_graph(session.graph)

			while step < self.training_iters:
				i = random.randint(0, num_examples - 1)

				features_in = np.array(data[i])
				features_in = np.reshape(features_in, [self.batch_size, -1, self.feature_size])
				features_in = np.pad(features_in, [[0,0],[0,self.max_seq_len - features_in.shape[1]],[0,0]], 'constant')
				dyn_seq_len = np.array([len(features_in)])

				labels_out_onehot = np.zeros([self.output_size], dtype=float)
				labels_out_onehot[gold[i]] = 1.0
				labels_out_onehot = np.reshape(labels_out_onehot, [self.batch_size, -1])

				_, acc, loss, onehot_pred, summary = session.run(
					[optimizer, accuracy, cost, pred, summarize_all], 
					feed_dict={x: features_in, y:labels_out_onehot}
					)

				self.writer.add_summary(summary, i)
				print("Iter= " + str(step+1) + ", Accuracy= " + str(acc) + ", Loss= " + str(loss))
				step += 1
