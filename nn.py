import tensorflow as tf
import numpy as np

# datax = 10000
# testx = 5000
# data = np.random.randint(0,2,(datax, 100))
# labels = np.zeros((datax,2))
# test_data = np.random.randint(0,2,(testx, 100))
# test_labels = np.zeros((testx,2))
# for i in range(datax):
# 	if (data[i,0]) == 1:
# 		labels[i,0] = 1
# 		labels[i,1] = 0
# 	else:
# 		labels[i,0] = 0
# 		labels[i,1] = 1
# for i in range(testx):
# 	if (test_data[i,0]) == 1:
# 		test_labels[i,0] = 1
# 		test_labels[i,1] = 0
# 	else:
# 		test_labels[i,0] = 0
# 		test_labels[i,1] = 1


data = np.load('data/training_data.npy')
labels = np.load('data/training_labels.npy')
test_data = np.load('data/testing_data.npy')
test_labels = np.load('data/testing_labels.npy')

#Parameters
training_epochs = 25
learning_rate = .01
batch_size = 256

#Network Parameters
num_hidden = 80
num_classes = 2 #male or female
num_examples = data.shape[0]
num_attributes = data.shape[1]

#Inputs
x = tf.placeholder(tf.float32, shape=(None, num_attributes))
y = tf.placeholder(tf.float32, shape=(None, num_classes))

def multilayer_perceptron(train_data, weights, biases):
	hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
	hidden_layer = tf.sigmoid(hidden_layer)

	out_layer = tf.matmul(hidden_layer, weights['out']) + biases['out']
	return out_layer

weights = {
	'hidden': tf.Variable(tf.random_normal([num_attributes, num_hidden])),
	'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}

biases = {
	'hidden': tf.Variable(tf.random_normal([num_hidden])),
	'out': tf.Variable(tf.random_normal([num_classes]))
}

model = multilayer_perceptron(x, weights, biases)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	#Training
	for epoch in range(training_epochs):
		average_cost = 0.
		total_batch = int(num_examples/batch_size)
		for i in range(total_batch):
			batch_x = data[i*batch_size:(i+1)*batch_size,:]
			batch_y = labels[i*batch_size:(i+1)*batch_size,:]

			_, c = sess.run([optimizer, cost],
				feed_dict = {x: batch_x, y: batch_y})

			# Compute loss
			average_cost += c / total_batch

		if epoch % 10 == 0:
			print("Epoch:" + str(epoch+1) + " cost = " + str(average_cost))
	print("Optimization Finished!")

	correct = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))

	accuracy = tf.reduce_mean(tf.cast(correct, "float"))
	print("Accuracy: " + str(accuracy.eval({x:test_data, y:test_labels})))