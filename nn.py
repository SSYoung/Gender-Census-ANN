import tensorflow as tf
import numpy as np
from CrossValidate import split_training_data

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
training_epochs = 20
learning_rate = .01
batch_size = 1024

#Network Parameters
num_hidden = 40
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

alphas = [0.01, 0.001, 0.0001]
hidden_layers = [2, 20, 40]
batch_sizes = [256, 1024, 2048]
k = 5


with tf.Session() as sess:
	sess.run(init)

	######### CROSS VALIDATION
	# avg_training_scores = np.zeros((len(alphas), len(batch_sizes)))
	# for a_index in range(len(alphas)):
	# 	alpha = alphas[a_index]
	# 	for b_index in range(len(batch_sizes)):
	# 		batch_size = batch_sizes[b_index]
	# 		batch_scores = np.zeros(k)
	# 		print('Grid Search with:\n\talpha: ' + str(alpha) + '\n\tBatch Size: ' + str(batch_size))
	# 		cross = 0
	# 		for X_train, y_train, X_test, y_test in split_training_data(data, labels):
	# 			sess.run(init)
	# 			# X_train, y_train, X_test, y_test = split_training_data(data, labels)
	# 			num_examples = X_train.shape[0]
	# 			epoch = 0
	# 			delta_training_cost,training_cost = 1,1

	# 			while epoch < training_epochs and np.abs(delta_training_cost) > 0.001:
	# 				training_cost_prev = training_cost
	# 				average_cost = 0.
	# 				total_batch = int(num_examples/batch_size)
	# 				for i in range(total_batch):
	# 					batch_x = X_train[i*batch_size:(i+1)*batch_size,:]
	# 					batch_y = y_train[i*batch_size:(i+1)*batch_size,:]

	# 					_, c = sess.run([optimizer, cost],
	# 						feed_dict = {x: batch_x, y: batch_y})

	# 					# Compute loss
	# 					average_cost += c / total_batch

	# 				if epoch % 10 == 0:
	# 					print("Epoch:" + str(epoch+1) + " cost = " + str(average_cost))
	# 				epoch += 1

	# 				training_cost = average_cost
	# 				delta_training_cost = training_cost - training_cost_prev

	# 			correct = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
	# 			accuracy = tf.reduce_mean(tf.cast(correct, "float"))
	# 			score = accuracy.eval({x:X_test, y:y_test})

	# 			print('\tFold ' + str(cross + 1) + ' of ' + str(k) + ' done. Accuracy: ' + str(score))

	# 			batch_scores[cross] = score
	# 			cross += 1

	# 		avg_training_scores[a_index,b_index] = np.mean(batch_scores)
	# print('GRID SEARCH SCORES WITH K=5')
	# print(avg_training_scores)

	######### Training
	correct = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct, "float"))

	N = 10
	training_error = np.zeros(N)
	testing_error = np.zeros(N)

	for iteration in range(N):
		print('Training Sample ', iteration)
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
				train = accuracy.eval({x:data, y:labels})
				test = accuracy.eval({x:test_data, y:test_labels})
				print(epoch, train, test)

		testing_error[iteration] = accuracy.eval({x:test_data, y:test_labels})

	print(testing_error)
	print(' '.join(str(round(a,4)) for a in testing_error))





	# print("Optimization Finished!")

	# correct = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))

	# accuracy = tf.reduce_mean(tf.cast(correct, "float"))
	# print("Accuracy: " + str(accuracy.eval({x:test_data, y:test_labels})))