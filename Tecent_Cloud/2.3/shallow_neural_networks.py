import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def add_layer(inputs, in_size, out_size, activation_function=None):
    w = tf.Variable(tf.random_normal(shape=[in_size, out_size], stddev=0.1)) # stddev=1
    b = tf.Variable(tf.zeros([1, out_size]) + 0.01)

    Z = tf.matmul(inputs, w) + b
    if activation_function is None:
        outputs = Z
    else:
        outputs = activation_function(Z)

    return outputs

if __name__ == "__main__":

    MNIST = input_data.read_data_sets("./", one_hot=True)

    learning_rate = 0.2 # 0.05
    batch_size = 128
    n_epochs = 10

    X = tf.placeholder(tf.float32, shape=[batch_size, 784])
    Y = tf.placeholder(tf.float32, shape=[batch_size, 10])

    l1 = add_layer(X, 784, 1000, activation_function=tf.nn.relu)
    prediction = add_layer(l1, 1000, 10, activation_function=None)

    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=prediction)
    loss = tf.reduce_mean(entropy)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        n_batches = int(MNIST.train.num_examples/batch_size)
        for i in range(n_epochs):
            for j in range(n_batches):
                X_batch, Y_batch = MNIST.train.next_batch(batch_size)
                _, _loss = sess.run([optimizer, loss], feed_dict={X:X_batch, Y:Y_batch})
                if j is 0:
                    print "Loss of epoch [{0}] batch [{1}]: {2}".format(i, j, _loss)

        # test mode
        n_batches = int(MNIST.test.num_examples/batch_size)
        total_correct = 0
        for i in range(n_batches):
            X_batch, Y_batch = MNIST.test.next_batch(batch_size)
            pred_y = sess.run(prediction, feed_dict={X:X_batch, Y:Y_batch})
            pred_correct = tf.equal(tf.argmax(Y_batch,1), tf.argmax(pred_y, 1))
            n_correct = tf.reduce_sum(tf.cast(pred_correct, tf.float32))

            total_correct += sess.run(n_correct)
        print "Accuracy: {0}".format(total_correct/MNIST.test.num_examples)





