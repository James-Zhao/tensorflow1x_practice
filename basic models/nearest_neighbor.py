import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("tmp/data/", one_hot=True)
# 只取5000个样本作为训练集 200个样本作为测试集
X_train, y_train = mnist.train.next_batch(5000)
X_test, y_test = mnist.test.next_batch(200)

x_tr = tf.placeholder("float", [None, 784])
x_te = tf.placeholder("float", [784])

distance = tf.reduce_sum(tf.abs(tf.add(x_tr, tf.negative(x_te))), reduction_indices=1)
pred = tf.argmin(distance, 0)

accuracy = 0

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(len(X_test)):
        nn_index = sess.run(pred, feed_dict={x_tr: X_train, x_te: X_test[i, :]})
        print("Test %i prediction: %i true class: %i" %(i, np.argmax(y_train[nn_index]), np.argmax(y_test[i])))

        if (np.argmax(y_train[nn_index]) == np.argmax(y_test[i])):
            accuracy += 1. /len(X_test)
    print("Done!")
    print("accuracy: %f" %accuracy)