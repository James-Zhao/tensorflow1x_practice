import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100

n_hidden_1 = 256
n_hidden_2 = 256
num_inputs = 784
num_classes = 10

X = tf.placeholder("float", [None, num_inputs])
y = tf.placeholder("float", [None, num_classes])

weights = {
    "h1": tf.Variable(tf.random_normal([num_inputs, n_hidden_1])),
    "h2": tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    "out": tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}

biases = {
    "b1": tf.Variable(tf.random_normal([n_hidden_1])),
    "b2": tf.Variable(tf.random_normal([n_hidden_2])),
    "out": tf.Variable(tf.random_normal([num_classes]))
}

# 定义模型
def neural_net(x):
    # fully connected layer
    layer_1 = tf.add(tf.matmul(x, weights["h1"]), biases["b1"])
    layer_2 = tf.add(tf.matmul(layer_1, weights["h2"]), biases["b2"])
    out_layer = tf.add(tf.matmul(layer_2, weights["out"]), biases["out"])
    return out_layer
logits = neural_net(X)

# 定义loss和optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)
    for step in range(1, num_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_op, feed_dict={X: batch_x, y: batch_y})
        if (step % display_step == 0 or step == 1):
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
                  sess.run(accuracy, feed_dict={X: mnist.test.images,
                                                y: mnist.test.labels}))

