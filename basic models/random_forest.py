import tensorflow as tf
from tensorflow.python.ops import resources
from tensorflow_core.contrib.tensor_forest.python import tensor_forest

from tensorflow_core.examples.tutorials.mnist import input_data
# one_hot是对y进行编码
mnist = input_data.read_data_sets("tmp/data/", one_hot=False)

# 超参
num_steps = 500
batch_size = 1024
num_classes = 10
num_features = 784
num_trees = 10
max_nodes = 1000

X = tf.placeholder(tf.float32, shape=[None, num_features])
y = tf.placeholder(tf.int32, shape=[None])

# Estimator 跟 Dataset 都是 Tensorflow 中的高级API。
# Estimator（评估器）是一种创建 TensorFlow 模型的高级方法，它包括了用于常见机器学习任务的预制模型，当然，你也可以使用它们来创建你的自定义模型。[^3]
# contrib.tensor_forest 详细的实现了随机森林算法（Random Forests）评估器，并对外提供 high-level API。
# 你只需传入 params 到构造器，params 使用 params.fill() 来填充，而不用传入所有的超参数，Tensor Forest 自己的 RandomForestGraphs 就能使用这些参数来构建整幅图。
hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                      num_features=num_features,
                                      num_trees=num_trees,
                                      max_nodes=max_nodes).fill()
forest_graph = tensor_forest.RandomForestGraphs(hparams)
# 得到training graph和loss
train_op = forest_graph.training_graph(X, y)
loss_op = forest_graph.training_loss(X, y)

infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(y,tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_vars = tf.group(tf.global_variables_initializer(),
    resources.initialize_resources(resources.shared_resources()))

sess = tf.Session()

sess.run(init_vars)

# Training
for i in range(1, num_steps + 1):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    _, l = sess.run([train_op, loss_op], feed_dict={X: batch_x, y: batch_y})
    if i % 50 == 0 or i == 1:
        acc = sess.run(accuracy_op, feed_dict={X: batch_x, y: batch_y})
        print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))

# Test Model
test_x, test_y = mnist.test.images, mnist.test.labels
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, y: test_y}))




