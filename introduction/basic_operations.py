import tensorflow as tf

# 常量
a = tf.constant(2)
b = tf.constant(3)

with tf.Session() as sess:
    print("a = 2, b = 3")
    print("addition with constants: %i" % sess.run(a + b))
    print("multiplication with constants: %i" % sess.run(a * b))

# 变量
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
    print("addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print("multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))

# 矩阵
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])

product = tf.matmul(matrix1, matrix2)
with tf.Session() as sess:
    print(sess.run(product))