import tensorflow as tf

# tf的hello world
hello = tf.constant("Hello, Tensorflow!")
sess = tf.Session()

print(sess.run(hello))