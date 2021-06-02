import tensorflow as tf
# tf的hello world
hello = tf.constant("hello world!")
sess = tf.Session()

print(sess.run(hello))