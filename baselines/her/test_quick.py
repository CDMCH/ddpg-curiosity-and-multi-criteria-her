import tensorflow as tf

with tf.Session() as sess:
    a = tf.constant(4)
    print(sess.run([a]))