import tensorflow as tf
import numpy as np

tf.reset_default_graph()


a = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='placeholder_a')
b = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='placeholder_b')
c = tf.placeholder(dtype=tf.float32, shape=[None,1], name='placeholder_c')
d = tf.multiply(c,20, name='multop')
e = tf.reduce_sum(d)

with tf.name_scope('NN'):
    layer_1 = tf.layers.dense(inputs=a, units=100, name='layer1')
    layer_2 = tf.layers.dense(inputs=layer_1, units=50, name='layer2')
    layer_3 = tf.layers.dense(inputs=layer_2, units=1, name='layer3')

mean = tf.reduce_mean(layer_3)

mean_value = tf.summary.scalar(name='mean', tensor=mean)
# tf.summary.scalar(name='placeholder_c_summary', tensor=c)
sum_value = tf.summary.scalar(name='sum', tensor=e)
summary_merged = tf.summary.merge_all()


with tf.Session() as sess:
    writer = tf.summary.FileWriter('/Users/stefruinard/Desktop/AI/A3C/graphs/', sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        a_values = np.random.randn(20, 3)
        b_values = np.random.randn(20, 3)
        c_values = np.random.randn(20, 1)
        feed_dict = {a: a_values, b: b_values, c: c_values}
        summary_run = sess.run(summary_merged, feed_dict=feed_dict)
        writer.add_summary(summary_run, global_step=i)
    writer.flush()