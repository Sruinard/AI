import tensorflow as tf 
import numpy as np

print('hello')
placeholder = tf.placeholder(dtype=tf.float32, shape=[None,5])
layer = tf.layers.dense(inputs=placeholder, units=100)     
a = 12
b = 33

def multiply(x,y):
    return x*y

c = multiply(a,b)
multiplied = multiply(c,b)

