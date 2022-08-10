# cython: language_level=3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

reverse_module = tf.load_op_library('./reverse_op.so')
reversed = reverse_module.reverse_it([[1, 2, 3], [4, 5, 6]])
print(reversed)

