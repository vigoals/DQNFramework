#coding=utf-8

import tensorflow as tf
import numpy as np

def getWeights(shape, stddev=0.01, dtype=tf.float32, name='weight'):
	return tf.Variable(
			tf.truncated_normal(shape, stddev=stddev, dtype=dtype), name=name)

def getBias(shape, stddev=0.01, dtype=tf.float32, name='bias'):
	return tf.Variable(
			tf.truncated_normal(shape, stddev=stddev, dtype=dtype), name=name)

def conv2d(input_, outputDim, kernelSize,
		strides, stddev=None, activation=None, name='conv2d'):
	inputDim = input_.get_shape().as_list()[-1]

	if stddev is None:
		stddev = 1/np.sqrt(kernelSize[0]*kernelSize[1]*inputDim)

	with tf.name_scope(name):
		weight = getWeights(
				[kernelSize[0], kernelSize[1], inputDim, outputDim], stddev)
		strides = [1, strides[0], strides[1], 1]
		conv = tf.nn.conv2d(input_, weight, strides, 'SAME', data_format='NHWC')
		bias = getBias([outputDim], stddev)
		out = tf.nn.bias_add(conv, bias, 'NHWC')
		# out = conv
		if activation is not None:
			out = activation(out)
		return out, weight, bias

def linear(input_, outputSize, stddev=None, activation=None, name='linear'):
	shape = input_.get_shape().as_list()

	if stddev is None:
		stddev = 1/np.sqrt(shape[1])

	with tf.name_scope(name):
		weight = getWeights([shape[1], outputSize], stddev)
		bias = getBias([outputSize], stddev)

		if activation is not None:
			return activation(tf.matmul(input_, weight) + bias), weight, bias
			# return activation(tf.nn.bias_add(tf.matmul(input_, weight), bias)), \
			# 		weight, bias
		else:
			return tf.matmul(input_, weight) + bias, weight, bias
			# return tf.nn.bias_add(tf.matmul(input_, weight), bias), weight, bias
