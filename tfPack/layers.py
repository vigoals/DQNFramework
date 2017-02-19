#coding=utf-8
import tensorflow as tf
import tfComm as tfc

class Layers(object):
	"""基本网络层"""
	def __init__(self, lastLayer=None):
		self.weights = None
		self.bias = None
		self.output = None
		self.lastLayer = lastLayer

class Placeholder(Layers):
	"""Placeholder"""
	def __init__(self, size, dtype=tf.float32):
		super(Placeholder, self).__init__()
		with tf.device(tfc.DEVICE):
			self.output = tf.placeholder(dtype, size)

class Linear(Layers):
	"""线性层"""
	def __init__(self, lastLayer, ind, outd, dtype=tf.float32):
		super(Linear, self).__init__(lastLayer)
		with tf.device(tfc.DEVICE):
			self.weights = tf.Variable(
					tf.truncated_normal([ind, outd], stddev=0.01))
			self.bias = tf.Variable(
					tf.truncated_normal([outd], stddev=0.01))
			self.output = tf.matmul(lastLayer.output, self.weights) + self.bias

class MSELoss(Layers):
	"""MSE"""
	def __init__(self, lastLayer, target, dtype=tf.float32):
		super(MSELoss, self).__init__(lastLayer)
		with tf.device(tfc.DEVICE):
			self.output = tf.reduce_mean(
					tf.square(lastLayer.output - target.output))
