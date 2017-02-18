#coding=utf-8
import tensorflow as tf
from layers import *

class Network(object):
	"""管理网路结构"""
	def __init__(self, network, dtype=tf.float32):
		self.dtype = tf.float32
		self.network = network
		self.layers = []
		self._create_network()
		self.target = None

	def _create_network(self):
		lastLayer = None
		for l in self.network:
			if l['type'] == 'placeholder':
				self.layers.append(Placeholder(l['size'], self.dtype))
			elif l['type'] == 'linear':
				self.layers.append(Linear(lastLayer, l['input'], l['output'], self.dtype))
			elif l['type'] == 'output':
				self.target = Placeholder(l['size'], self.dtype)
			elif l['type'] == 'loss':
				pass
			elif l['type'] == 'optimizer':
				pass
			else:
				assert False, "无法处理的层"

			lastLayer = self.layers[len(self.layers) - 1]

	def _create_loass(self, l, lastLayer):
		pass

	def _create_optimizer(self, l):
		pass

	# def create_network(network):
	# 	pass
	#
	# def create_loss(loss):
	# 	pass
