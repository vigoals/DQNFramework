#coding=utf-8
import tensorflow as tf
from netComm import *
from baseAgent import BaseAgent
import numpy as np

class AtariDQN(BaseAgent):
	"""docstring for AtariDQN"""
	def __init__(self, opt):
		super(AtariDQN, self).__init__(opt)
		self.sess = tf.Session()
		self.device = opt.get('device')
		self.histLen = opt.get('histLen', 4)
		self.height = opt.get('height', 84)
		self.width = opt.get('width', 84)
		self.stateDim = self.histLen*self.height*self.width
		self.nActions = opt.get('nActions')
		self.targetFreq = opt.get('targetFreq', 10000)
		self.clipDelta = opt.get('clipDelta', 1)
		self.maxReward = opt.get('maxReward', 1)
		self.minReward = opt.get('minReward', -self.maxReward)

		self.QNetwork = self.createNetwork()
		if self.targetFreq > 0:
			self.QTarget = self.createNetwork()
			self.sess.run(tf.global_variables_initializer())
			self.updateTarget()
		else:
			self.QTarget = self.QNetwork
			self.sess.run(tf.global_variables_initializer())

	def updateTarget(self):
		with tf.device(self.device):
			for i in range(len(self.QNetwork['paras'])):
				tmp = self.QNetwork['paras'][i]
				for j in range(len(tmp)):
					w = self.sess.run(tmp[j])
					op = tf.assign(self.QTarget['paras'][i][j], w)
					self.sess.run(op)

	def createNetwork(self):
		parameters = []
		inputPH = None
		net = None

		with tf.device(self.device):
			l0 = tf.placeholder(tf.float32, [None, self.stateDim])
			inputPH = l0
			l1 = tf.reshape(l0, [-1, self.histLen, self.height, self.width])
			l2 = tf.transpose(l1, [0, 2, 3, 1])
			l3, w, b = conv2d(l2, 32, [8, 8], [4, 4], activation=tf.nn.relu)
			parameters.append((w, b))
			l4, w, b = conv2d(l3, 64, [4, 4], [2, 2], activation=tf.nn.relu)
			parameters.append((w, b))
			l5, w, b = conv2d(l4, 64, [3, 3], [1, 1], activation=tf.nn.relu)
			parameters.append((w, b))

			shape = l5.get_shape().as_list()[1:]
			l6 = tf.reshape(l5,
					[-1, reduce(lambda x, y: x*y, shape)])
			l7, w, b = linear(l6, 512, activation=tf.nn.relu)
			parameters.append((w, b))
			l8, w, b = linear(l7, self.nActions)
			parameters.append((w, b))
			net = l8

		return {'net':net, 'inputPH':inputPH, 'paras':parameters}

	def q(self, state):
		return self.sess.run(self.QNetwork['net'],
				feed_dict={self.QNetwork['inputPH']:state})

	def tq(self, state):
		return self.sess.run(self.QTarget['net'],
				feed_dict={self.QTarget['inputPH']:state})

	def policy(self, state, ep):
		# epsilon-greedy
		if np.random.rand() <= ep:
			return np.random.randint(self.nActions), 0
		else:
			q = self.q(state).reshape(-1)
			action = np.argmax(q)
			return action, q

	def perceive(self, step, observation, reward, terminal):
		pass
