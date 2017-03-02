#coding=utf-8


#coding=utf-8
import tensorflow as tf
from netComm import *
from baseAgent import BaseAgent
import numpy as np
import gameBuf

class DQN(BaseAgent):
	def __init__(self):
		super(AtariDQN, self).__init__(opt)
		# GPU 不会全部占用
		config = tf.ConfigProto()
		# config.log_device_placement = True
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config)
		self.device = opt.get('device')
		self.stateDim = opt.get('stateDim')
		self.nActions = opt.get('nActions')
		# self.clipDelta = opt.get('clipDelta')
		# self.maxReward = opt.get('maxReward')
		# self.minReward = opt.get('minReward')
		self.learningRate = opt.get('learningRate', 0.001)
		self.learnStart = opt.get('learnStart', 500)
		self.discount = opt.get('discount', 0.7)
		self.trainFreq = opt.get('trainFreq', 1)
		self.targetFreq = opt.get('targetFreq', 1000)
		self.batchSize = opt.get('batchSize', 32)
		self.evalBatchSize = opt.get('evalBatchSize', 100)

		self.gameBuf = gameBuf.BaseBuf(opt)
		self.evalBuf = None
		# self.evalBuf = gameBuf.BaseBuf(opt)
		# self.evalBuf.reset(2)
		self.step = None

		with tf.device(self.device):
			self.QNetwork = Network(self.stateDim,
					self.nActions, linearLayers=(512, 512),
					sess=self.sess, name='QNetwork')

			if targetFreq > 0:
				self.QTarget = Network(self.stateDim,
					self.nActions, linearLayers=(512, 512),
					sess=self.sess, name='QTarget')
			else:
				self.QTarget = self.QNetwork

			self.optimizer = Optimizer(self.QTarget, self.learningRate,
					self.nActions, self.clipDelta)

			self.sess.run(tf.global_variables_initializer())
			self.updateTarget()

	def updateTarget(self):
		paras = self.QNetwork.getParas()
		self.QTarget.setParas(paras)

	def q(self, state):
		state = state.reshape([-1, self.stateDim])
		return self.QNetwork.forward(state)

	def tq(self, state):
		state = state.reshape([-1, self.stateDim])
		return self.QTarget.forward(state)

	def policy(self, state, ep):
		# epsilon-greedy
		if np.random.rand() <= ep:
			return np.random.randint(self.nActions), 0
		else:
			q = self.q([state]).reshape(-1)
			action = np.argmax(q)
			return action, q
