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
		self.stateLow = np.array(opt.get('stateLow'))
		self.stateHigh = np.array(opt.get('stateHigh'))
		self.nActions = opt.get('nActions')
		self.clipDelta = opt.get('clipDelta')
		self.maxReward = opt.get('maxReward')
		self.minReward = opt.get('minReward',
				-self.maxReward if self.maxReward is not None else None)
		self.learningRate = opt.get('learningRate', 0.001)
		self.learnStart = opt.get('learnStart', 500)
		self.discount = opt.get('discount', 0.7)
		self.trainFreq = opt.get('trainFreq', 1)
		self.targetFreq = opt.get('targetFreq', 1000)
		self.batchSize = opt.get('batchSize', 32)
		self.evalBatchSize = opt.get('evalBatchSize', 100)

		self.gameBuf = gameBuf.BaseBuf(opt)
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

	def preprocess(self, step, observation, reward, terminal, eval_):
		if not eval_ and step > 0:
			r = reward
			r = min(r, self.maxReward) if self.maxReward is not None else r
			r = max(r, self.minReward) if self.minReward is not None else r
			self.gameBuf.setReward(r)
		return state

	def perceive(self, step, observation, reward, terminal, ep, eval_):

		screen = self.preprocess(step, observation, reward, terminal, eval_)
		state = None
		if not eval_:
			self.gameBuf.add(step, screen, terminal)
			self.step = step
			state = self.gameBuf.getState()
		else:
			self.evalBuf.add(step, screen, terminal)
			state = self.evalBuf.getState()

		action, q = self.policy(state, ep)

		if not eval_:
			self.gameBuf.setAction(action)

			# train
			if step > self.learnStart \
					and self.trainFreq > 0 \
					and step%self.trainFreq == 0:
				self.train()

			# update target
			if step > self.learnStart \
					and self.targetFreq > 0 \
					and step%self.targetFreq == 0:
				self.updateTarget()

		return action, q

	# def trainerRun(self, state, targets, action):
	# 	deltas, _ = self.sess.run(
	# 			(self.trainer['deltas'], self.trainer['updateGrads']),
	# 			feed_dict={self.QNetwork['inputPH'] : state,
	# 			self.trainer['targetsPH'] : targets,
	# 			self.trainer['actionPH'] : action})
	#
	# 	return np.abs(deltas).mean()

	def computTargets(self, batch):
		state = batch['state']
		action = batch['action']
		reward = batch['reward']
		terminal = batch['terminal']
		stateNext = batch['stateNext']

		q2 = self.tq(stateNext)
		q2Max = q2.max(1)
		targets = reward + self.discount*(1 - terminal)*q2Max

		return state, targets, action

	def computeDeltas(self, k=None):
		k = k or self.evalBatchSize
		batch = self.gameBuf.sample(k)
		state, targets, action = self.computTargets(batch)

		# deltas, q, grads = self.sess.run(
		# 		(self.trainer['deltas'], self.QNetwork['net'],
		# 		self.trainer['grads']),
		# 		feed_dict={self.QNetwork['inputPH'] : state,
		# 		self.trainer['targetsPH'] : targets,
		# 		self.trainer['actionPH'] : action})

		return deltas, q, grads, targets

	def train(self):
		batch = self.gameBuf.sample(self.batchSize)
		state, targets, action = self.computTargets(batch)
		self.trainerRun(state, targets, action)

	def report(self):
		pass

	def save(self, path, tag=None):
		pass

	def load(self, path, tag=None):
		pass
