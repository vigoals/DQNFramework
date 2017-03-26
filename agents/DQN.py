#coding=utf-8


#coding=utf-8
import tensorflow as tf
from netComm import *
from baseAgent import BaseAgent
import numpy as np

class DQN(BaseAgent):
	def __init__(self, opt, sess=None, buildNet=True):
		super(DQN, self).__init__(opt)
		# GPU 不会全部占用
		config = tf.ConfigProto()
		# config.log_device_placement = True
		config.gpu_options.allow_growth = True
		self.sess = sess if sess is not None else tf.Session(config=config)
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
		self.convLayers = opt.get('convLayers')
		self.convShape = opt.get('convShape')
		self.linearLayers = opt.get('linearLayers')
		self.dueling = opt.get('dueling', False)
		self.doubleDQN = opt.get('doubleDQN', False)
		self.maxScale = opt.get('maxScale', 10)

		tmp = opt.get('buf').split('.')
		exec('import ' + tmp[0])
		exec('Buf = ' + opt.get('buf'))
		self.gameBuf = Buf(opt)
		self.step = None

		if buildNet:
			with tf.device(self.device):
				self.QNetwork = DQNNetwork(self.stateDim,
						self.nActions,
						convLayers=self.convLayers,
						convShape=self.convShape,
						linearLayers=self.linearLayers,
						dueling=self.dueling,
						sess=self.sess, name='QNetwork')

				if self.targetFreq > 0:
					self.QTarget = DQNNetwork(self.stateDim,
						self.nActions,
						convLayers=self.convLayers,
						convShape=self.convShape,
						linearLayers=self.linearLayers,
						dueling=self.dueling,
						sess=self.sess, name='QTarget')
				else:
					self.QTarget = self.QNetwork

				self.optimizer = DQNOptimizer(self.QNetwork, self.learningRate,
						self.nActions, self.clipDelta)

				self.sess.run(tf.global_variables_initializer())
				self.updateTarget()

			self.saver = tf.train.Saver(self.QNetwork.paras)

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
			q = self.q(state).reshape(-1)
			action = np.argmax(q)
			return action, q

	def preprocess(self, step, observation, reward, terminal, eval_):
		if not eval_ and step > 0:
			r = reward
			r = min(r, self.maxReward) if self.maxReward is not None else r
			r = max(r, self.minReward) if self.minReward is not None else r
			self.gameBuf.setReward(r)

		observation = np.array(observation)
		mean = (self.stateLow + self.stateHigh)/2
		scale = (self.stateHigh - self.stateLow)/2
		# maxScale
		if self.maxScale is not None:
			scale[scale > self.maxScale] = self.maxScale
		state = (observation - mean)/scale

		if not eval_:
			self.gameBuf.add(step, state, terminal)

		return state

	def perceive(self, step, observation, reward, terminal, ep, eval_):
		state = self.preprocess(step, observation, reward, terminal, eval_)

		if not eval_:
			self.step = step

		action, q = self.policy(state, ep)

		if not eval_:
			self.gameBuf.setAction(action)

			# train
			if step > self.learnStart \
					and self.trainFreq > 0 \
					and step%self.trainFreq == self.trainFreq - 1
					and len(self.gameBuf) > self.batchSize:
				self.train()

			# update target
			if step > self.learnStart \
					and self.targetFreq > 0 \
					and step%self.targetFreq == self.targetFreq - 1:
				self.updateTarget()

		return action, q

	def computTargets(self, batch):
		state = batch['state']
		action = batch['action']
		reward = batch['reward']
		terminal = batch['terminal']
		stateNext = batch['stateNext']

		targets = None
		if not self.doubleDQN:
			q2 = self.tq(stateNext)
			q2Max = q2.max(1)
			targets = reward + self.discount*(1 - terminal)*q2Max
		else:
			q2 = self.tq(stateNext)
			q2a = self.q(stateNext).argmax(1)
			q2Max = q2[:, q2a][:, 0]
			targets = reward + self.discount*(1 - terminal)*q2Max

		return state, targets, action

	def train(self):
		batch = self.gameBuf.sample(self.batchSize)
		state, targets, action = self.computTargets(batch)
		# self.trainerRun(state, targets, action)
		self.optimizer.train(state, targets, action)

	def report(self):
		batch = self.gameBuf.sample(self.evalBatchSize)
		state, targets, action = self.computTargets(batch)

		deltas, q, grads, ms, m = self.optimizer.getInfo(state, targets, action)

		print 'TD:%10.6f' % np.abs(deltas).mean()
		print 'deltas mean:%10.6f' % deltas.mean()
		print 'deltas std:%10.6f' % deltas.std()
		print 'Q mean:%10.6f' % q.mean()
		print 'Q std:%10.6f' % q.std()

		paras = self.QNetwork.getParas()
		norms = []
		maxs = []
		print 'Paras info:'
		for w in paras:
			norms.append(np.abs(w).mean())
			maxs.append(np.abs(w).max())
		print 'paras norms: ' + str(norms)
		print 'paras maxs:' + str(maxs)

		norms = []
		maxs = []
		print 'Grads info:'
		for w in grads:
			norms.append(np.abs(w).mean()/self.evalBatchSize)
			maxs.append(np.abs(w).max()/self.evalBatchSize)
		print 'grads norms: ' + str(norms)
		print 'grads maxs:' + str(maxs)

	def save(self, path, tag=None):
		if not tag:
			path = path + '/agent'
		else:
			path = path + '/agent-' + tag

		self.saver.save(self.sess, path)

	def load(self, path, tag=None):
		if not tag:
			path = path + '/agent'
		else:
			path = path + '/agent-' + tag

		self.saver.restore(self.sess, path)
		self.updateTarget()
