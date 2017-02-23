#coding=utf-8
import tensorflow as tf
from netComm import *
from baseAgent import BaseAgent
import numpy as np
import gameBuf
from PIL import Image
import deepdish as dd

class AtariDQN(BaseAgent):
	"""docstring for AtariDQN"""
	def __init__(self, opt):
		super(AtariDQN, self).__init__(opt)
		# GPU 不会全部占用
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config)
		self.device = opt.get('device')
		self.histLen = opt.get('histLen', 4)
		self.height = opt.get('height', 84)
		self.width = opt.get('width', 84)
		self.stateDim = self.histLen*self.height*self.width
		self.nActions = opt.get('nActions')
		self.clipDelta = opt.get('clipDelta', 1)
		self.maxReward = opt.get('maxReward', 1)
		self.minReward = opt.get('minReward', -self.maxReward)
		self.learningRate = opt.get('learningRate', 0.0025)
		self.learnStart = opt.get('learnStart', 50000)
		self.discount = opt.get('discount', 0.99)
		self.trainFreq = opt.get('trainFreq', 4)
		self.targetFreq = opt.get('targetFreq', 10000)
		self.batchSize = opt.get('batchSize', 32)
		self.evalBatchSize = opt.get('evalBatchSize', 1000)

		self.gameBuf = gameBuf.AtariBuf(opt)
		self.evalBuf = gameBuf.AtariBuf(opt)
		self.evalBuf.reset(self.histLen*2)
		self.step = None

		self.QNetwork = self.createNetwork()
		self.trainer = self.createTrainer()
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
				w = self.sess.run(tmp)
				op = tf.assign(self.QTarget['paras'][i], w)
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
			parameters.append(w)
			parameters.append(b)
			l4, w, b = conv2d(l3, 64, [4, 4], [2, 2], activation=tf.nn.relu)
			parameters.append(w)
			parameters.append(b)
			l5, w, b = conv2d(l4, 64, [3, 3], [1, 1], activation=tf.nn.relu)
			parameters.append(w)
			parameters.append(b)

			shape = l5.get_shape().as_list()[1:]
			l6 = tf.reshape(l5,
					[-1, reduce(lambda x, y: x*y, shape)])
			l7, w, b = linear(l6, 512, activation=tf.nn.relu)
			parameters.append(w)
			parameters.append(b)
			l8, w, b = linear(l7, self.nActions)
			parameters.append(w)
			parameters.append(b)
			net = l8

		return {'net':net, 'inputPH':inputPH, 'paras':parameters}

	def createTrainer(self):
		targetsPH = None
		actionPH = None
		deltas = None
		deltasCliped = None
		loss = None
		optim = None

		with tf.device(self.device):
			targetsPH = tf.placeholder(tf.float32, [None])
			actionPH = tf.placeholder(tf.int32, [None])

			actionOneHot = tf.one_hot(actionPH, self.nActions, 1.0, 0.0)

			q = tf.reduce_sum(self.QNetwork['net']*actionOneHot, 1)
			deltas = targetsPH - q


			# clip delta
			if self.clipDelta:
				deltasCliped = tf.clip_by_value(
						deltas, -self.clipDelta, self.clipDelta)

				loss = tf.reduce_mean(tf.square(deltasCliped)/2
						+ (tf.abs(deltas) - tf.abs(deltasCliped))*self.clipDelta)
			else:
				loss = tf.reduce_mean(tf.square(deltas)/2)

			optim = tf.train.RMSPropOptimizer(
					self.learningRate, 0.95, 0.95, 0.01).minimize(loss)

		return {'targetsPH':targetsPH, 'actionPH':actionPH,
				'deltas':deltasCliped if deltasCliped is not None else deltas,
				'optim':optim}

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
			q = self.q([state]).reshape(-1)
			action = np.argmax(q)
			return action, q

	def preprocess(self, observation):
		screen = Image.fromarray(observation)
		screen = screen.convert('L')
		screen = screen.resize((84, 84))
		screen = np.asarray(screen)
		return screen

	def perceive(self, step, observation, reward, terminal, ep, eval_):
		if not eval_ and step > 0:
			r = max(min(reward, self.maxReward), self.minReward)
			self.gameBuf.setReward(r)

		screen = self.preprocess(observation)
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
			if step > self.learnStart and step%self.trainFreq == 0:
				self.train()

			# update target
			if step > self.learnStart and step%self.targetFreq == 0:
				self.updateTarget()

		return action, q

	def trainerRun(self, state, targets, action):
		deltas, _ = self.sess.run(
				(self.trainer['deltas'], self.trainer['optim']),
				feed_dict={self.QNetwork['inputPH'] : state,
				self.trainer['targetsPH'] : targets,
				self.trainer['actionPH'] : action})

		return np.abs(deltas).mean()

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

	def computeDeltas(self):
		batch = self.gameBuf.sample(self.evalBatchSize)
		state, targets, action = self.computTargets(batch)

		deltas, q = self.sess.run(
				(self.trainer['deltas'], self.QNetwork['net']),
				feed_dict={self.QNetwork['inputPH'] : state,
				self.trainer['targetsPH'] : targets,
				self.trainer['actionPH'] : action})

		return deltas, q

	def train(self):
		batch = self.gameBuf.sample(self.batchSize)
		state, targets, action = self.computTargets(batch)
		self.trainerRun(state, targets, action)

	def report(self):
		if len(self.gameBuf) > 1:
			deltas, q = self.computeDeltas()
			print 'TD:%10.6f' % np.abs(deltas).mean()
			print 'deltas std:%10.6f' % deltas.std()
			print 'Q mean:%10.6f' % q.mean()
			print 'Q std:%10.6f' % q.std()

	def save(self, path, tag=None):
		if not tag:
			path = path + '/agent.h5'
		else:
			path = path + '/agent-' + tag + '.h5'
		paras = []
		for i in range(len(self.QNetwork['paras'])):
			tmp = self.QNetwork['paras'][i]
			w = self.sess.run(tmp)
			paras.append(w)

		try:
			dd.io.save(path, paras)
		except IOError:
			print 'WARNING: 保存agent到 %s 失败' % path
