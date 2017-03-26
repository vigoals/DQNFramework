#coding=utf-8

import tensorflow as tf
import numpy as np

def getWeights(shape, stddev=0.01, dtype=tf.float32, name='weights'):
	return tf.get_variable(name, shape,
        initializer=tf.random_uniform_initializer(-stddev, stddev))
	# return tf.Variable(
	# 		tf.truncated_normal(shape, stddev=stddev, dtype=dtype), name=name)

def getBias(shape, stddev=0.01, dtype=tf.float32, name='bias'):
	return tf.get_variable(name, shape,
        initializer=tf.random_uniform_initializer(-stddev, stddev))
	# return tf.Variable(
	# 		tf.truncated_normal(shape, stddev=stddev, dtype=dtype), name=name)

def conv2d(input_, outputDim, kernelSize,
		strides, stddev=None, activation=None, name='conv2d'):
	inputDim = input_.get_shape().as_list()[-1]

	if stddev is None:
		stddev = 1/np.sqrt(kernelSize[0]*kernelSize[1]*inputDim)

	with tf.variable_scope(name):
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

	with tf.variable_scope(name):
		weight = getWeights([shape[1], outputSize], stddev)
		bias = getBias([outputSize], stddev)

		if activation is not None:
			return activation(tf.matmul(input_, weight) + bias), weight, bias
			# return activation(tf.nn.bias_add(tf.matmul(input_, weight), bias)), \
			# 		weight, bias
		else:
			return tf.matmul(input_, weight) + bias, weight, bias
			# return tf.nn.bias_add(tf.matmul(input_, weight), bias), weight, bias

# class RMSPropOptimizer(object):
# 	"""RMSPropOptimizer"""
# 	def __init__(self, varList, learningRate, loss, decay=0.95,
# 			epsilon=0.01, centered=False, sess=None, name='rmsprop'):
# 		super(RMSPropOptimizer, self).__init__()
#
# 		self.varList = varList
# 		self.learningRate = learningRate
# 		self.loss = loss
# 		self.decay = decay
# 		self.epsilon = epsilon
# 		self.centered = centered
# 		self.sess = sess
# 		self.name = name
#
# 		self.grads = []
# 		self.meanGrad = []
# 		self.meanSquare = []
#
# 		self.updateMeanSquare = []
# 		self.updateMeanGrad = []
# 		self.updateGrad = []
#
# 		self.build()
#
# 	def build(self):
# 		with tf.variable_scope(self.name):
# 			i = 0
# 			for v in self.varList:
# 				g = tf.gradients(self.loss, v)[0]
# 				self.grads.append(g)
#
# 				mg = tf.get_variable('mg-' + str(i), v.get_shape().as_list(),
# 			        	initializer=tf.zeros_initializer())
# 				self.meanGrad.append(mg)
#
# 				op = tf.assign(mg, self.decay*mg + (1-self.decay)*g)
# 				self.updateMeanGrad.append(op)
#
# 				ms = tf.get_variable('ms-' + str(i), v.get_shape().as_list(),
# 			        	initializer=tf.zeros_initializer())
# 				self.meanSquare.append(ms)
#
# 				op = tf.assign(ms, self.decay*ms + (1-self.decay)*tf.pow(g, 2))
# 				self.updateMeanSquare.append(op)
#
# 				if not self.centered:
# 					op = tf.assign(v,
# 							v - self.learningRate/tf.sqrt(ms + self.epsilon)*g)
# 				else:
# 					op = tf.assign(v,
# 							v - self.learningRate/tf.sqrt(ms - tf.pow(mg, 2) + self.epsilon)*g)
# 				self.updateGrad.append(op)
# 				i += 1
#
# 	def computeGrads(self, feed_dict={}):
# 		return self.sess.run(self.grads, feed_dict=feed_dict)
#
# 	def applyGrads(self, feed_dict={}):
# 		self.sess.run(
# 				self.updateMeanGrad + self.updateMeanSquare + self.updateGrad,
# 				feed_dict=feed_dict)
#
# 	def getMeanSquare(self):
# 		return self.sess.run(self.meanSquare)
#
# 	def getMeanGrad(self):
# 		return self.sess.run(self.meanGrad)

class DQNNetwork(object):
	"""Network"""
	def __init__(self, inputDim, outputDim,
			convLayers=None, convShape=None,
			linearLayers=None, dueling=False,
			sess=None, name='net'):
		self.paras = []
		self.parasAssigns = []
		self.parasAssignsPH = []
		self.input = None
		self.inputDim = inputDim
		self.outputDim = outputDim
		self.convShape = convShape
		self.convLayers = convLayers
		self.dueling = dueling
		self.linearLayers = linearLayers
		self.name = name
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self.sess = sess if sess is not None else tf.Session(config)
		self.layers = []
		self.reshape = None

		self.output = self.buidNet()

	def buidNet(self):
		lastOp = None
		with tf.variable_scope(self.name):
			self.input = tf.placeholder(
					tf.float32, [None, self.inputDim], name='input')

			lastOp = self.input

			if self.convLayers is not None:
				lastOp = tf.reshape(lastOp ,[-1] + self.convShape)
				self.reshape = lastOp

				for i, l in enumerate(self.convLayers):
					name = 'conv' + str(i)
					lastOp, w, b = conv2d(lastOp,
							l[0],	# outputSize
							l[1],	# kernelSize
							l[2],	# strides
							activation=tf.nn.relu, name=name)
					self.paras.append(w)
					self.paras.append(b)
					self.layers.append(lastOp)

				shape = lastOp.get_shape().as_list()[1:]
				lastOp = tf.reshape(lastOp,
						[-1, reduce(lambda x, y: x*y, shape)])
			if self.linearLayers is not None:
				for i, l in enumerate(self.linearLayers):
					name = 'linear' + str(i)
					lastOp, w, b = linear(lastOp, self.linearLayers[0],
							activation=tf.nn.relu, name=name)
					self.paras.append(w)
					self.paras.append(b)
					self.layers.append(lastOp)

			if not self.dueling:
				lastOp, w, b = linear(lastOp, self.outputDim, name='output')
				self.paras.append(w)
				self.paras.append(b)
			else:
				duelA, w, b = linear(lastOp, self.outputDim, name='duelA')
				self.paras.append(w)
				self.paras.append(b)
				duelV, w, b = linear(lastOp, 1, name='duelV')
				self.paras.append(w)
				self.paras.append(b)

				lastOp = duelV + \
						(duelA - tf.reduce_mean(duelA, 1, keep_dims=True))

			# 用于设置paras
			self.parasAssigns = []
			self.parasAssignsPH = []
			for p in self.paras:
				ph = tf.placeholder(tf.float32, p.get_shape().as_list())
				op = tf.assign(p, ph)
				self.parasAssignsPH.append(ph)
				self.parasAssigns.append(op)

		return lastOp

	def getParas(self):
		return self.sess.run(self.paras)

	def setParas(self, paras):
		for i, p in enumerate(paras):
			ph = self.parasAssignsPH[i]
			pa = self.parasAssigns[i]
			self.sess.run(pa, feed_dict={ph:p})

	def forward(self, input_):
		return self.sess.run(self.output, feed_dict={self.input:input_})

class DQNOptimizer(object):
	def __init__(self, net, learningRate,
			nActions, clipDelta=None, name='optimizer'):
		self.net = net
		self.sess = net.sess
		self.ms = []
		self.m = []

		with tf.variable_scope(name):
			self.targetsPH = tf.placeholder(tf.float32, [None], name='targetsPH')
			self.actionPH = tf.placeholder(tf.int32, [None], name='actionPH')

			self.actionOneHot = tf.one_hot(self.actionPH, nActions, 1.0, 0.0)

			self.qOneHot = tf.reduce_sum(net.output*self.actionOneHot, 1)
			self.deltas = self.targetsPH - self.qOneHot

			if clipDelta:
				deltasCliped = tf.clip_by_value(
						self.deltas, -clipDelta, clipDelta)

				self.loss = tf.reduce_sum(tf.square(deltasCliped)/2 \
						+ (tf.abs(self.deltas) - tf.abs(deltasCliped))*clipDelta)
				self.deltas = deltasCliped
			else:
				self.loss = tf.reduce_sum(tf.square(self.deltas)/2)

			self.optim = tf.train.RMSPropOptimizer(learningRate,
					decay=0.95, epsilon=0.01, centered=True)

			self.grads = self.optim.compute_gradients(self.loss, net.paras)
			self.applyGrads = self.optim.apply_gradients(self.grads)

			for p in net.paras:
				ms = self.optim.get_slot(p, 'rms')
				m = self.optim.get_slot(p, 'm')
				self.ms.append(ms)
				if m is not None:
					self.m.append(m)

	def train(self, state, targets, action):
		self.sess.run(self.applyGrads,
				feed_dict={self.net.input : state,
				self.targetsPH : targets,
				self.actionPH : action})

	def getInfo(self, state, targets, action):
		deltas, q, grads, ms, m = self.sess.run((self.deltas,
				self.net.output,
				self.grads,
				self.ms,
				self.m),
				feed_dict={})

		return deltas, q, grads, ms, m
