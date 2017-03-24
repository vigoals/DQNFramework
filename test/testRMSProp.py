#!/usr/bin/python
#coding=utf-8
import sys
sys.path.append('.')

import agents
from agents.netComm import *
import tensorflow as tf
import numpy as np

sess = tf.Session()
ph = tf.placeholder(tf.float32, [None, 10])

l1, w, b = linear(ph, 5)

outPh = tf.placeholder(tf.float32, [None, 5])
loss = tf.reduce_mean(tf.square(l1 - outPh))

optim = RMSPropOptimizer([w, b], 0.00025, loss, sess=sess)

sess.run(tf.global_variables_initializer())

input_ = np.random.random((2, 10))
output = np.random.random((2, 5))

for i in range(5):
	grad = optim.computeGrads({ph:input_, outPh:output})
	# print grad
	optim.applyGrads({ph:input_, outPh:output})

	print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
	print optim.getMeanGrad()
	print '####'
	print optim.getMeanSquare()

	print
	print
	print
	print

	# print sess.run(loss, {ph:input_, outPh:output})
