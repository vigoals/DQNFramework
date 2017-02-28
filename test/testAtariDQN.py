#!/usr/bin/python
#coding=utf-8

import sys
sys.path.append('.')

import numpy as np
import agents
from optionParser import OptionParser

opt = OptionParser()
opt.set('nActions', 4)
agent = agents.AtariDQN(opt)

state = np.zeros((4, agent.stateDim))
targets = np.array([10, -10, -5, 5])
action = np.array([0, 1, 2, 3])
# print agent.q(state)
# print agent.tq(state)

print agent.q(state)
print agent.tq(state)
for i in range(1000):
	print 'step:%10d   \r' % i,
	deltas, _ = agent.sess.run(
			(agent.trainer['deltas'], agent.trainer['updateGrads']),
			feed_dict={agent.QNetwork['inputPH']:state,
			agent.trainer['targetsPH']:targets,
			agent.trainer['actionPH']:action})

print agent.q(state)

targets = np.array([0, 0, 0, 0])
for i in range(1000):
	print 'step:%10d   \r' % i,
	deltas, _ = agent.sess.run(
			(agent.trainer['deltas'], agent.trainer['updateGrads']),
			feed_dict={agent.QNetwork['inputPH']:state,
			agent.trainer['targetsPH']:targets,
			agent.trainer['actionPH']:action})
print agent.q(state)
