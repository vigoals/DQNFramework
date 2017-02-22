#coding=utf-8

import sys
sys.path.append('.')

import numpy as np
import agents
from optionParser import OptionParser

opt = OptionParser()
opt.set('nActions', 4)
agent = agents.AtariDQN(opt)

state = np.zeros((3, agent.stateDim))
targets = np.array([10, -10, 5])
action = np.array([0, 1, 3])
# print agent.q(state)
# print agent.tq(state)

print agent.q(state)
for i in range(1000):
	deltas, _ = agent.sess.run(
			(agent.trainer['deltas'], agent.trainer['optim']),
			feed_dict={agent.QNetwork['inputPH']:state,
			agent.trainer['targetsPH']:targets,
			agent.trainer['actionPH']:action})

print agent.q(state)
print deltas
