#!/usr/bin/python
#coding=utf-8

import sys
sys.path.append('.')

import players
from optionParser import OptionParser
import matplotlib.pyplot as plt
import numpy as np
import time

opt = OptionParser()
opt.set('trainFreq', 0)
player = players.EvalPlayer(opt)
agent = player.agent
gameEnv = player.gameEnv
evalBuf = agent.evalBuf
gameBuf = agent.gameBuf
histLen = opt.get('histLen')
width = opt.get('width')
height = opt.get('height')

def plotState(state, stateNext):
	state = (state.reshape(height, width, histLen)*255).astype(np.float)
	stateNext = (stateNext.reshape(height, width, histLen)*255).astype(np.float)
	plt.subplot(241)
	plt.imshow(state[:, :, 0], cmap=plt.cm.gray)
	plt.subplot(242)
	plt.imshow(state[:, :, 1], cmap=plt.cm.gray)
	plt.subplot(243)
	plt.imshow(state[:, :, 2], cmap=plt.cm.gray)
	plt.subplot(244)
	plt.imshow(state[:, :, 3], cmap=plt.cm.gray)

	plt.subplot(245)
	plt.imshow(stateNext[:, :, 0], cmap=plt.cm.gray)
	plt.subplot(246)
	plt.imshow(stateNext[:, :, 1], cmap=plt.cm.gray)
	plt.subplot(247)
	plt.imshow(stateNext[:, :, 2], cmap=plt.cm.gray)
	plt.subplot(248)
	plt.imshow(stateNext[:, :, 3], cmap=plt.cm.gray)


observation, reward, terminal = player.reset(False)
for i in range(1000):
	print 'step:%10d   \r' % i,
	player.action, _ = \
			agent.perceive(i,  observation, reward, terminal, 1, False)

	# gameEnv.render()
	observation, reward, terminal = player.oneStep(False)

# agent.report()

# plt.ion()
# plt.show()
#
# while True:
# 	batch = gameBuf.sample(10)
# 	k = np.random.randint(10)
# 	state = batch['state'][k]
# 	stateNext = batch['stateNext'][k]
# 	plotState(state, stateNext)
#
# 	raw_input()
