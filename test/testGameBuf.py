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

def plotState(state, stateNext=None):
	state = (state.reshape(height, width, histLen)*255).astype(np.float)
	# stateNext = (stateNext.reshape(height, width, histLen)*255).astype(np.float)
	plt.subplot(141)
	plt.imshow(state[:, :, 0], cmap=plt.cm.gray)
	plt.subplot(142)
	plt.imshow(state[:, :, 1], cmap=plt.cm.gray)
	plt.subplot(143)
	plt.imshow(state[:, :, 2], cmap=plt.cm.gray)
	plt.subplot(144)
	plt.imshow(state[:, :, 3], cmap=plt.cm.gray)

	# plt.subplot(245)
	# plt.imshow(stateNext[:, :, 0], cmap=plt.cm.gray)
	# plt.subplot(246)
	# plt.imshow(stateNext[:, :, 1], cmap=plt.cm.gray)
	# plt.subplot(247)
	# plt.imshow(stateNext[:, :, 2], cmap=plt.cm.gray)
	# plt.subplot(248)
	# plt.imshow(stateNext[:, :, 3], cmap=plt.cm.gray)


plt.ion()
plt.show()
observation, reward, terminal = player.reset(True)
for i in range(1000):
	print 'step:%10d   \r' % i,
	gameEnv.render()
	player.action, _ = \
			agent.perceive(i,  observation, reward, terminal, 1, False)

	state = gameBuf.getState()
	plotState(state)

	observation, reward, terminal = player.oneStep(True)

	print "!!!!!!!!!!!!!!!"
	print player.action
	print reward, terminal

	raw_input()

# agent.report()
#
# plt.ion()
# plt.show()
#
# while True:
# 	batch = gameBuf.sample(10)
# 	k = np.random.randint(10)
# 	state = batch['state'][k]
# 	terminal = batch['terminal'][k]
# 	if terminal:
# 		stateNext = batch['stateNext'][k]
# 		plotState(state, stateNext)
#
# 		raw_input()
