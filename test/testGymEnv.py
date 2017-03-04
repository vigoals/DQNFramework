#!/usr/bin/python
#coding=utf-8
import sys
sys.path.append('.')

import numpy as np
# from optionParser import OptionParser
# opt = OptionParser()
import gameEnv

game = gameEnv.GymEnv('CartPole-v0')

print game.getActions()
print game.getObservationSpace()

observation, reward, terminal = game.newGame()
episodeReward = reward
episode = 0
totalReward = 0
while True:
	# print reward, terminal
	game.render()

	if not terminal:
		observation, reward, terminal = game.step(game.sample())
		episodeReward += reward
	else:
		observation, reward, terminal = game.newGame()
		totalReward += episodeReward
		episode += 1
		if episode >= 30:
			break
		episodeReward = 0

print float(totalReward)/episode
