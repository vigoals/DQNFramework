#!/usr/bin/python
#coding=utf-8
import sys
sys.path.append('.')

import numpy as np
# from optionParser import OptionParser
# opt = OptionParser()
import gameEnv

game = gameEnv.GymEnv('CartPole-v1')

print game.getActions()
print game.getObservationSpace()

observation, reward, terminal = game.newGame()
while True:
	# print reward, terminal
	game.render()

	if not terminal:
		observation, reward, terminal = game.step(game.sample())
	else:
		observation, reward, terminal = game.newGame()
