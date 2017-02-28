#!/usr/bin/python
#coding=utf-8
import sys
sys.path.append('.')

import numpy as np
# from optionParser import OptionParser
# opt = OptionParser()
import gameEnv

game = gameEnv.AtariEnv('breakout', 4, 30)

observation, reward, terminal = game.newGame()
training = False
while True:
	game.render()

	if not terminal:
		observation, reward, terminal = game.step(game.sample(), training=training)
	else:
		observation, reward, terminal = game.nextRandomGame(training=training)
