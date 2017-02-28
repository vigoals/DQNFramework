#!/usr/bin/python
#coding=utf-8
import sys
sys.path.append('.')

import gameEnv
game = gameEnv.AtariEnv('breakout', 4, 30)
observation, reward, terminal = game.newGame()
game.render()

episodeStep = 0
action = None
while True:
	episodeStep += 1
	if not terminal:
		observation, reward, terminal = game.step(
				action if action is not None else game.sample(), True)
	else:
		observation, reward, terminal = game.nextRandomGame(True)
		action = 0
		print 'step:%10d' % episodeStep
		episodeStep = 0
		print 'Terminal'
	game.render()
