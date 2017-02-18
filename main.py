#!/usr/bin/python
#coding=utf-8

# import gameEnv
# from PIL import Image
# import numpy as np
#
# env = gameEnv.AtariEnv('Breakout-v0', 4, 30)
# print env.getActions()
# observation, reward, terminal = env.newGame()
#
# for i in range(10000):
# 	# env.render()
# 	action = env.sample()
#
# 	screen = Image.fromarray(observation)
# 	screen = screen.convert('L')
# 	screen = screen.resize((84, 84))
# 	screen.show()
# 	print np.array(screen.getdata()).shape
# 	assert False
#
# 	if not terminal:
# 		observation, reward, terminal = env.step(action)
# 	else:
# 		observation, reward, terminal = env.nextRandomGame(training=True)
# 		print "terminal"

from optionParser import OptionParser
from playGame import PlayGame

opt = OptionParser()
pg = PlayGame(opt)
