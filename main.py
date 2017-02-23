#!/usr/bin/python
#coding=utf-8

from optionParser import OptionParser
import players
import time
import os

print "Start at %s" % time.asctime()
savePath = opt.get('savePath', './save')

try:
	os.makedirs(savePath)
except OSError:
	pass

opt = OptionParser()
player = players.ExplorePlayer(opt)
player.run(opt.get('steps'))
