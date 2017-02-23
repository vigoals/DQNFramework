#!/usr/bin/python
#coding=utf-8

from optionParser import OptionParser
import players
import time
import os

print "Start at %s" % time.asctime()
print "PID:%5d" % os.getpid()

opt = OptionParser()
savePath = opt.get('savePath', './save')
try:
	os.makedirs(savePath)
except OSError:
	pass

player = players.ExplorePlayer(opt)
print str(opt)
opt.save(savePath)
player.run(opt.get('steps'))
