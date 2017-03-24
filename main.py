#!/usr/bin/python
#coding=utf-8

import os
print "PID:%10d" % os.getpid()

from optionParser import OptionParser
import players
import time


if __name__ == '__main__':
	opt = OptionParser()
	savePath = opt.get('savePath', './save')
	try:
		os.makedirs(savePath)
	except OSError:
		pass

	player = players.ExplorePlayer(opt)
	print "Start at %s" % time.asctime()
	print str(opt)
	opt.save(savePath)
	player.run(opt.get('steps'))
