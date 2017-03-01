#!/usr/bin/python
#coding=utf-8

from toolsComm import *

if __name__ == '__main__':
	opt = OptionParser()
	opt.set('render', True)
	opt.set('trainFreq', 0)
	player = players.EvalPlayer(opt)

	savePath = opt.get('savePath')
	agent = player.agent
	agent.load(savePath)

	player.run(opt.get('evalMaxSteps'),
			opt.get('evalMaxEpisode'), training=False)

	print player.getInfo()
