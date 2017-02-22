#!/usr/bin/python
#coding=utf-8

from optionParser import OptionParser
import players

opt = OptionParser()
player = players.ExplorePlayer(opt)

player.run(opt.get('steps'))
