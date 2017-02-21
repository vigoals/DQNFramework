#coding=utf-8

import sys
sys.path.append('.')

import numpy as np
import agents
from optionParser import OptionParser

opt = OptionParser()
opt.set('nActions', 4)
agent = agents.AtariDQN(opt)
