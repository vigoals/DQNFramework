#coding=utf-8

import sys
sys.path.append('.')

import numpy as np
import gameBuf
from optionParser import OptionParser
opt = OptionParser()
buf = gameBuf.AtariBuf(opt)


for i in range(1000):
	buf.add(i, np.random.random((4, 4)), False)
	buf.setAction(np.random.randint(4))
	buf.setReward(np.random.randint(2))

buf.add(1000, np.random.random((4, 4)), True)

print buf.sample(5)
