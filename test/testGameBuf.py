#coding=utf-8

import sys
sys.path.append('.')

import numpy as np
import gameBuf
from optionParser import OptionParser
opt = OptionParser()
buf = gameBuf.AtariBuf(opt)


for i in range(1000):
	state = np.random.random((2, 2))
	buf.add(i, state, False)
	buf.setAction(np.random.randint(4))
	buf.setReward(np.random.randint(2))

state = np.empty((2, 2)).astype(np.float)
state[()] = 1000.0/2000.0
buf.add(1000, state, True)

print buf.buf[:10]
print buf.get()
print buf.get()
