#coding=utf-8

import sys
sys.path.append('.')

import numpy as np
import gameBuf
from optionParser import OptionParser
opt = OptionParser()
buf = gameBuf.AtariBuf(opt)


for i in range(10):
	k = 1.0/(i + 1)
	terminal = False
	if i == 9:
		terminal = True
	state = np.array([[k, 0], [0, k]])
	buf.add(i, state, terminal)
	buf.setAction(np.random.randint(4))
	buf.setReward(np.random.randint(2))

# state = np.empty((2, 2)).astype(np.float)
# state[()] = 1000.0/2000.0
# buf.add(1000, state, True)

batch =  buf.sample(1)
print batch['state']
print batch['stateNext']
