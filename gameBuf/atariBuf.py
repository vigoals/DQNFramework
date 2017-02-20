#coding=utf-8
from baseBuf import BaseBuf
import copy
import numpy as np

class AtariBuf(object):
	"""docstring for AtariBuf"""
	def __init__(self, opt):
		self.histLen = opt.get('histLen', 4)
		self.height = opt.get('height', 84)
		self.width = opt.get('width', 84)
		super(AtariBuf, self).__init__(opt)

	def statePreProcess(self, state):
		return (state*255).astype(np.uint8)

	def getState(i):
		assert i <= len(self.buf), '超出范围'
		shape = self.buf[0]['state'].shape
		shape[1:] = shape
		shape[0] = self.histLen
		state = np.zeros(shape)

		k = i
		for j in range(self.histLen - 1, -1, -1):
			state[j] = self.buf[k]['state'].astype(np.float)/255
			k = k - 1
			if k < 0 or self.buf[k]['terminal']:
				break

		return state
