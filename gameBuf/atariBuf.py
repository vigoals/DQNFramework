#coding=utf-8
from baseBuf import BaseBuf
import copy
import numpy as np

class AtariBuf(BaseBuf):
	"""docstring for AtariBuf"""
	def __init__(self, opt):
		super(AtariBuf, self).__init__(opt)
		self.histLen = opt.get('histLen', 4)
		# self.height = opt.get('height', 84)
		# self.width = opt.get('width', 84)

	def statePreProcess(self, state):
		state = state.reshape(-1)
		return (state*255).astype(np.uint8)

	def getState(self, i=None):
		i = i or (len(self.buf) - 1)
		assert i <= len(self.buf), '超出范围'
		shape = list(self.buf[0]['state'].shape)
		shape[1:] = shape
		shape[0] = self.histLen
		state = np.zeros(shape)

		k = i
		for j in range(self.histLen - 1, -1, -1):
			state[j] = self.buf[k]['state'].astype(np.float)/255
			k = k - 1
			if k < 0 or self.buf[k]['terminal']:
				break

		return state.reshape(-1)
