#coding=utf-8
from baseBuf import BaseBuf
import copy
import numpy as np

class AtariBuf(BaseBuf):
	"""docstring for AtariBuf"""
	def __init__(self, opt):
		super(AtariBuf, self).__init__(opt)
		self.histLen = opt.get('histLen', 4)
		self.histSelect = opt.get('histSelect', range(-self.histLen + 1, 1))
		assert len(self.histSelect) == self.histLen, 'histSelect与histLen不相符'
		assert self.histSelect[-1] == 0, 'histSelect最后一项必须为0'

	def statePreProcess(self, state):
		state = state.reshape(-1)
		return state.copy().astype(np.uint8)

	def getState(self, i=None):
		i = i if i is not None else -1
		i = i if i >= 0 else (len(self.buf) + i)
		assert 0 <= i < len(self.buf), '超出范围'
		shape = list(self.buf[0]['state'].shape)
		shape.append(self.histLen)
		state = np.zeros(shape)

		k = i
		for j in range(self.histLen - 1, -1, -1):
			t = False
			while k > i + self.histSelect[j]:
				k -= 1
				if k < 0 or self.buf[k]['terminal']:
					t = True
					break
			if t:
				break
			state[:, j] = self.buf[k]['state'].astype(np.float)/255.0

		return state.reshape(-1)
