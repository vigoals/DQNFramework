# coding=utf-8
# Learning to Play in a Day:
# 		Faster Deep Reinforcement Learning by Optimality Tightening

from atariBuf import AtariBuf
import copy
import numpy as np

class OTBuf(AtariBuf):
	def __init__(self, opt):
		self.K = opt.get('K', 4)
		self.discount = opt.get('discount', 0.99)
		super(OTBuf, self).__init__(opt)

	def add(self, step, state, terminal):
		super(OTBuf, self).add(step, state, terminal)
		self.buf[-1]['R'] = 0

	def computeR(self):
		R = 0

		for i in range(len(self), max(len(self) - 2001, 0), -1):
			R = self.buf[i]['reward'] + self.discount*R

			if self.buf[i]['terminal']:
				R = 0

				if i < len(self) - 1000:
					break

			self.buf[i]['R'] = R

	def sample(self, n):
		batch = super(OTBuf, self).sample(n)

		steps = batch['steps']

		otStates = np.zeros(self.K*2*len(steps)
				+ list(self.buf[0]['state'].shape))
		otR = [0]*(self.K*2*len(steps))
		otTags = [False]*(self.K*2*len(steps))

		for i, step in enumerate(steps):
			index = self.stepToIndex(step)

			k = 2*i;
			for j in range(index-1, index-self.K-1, -1):
				if self.buf[j]['terminal']:
					break
				otStates[k] = self.getState(j)
				otTags[k] = True
				otR[k] = self.buf[j]['R']
				k += 1

			if self.buf[index]['terminal']:
				 continue

			k = 2*i + self.K
			for j in range(index+1, index+self.K+1):
				if self.buf[j]['terminal']:
					break
				otStates[k] = self.getState(j)
				otTags[k] = True
				otR[k] = self.buf[j]['R']
				k += 1

		batch['otStates'] = otStates
		batch['otR'] = otR
		batch['otTags'] = otTags
		return batch
