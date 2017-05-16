# coding=utf-8
# Learning to Play in a Day:
# 		Faster Deep Reinforcement Learning by Optimality Tightening

from atariDQN import AtariDQN
from PIL import Image
import numpy as np
import gameBuf

class AtariDQNOT(AtariDQN):
	MAXBATCH = 1000
	def q(self, state):
		state = state.reshape([-1, self.stateDim])
		n = state.shape[0]

		tmp = []
		k = 0

		while k < n:
			s = state[k: k+MAXBATCH+1]
			tmp.append(self.QNetwork.forward(s))
			k += MAXBATCH

		return np.concatenate(tmp, 0)

	def tq(self, state):
		state = state.reshape([-1, self.stateDim])
		n = state.shape[0]

		tmp = []
		k = 0

		while k < n:
			s = state[k: k+MAXBATCH+1]
			tmp.append(self.QTarget.forward(s))
			k += MAXBATCH

		return np.concatenate(tmp, 0)


	def computTargets(self, batch):
		state, targets, action = super(AtariDQNOT, self).computTargets(batch)

		otStates = batch['otStates']
		otR = batch['otR']
		otTags = batch['otTags']

		L = np.zeros(otStates.shape[0])
		U = np.zeros(otStates.shape[0])

		return state, targets, action, L, U

	def train(self):
		# batch = self.gameBuf.sample(self.batchSize)
		# if batch is None:
		# 	return
		# state, targets, action = self.computTargets(batch)
		# # self.trainerRun(state, targets, action)
		# self.optimizer.train(state, targets, action)

		pass

	def report(self):
		# batch = self.gameBuf.sample(self.evalBatchSize)
		# if batch is None:
		# 	return
		# state, targets, action = self.computTargets(batch)
		#
		# deltas, q, grads, ms, m = self.optimizer.getInfo(state, targets, action)
		#
		# print 'TD:%10.6f' % np.abs(deltas).mean()
		# print 'deltas mean:%10.6f' % deltas.mean()
		# print 'deltas std:%10.6f' % deltas.std()
		# print 'Q mean:%10.6f' % q.mean()
		# print 'Q std:%10.6f' % q.std()
		#
		# paras = self.QNetwork.getParas()
		# norms = []
		# maxs = []
		# print 'Paras info:'
		# for w in paras:
		# 	norms.append(np.abs(w).mean())
		# 	maxs.append(np.abs(w).max())
		# print 'paras norms: ' + str(norms)
		# print 'paras maxs:' + str(maxs)
		#
		# norms = []
		# maxs = []
		# print 'Grads info:'
		# for w in grads:
		# 	norms.append(np.abs(w).mean()/self.evalBatchSize)
		# 	maxs.append(np.abs(w).max()/self.evalBatchSize)
		# print 'grads norms: ' + str(norms)
		# print 'grads maxs:' + str(maxs)

		pass
