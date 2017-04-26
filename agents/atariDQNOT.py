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
		state = np.array(state)
		n = state.shape[0]

		assert False

	def computTargets(self, batch):
		state, targets, action = super(AtariDQNOT, self).computTargets(batch)

		otStates = batch['otStates']
		otR = batch['otR']
		otTags = batch['otTags']

		assert False
