#coding=utf-8
from DQN import DQN
from PIL import Image
import numpy as np
import gameBuf

class AtariDQN(DQN):
	def __init__(self, opt, sess=None, buildNet=True):
		self.histLen = opt.get('histLen', 4)
		self.height = opt.get('height', 84)
		self.width = opt.get('width', 84)
		self.stateDim = self.histLen*self.height*self.width
		self.convShape = [self.height, self.width, self.histLen]
		opt.set('convShape', self.convShape)
		opt.set('stateDim', self.stateDim)

		super(AtariDQN, self).__init__(opt, sess, buildNet)

		tmp = opt.get('buf').split('.')
		exec('import ' + tmp[0])
		exec('Buf = ' + opt.get('buf'))
		self.evalBuf = Buf(opt)
		self.evalBuf.reset(self.histLen*2)

	def preprocess(self, step, observation, reward, terminal, eval_):
		if not eval_ and step > 0:
			r = reward
			r = min(r, self.maxReward) if self.maxReward is not None else r
			r = max(r, self.minReward) if self.minReward is not None else r
			self.gameBuf.setReward(r)

		screen = Image.fromarray(observation)
		screen = screen.convert('L')
		screen = screen.resize((84, 84))
		screen = np.asarray(screen)

		if not eval_:
			self.gameBuf.add(step, screen, terminal)
			state = self.gameBuf.getState()
		else:
			self.evalBuf.add(step, screen, terminal)
			state = self.evalBuf.getState()

		return state
