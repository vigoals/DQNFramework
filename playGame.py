#coding=utf-8

import gameEnv

class PlayGame:
	def __init__(self, opt, agent=None):
		exec('Env = gameEnv.' + opt.get('gameEnv'))
		self.env = opt.get('env')
		self.actrep = opt.get('actrep', 4)
		self.randomStarts = opt.get('randomStarts', 30)
		self.gameEnv = Env(self.env, self.actrep, self.randomStarts)
		self.nActions = self.gameEnv.getActionSpace()
		self.width = opt.get('width', 84)
		self.heigth = opt.get('heigth', 84)

	def run(max_steps=None, max_episode=None):
		assert (max_steps is not  None) or (max_episode is not None), \
				"游戏无法结束"
