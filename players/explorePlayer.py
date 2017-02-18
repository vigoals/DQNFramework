#coding=utf-8

from player import Player

class ExplorePlayer(Player):
	"""用于学习过程探索游戏环境"""
	def __init__(self, opt, agent=None):
		super(ExplorePlayer, self).__init__(opt, agent)

	def onStartStep(self):
		self.gameEnv.render()
		self.action = self.gameEnv.sample()
		print self.gameEnv.lives
