#coding=utf-8

from player import Player

class ExplorePlayer(Player):
	"""用于学习过程探索游戏环境"""
	def __init__(self, opt, agent=None):
		super(ExplorePlayer, self).__init__(opt, agent)
		self.reportFreq = opt.get('reportFreq', 10000)

	def onStartStep(self):
		self.gameEnv.render()
		self.action, _ = self.agent.perceive(self.step, self.observation,
				self.reward, self.terminal, 1, not self.training)

	def report(self):
		print "Report in step %8d" % self.step

	def onEndStep(self):
		if self.step%self.reportFreq == 0:
			self.report()
