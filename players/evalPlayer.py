#coding=utf-8

from player import Player

class EvalPlayer(Player):
	"""用于评估Agent"""
	def __init__(self, opt, agent=None):
		super(EvalPlayer, self).__init__(opt, agent)
		self.epsTest = opt.get('epsTest', 0.05)
		self.render = opt.get('render', False)

	def onStartStep(self):
		if self.render:
			self.gameEnv.render()
		self.action, _ = self.agent.perceive(self.step, self.observation,
				self.reward, self.terminal, self.epsTest, not self.training)
