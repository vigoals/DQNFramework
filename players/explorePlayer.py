#coding=utf-8

from player import Player
import time
from evalPlayer import EvalPlayer

class ExplorePlayer(Player):
	"""用于学习过程探索游戏环境"""
	def __init__(self, opt, agent=None):
		super(ExplorePlayer, self).__init__(opt, agent)
		self.reportFreq = opt.get('reportFreq', 10000)
		self.learnStart = opt.get('learnStart', 50000)
		self.epsEnd = opt.get('epsEnd', 0.1)
		self.epsEndT = opt.get('epsEndT', 1e6)
		self.render = opt.get('render', False)
		self.evalFreq = opt.get('evalFreq', 1e5)
		self.evalMaxSteps = opt.get('evalMaxSteps', 125000)
		self.evalMaxEpisode = opt.get('evalMaxEpisode', 30)
		self.evalPlayer = EvalPlayer(opt, self.agent)
		self.saveFreq = opt.get('saveFreq', 10000)
		self.savePath = opt.get('savePath', './save')

	def onStartStep(self):
		ep = 1
		if self.step >= self.epsEndT:
			ep = self.epsEnd
		elif self.step > self.learnStart:
			ep = self.epsEnd + \
					(self.epsEndT - self.step)*(1 - self.epsEnd)/ \
					(self.epsEndT - self.learnStart)

		if self.render:
			self.gameEnv.render()

		self.action, q = self.agent.perceive(self.step, self.observation,
				self.reward, self.terminal, ep, not self.training)

	def report(self):
		print
		print "Report in step %8d" % self.step
		t = int(time.time() - self.startTime)
		print "Run time:%10d" % t
		self.agent.report()

	def eval(self):
		print
		print "Eval in step %8d" % self.step
		t = int(time.time() - self.startTime)
		print "Run time:%10d" % t

		self.evalPlayer.run(self.evalMaxSteps, self.evalMaxEpisode, False)
		info = self.evalPlayer.getInfo()
		info['step'] = self.step
		info['time'] = t

		print "evalRunTime:%5d, evalRunStep:%7d, "\
				"totalReward:%10d, episode:%6d, avgReward:%10d" % \
				(info['runTime'], info['runStep'],
				info['totalReward'], info['episode'], info['avgReward'])

	def save(self):
		pass

	def onEndStep(self):
		if self.step%self.reportFreq == 0:
			self.report()

		if self.step > self.learnStart and \
				self.evalFreq > 0 and \
				self.step%self.evalFreq == 0:
			self.eval()

		if self.step > self.learnStart and \
				self.saveFreq > 0 and \
				self.step%self.saveFreq == 0:
			self.save()
