#coding=utf-8

from player import Player
import time
from evalPlayer import EvalPlayer
import json
import comm

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
		self.evalInfo = []
		self.maxEvalReward = -1

	def reset(self):
		super(ExplorePlayer, self).reset()
		if len(self.evalInfo) > 0:
			self.step = self.evalInfo[-1].step

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

		self.evalPlayer.run(self.evalMaxSteps, self.evalMaxEpisode,
			training=False)
		info = self.evalPlayer.getInfo()
		info['step'] = self.step
		info['time'] = t

		print "evalRunTime:%5d, evalRunStep:%7d, "\
				"totalReward:%10d, episode:%6d, avgReward:%10.5f" % \
				(info['runTime'], info['runStep'],
				info['totalReward'], info['episode'], info['avgReward'])

		self.evalInfo.append(info)

		if info['avgReward'] > self.maxEvalReward:
			self.maxEvalReward = info['avgReward']
			self.agent.save(self.savePath, 'best')

	def save(self):
		self.agent.save(self.savePath)

		path = self.savePath + '/evalInfo.json'
		try:
			str_ = json.dumps(self.evalInfo, indent=4,
					sort_keys=False, ensure_ascii=False)
			f = open(path, 'w')
			f.write(str_)
			f.close()
		except IOError:
			print 'Error: 保存evalInfo出错。'
			exit()

	def load(self):
		self.agent.load(self.savePath)
		path = self.savePath + '/evalInfo.json'

		try:
			self.evalInfo = loadJsonFromFile(path)
		except IOError:
			self.evalInfo = []

	def onEndStep(self):
		if self.step%self.reportFreq == self.reportFreq - 1:
			self.report()

		if self.step > self.learnStart and \
				self.evalFreq > 0 and \
				self.step%self.evalFreq == self.evalFreq - 1:
			self.eval()

		if self.step > self.learnStart and \
				self.saveFreq > 0 and \
				self.step%self.saveFreq == self.saveFreq - 1:
			self.save()
