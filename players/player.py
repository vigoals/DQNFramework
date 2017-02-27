#coding=utf-8

import gameEnv
import agents
import time

class Player(object):
	def __init__(self, opt, agent=None):
		exec('Env = ' + opt.get('gameEnv'))
		self.env = opt.get('env')
		self.actrep = opt.get('actrep', 4)
		self.randomStarts = opt.get('randomStarts', 30)
		self.gameEnv = Env(self.env, self.actrep, self.randomStarts)
		self.nActions = opt.get('nActions', self.gameEnv.getActions())

		exec('AGENT = ' + opt.get('agent'))
		self.agent = agent if agent is not None else AGENT(opt)

		# run 所要用到的数据
		self.reset()

	def reset(self, training=True):
		self.step = 0
		self.episode = 0
		self.action = 0
		self.totalReward = 0
		self.observation, self.reward, self.terminal = self.gameEnv.newGame()
		self.episodeReward = self.reward
		self.training = training
		self.startTime = time.time()

		return self.observation, self.reward, self.terminal

	def oneStep(self, training=True):
		if not self.terminal:
			self.observation, self.reward, self.terminal = \
					self.gameEnv.step(self.action, training=training)
		else:
			self.observation, self.reward, self.terminal = \
					self.gameEnv.nextRandomGame(training=training)

		return self.observation, self.reward, self.terminal

	def run(self, maxSteps=None, maxEpisode=None, training=True):
		assert (maxSteps is not  None) or (maxEpisode is not None), \
				"游戏无法结束"

		self.reset(training)

		self.onStartRun()

		while True:
			# self.gameEnv.render()
			# action = self.gameEnv.sample()
			self.onStartStep()

			self.oneStep(training)

			if not self.terminal:
				self.episodeReward += self.reward
			else:
				self.totalReward += self.episodeReward
				self.episode += 1
				self.onEndEpisode()
				self.episodeReward = 0

			self.onEndStep()

			self.step += 1
			if maxSteps is not None and self.step >= maxSteps:
				break
			if maxEpisode is not None and self.episode >= maxEpisode:
				break

		self.endTime = time.time()

		self.onEndRun()

	def onStartRun(self):
		pass

	def onEndRun(self):
		pass

	def onStartStep(self):
		pass

	def onEndStep(self):
		pass

	def onEndEpisode(self):
		pass

	def getInfo(self):
		return {
			'runTime' : int(self.endTime - self.startTime),
			'totalReward' : self.totalReward,
			'episode' : self.episode,
			'avgReward' : float(self.totalReward)/self.episode,
			'runStep' : self.step
		}
