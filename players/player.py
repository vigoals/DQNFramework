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
		self.observation = None
		self.reward = None
		self.terminal = None
		self.step = 0
		self.episode = 0
		self.action = 0
		self.episodeReward = 0
		self.totalReward = 0
		self.training = False
		self.startTime = 0
		self.endTime = 0

	def run(self, max_steps=None, max_episode=None, training=True):
		assert (max_steps is not  None) or (max_episode is not None), \
				"游戏无法结束"

		self.step = 0
		self.episode = 0
		self.action = 0
		self.totalReward = 0
		self.observation, self.reward, self.terminal = self.gameEnv.newGame()
		self.episodeReward = self.reward
		self.training = training
		self.startTime = time.time()

		self.onStartRun()

		while True:
			# self.gameEnv.render()
			# action = self.gameEnv.sample()
			self.onStartStep()

			if not self.terminal:
				self.observation, self.reward, self.terminal = \
						self.gameEnv.step(self.action, training)
				self.episodeReward += self.reward
			else:
				self.onEndEpisode()
				self.observation, self.reward, self.terminal = \
						self.gameEnv.nextRandomGame(training=True)
				self.totalReward += self.episodeReward
				self.episode += 1
				self.episodeReward = 0

			self.onEndStep()

			self.step += 1
			if max_steps is not None and self.step >= max_steps:
				break
			if max_episode is not None and self.episode >= max_episode:
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
			'avgReward' : self.totalReward/self.episode,
			'runStep' : self.step
		}
