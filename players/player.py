#coding=utf-8

import gameEnv

class Player(object):
	def __init__(self, opt, agent=None):
		exec('Env = ' + opt.get('gameEnv'))
		self.env = opt.get('env')
		self.actrep = opt.get('actrep', 4)
		self.randomStarts = opt.get('randomStarts', 30)
		self.gameEnv = Env(self.env, self.actrep, self.randomStarts)
		self.nActions = opt.get('nActions', self.gameEnv.getActionSpace())

		# run 所要用到的数据
		self.observation = None
		self.reward = None
		self.terminal = None
		self.step = 0
		self.episode = 0
		self.action = 0
		self.episode_reward = 0
		self.total_reward = 0

	def run(self, max_steps=None, max_episode=None, training=True):
		assert (max_steps is not  None) or (max_episode is not None), \
				"游戏无法结束"

		self.step = 0
		self.episode = 0
		self.action = 0
		self.total_reward = 0
		self.observation, self.reward, self.terminal = self.gameEnv.newGame()
		self.episode_reward = self.reward

		self.onStartRun()

		while True:
			# self.gameEnv.render()
			# action = self.gameEnv.sample()
			self.onStartStep()

			if not self.terminal:
				self.observation, self.reward, self.terminal = \
						self.gameEnv.step(self.action, training)
				self.episode_reward += self.reward
			else:
				self.onEndEpisode()
				self.observation, self.reward, self.terminal = \
						self.gameEnv.nextRandomGame(training=True)
				self.total_reward += self.episode_reward
				self.episode += 1
				self.episode_reward = 0

			self.onEndStep()

			self.step += 1
			if max_steps is not None and self.step > max_steps:
				break
			if max_episode is not None and self.episode > max_episode:
				break

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
