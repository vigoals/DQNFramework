#coding=utf-8

import gym
from env import Env
import numpy as np

class AtariEnv(Env):
	def __init__(self, env, actrep=1, randomStarts=1):
		Env.__init__(self, env, actrep, randomStarts)
		self.reset()

	def reset(self):
		Env.reset(self)
		self.game = gym.make(self.env)
		assert type(self.game) == gym.envs.atari.AtariEnv, "只支持atari游戏"
		self.actionSpace = self.game.action_space
		observation = self.game.reset()
		self.lives = 0
		self._updateState(observation, 0, False, -1)

	def render(self):
		self.game.render()

	def _updateState(self, observation, reward, terminal, lives):
		self.observation = observation
		self.reward = reward
		self.lives = lives
		self.terminal = terminal

	def getState(self):
		return self.observation, self.reward, self.terminal

	def sample(self):
		return self.actionSpace.sample()

	def _randomStep(self):
		return self.game.step(self.sample())

	def step(self, action, training=False):
		rewardSum = 0
		terminal = False
		observation = None
		lives = None

		for _ in range(self.actrep):
			observation, reward, terminal, info = self.game.step(action)
			rewardSum += reward
			# atari游戏判断是否存活
			if info.has_key('ale.lives'):
				lives = info['ale.lives']
			else:
				lives = None
			if training and lives and lives < self.lives:
				terminal = True

			if terminal:
				break

		self._updateState(observation, rewardSum, terminal, lives)
		return self.getState()

	def newGame(self):
		self.game.reset()
		observation, reward, terminal, info = self.game.step(0)
		lives = info.has_key('ale.lives') and info['ale.lives'] or None

		self._updateState(observation, reward, terminal, lives)
		return self.getState()

	def nextRandomGame(self, k=None, training=False):
		observation = self.observation
		reward = self.reward
		terminal = self.terminal
		lives = self.lives
		k = k or np.random.randint(0, self.randomStarts)

		if not training:
			observation, reward, terminal = self.newGame()
		else:
			observation, reward, terminal, info = self.game.step(0)
			# while not terminal:
			# 	observation, reward, terminal, info = self.game.step(0)
			# 	lives = info.has_key('ale.lives') and info['ale.lives'] or None
			# 	if training and lives and lives < self.lives:
			# 		break
			if terminal:
				self.game.reset()

		for i in range(k):
			observation, reward, terminal, info = self.game.step(0)
			if terminal:
				print "WARNING: Terminal signal received after %d 0-steps" % (i+1)
				break

		observation, reward, terminal, info = self.game.step(0)
		self._updateState(observation, reward, terminal, lives)
		return self.getState()

	def getActions(self):
		return self.game.action_space.n
