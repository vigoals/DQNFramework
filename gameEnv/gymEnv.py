#coding=utf-8

import gym
from env import Env
import numpy as np

class GymEnv(Env):
	def __init__(self, env, *args):
		Env.__init__(self, env)
		self._reset()

	def _reset(self):
		self.game = gym.make(self.env)
		try:
			assert not isinstance(self.game.env, gym.envs.atari.AtariEnv), \
					"atari游戏请用AtariEnv"
		except AttributeError:
			pass
		self.actionSpace = self.game.action_space
		self.observationSpace = self.game.observation_space
		observation = self.game.reset()
		self._updateState(observation, 0, False)

	def render(self):
		self.game.render()

	def _updateState(self, observation, reward, terminal):
		self.observation = observation
		self.reward = reward
		self.terminal = terminal

	def getState(self):
		return self.observation, self.reward, self.terminal

	def sample(self):
		return self.actionSpace.sample()

	def _randomStep(self):
		return self.game.step(self.sample())

	# training 参数无效
	def step(self, action, training=False):
		observation, reward, terminal, _ = self.game.step(action)
		self._updateState(observation, reward, terminal)
		return self.getState()

	def newGame(self):
		observation = self.game.reset()
		self._updateState(observation, 0, False)
		return self.getState()

	# 没有nextRandomGame
	def nextRandomGame(self, k=None, training=False):
		assert False, '不支持randomGame'

	def getActions(self):
		if isinstance(self.actionSpace, gym.spaces.Discrete):
			return self.actionSpace.n
		else:
			assert False, '还未实现'

	def getObservationSpace(self):
		if isinstance(self.observationSpace, gym.spaces.Box):
			return self.observationSpace.shape, \
					self.observationSpace.low, \
					self.observationSpace.high
		else:
			assert False, '还未实现'
