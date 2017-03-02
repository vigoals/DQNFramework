#coding=utf-8

class BaseAgent(object):
    """Agent基类"""
    def __init__(self, opt):
		pass

    def perceive(self, step, observation, reward, terminal, ep, eval):
        return 0, 0

	def report(self):
		pass

	def save(self, savePath):
		pass
