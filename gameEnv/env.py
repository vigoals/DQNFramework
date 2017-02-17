class Env:
	def __init__(self, env, actrep=1, randomStarts=1):
		self.actrep = actrep
		self.randomStarts = randomStarts
		self.env = env

	def reset(self):
		pass

	def render(self):
		pass

	def step(self, action, training):
		pass

	def newGame(self):
		pass

	def nextRandomGame(self, k):
		pass

	def getActionSpace(self):
		pass

	def getObservationSpace(self):
		pass
