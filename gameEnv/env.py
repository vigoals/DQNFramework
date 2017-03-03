class Env:
	def __init__(self, env):
		self.env = env

	def render(self):
		pass

	def step(self, action, training):
		pass

	def newGame(self):
		pass

	def nextRandomGame(self, k):
		pass

	def getActions(self):
		return None

	def getObservationSpace(self):
		return None, None, None
