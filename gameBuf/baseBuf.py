#coding=utf-8
import numpy as np

class BaseBuf(object):
	"""docstring for BaseBuf."""
	def __init__(self, opt):
		super(BaseBuf, self).__init__()
		self.size = opt.get('bufSize', 10000)
		self.reset()

	def reset(self, size=None):
		self.size = size or self.size
		self.buf = []
		self.nowEpisode = []
		self.episodes = [self.nowEpisode]

	def statePreProcess(self, state):
		return state.copy()

	def add(self, step, state, terminal):
		state = self.statePreProcess(state)
		tmp = {'step':step, 'state':state, 'action':None, \
				'reward':None, 'terminal':terminal}
		self.buf.append(tmp)
		self.nowEpisode.append(tmp)

		if terminal:
			self.nowEpisode = []
			self.episodes.append(self.nowEpisode)

		if len(self.buf) > self.size + len(self.episodes[0]):
			self.buf = self.buf[len(self.episodes[0]):]
			self.episodes = self.episodes[1:]

	def setAction(self, action):
		tmp = self.buf[-1]
		tmp['action'] = action

	def setReward(self, reward):
		tmp = self.buf[-1]
		tmp['reward'] = reward

	def getState(self, i=None):
		i = i if i is not None else -1
		i = i if i >= 0 else (len(self.buf) + i)
		assert 0 <= i < len(self.buf), '超出范围'
		return self.buf[i]['state'].copy()

	def getStateByStep(self, step):
		i = step - self.buf[0]['step']
		assert self.buf[i]['step'] == step, 'step 计数出错'
		return self.getState(i)

	def get(self, i=None):
		i = i if i is not None else -1
		i = i if i >= 0 else (len(self.buf) + i)
		assert 0 <= i < len(self.buf), '超出范围'
		return self.buf[i]['step'], self.getState(i), self.buf[i]['action'], \
				self.buf[i]['reward'], self.buf[i]['terminal']

	def getByStep(self, step):
		i = step - self.buf[0]['step']
		assert self.buf[i]['step'] == step, 'step 计数出错'
		return self.get(i)

	def sample(self, n):
		i = 0
		steps = []
		state = []
		reward = []
		action = []
		terminal = []
		stateNext = []

		while i < n:
			k = np.random.randint(len(self.buf) - 1)
			if not self.buf[k]['terminal']:
				step, s1, a, r, _= self.get(k)
				_, s2, _, _, t = self.get(k + 1)
				steps.append(step)
				state.append(s1)
				reward.append(r)
				action.append(a)
				terminal.append(t)
				stateNext.append(s2)
				i += 1

		batch = {
			'steps' : steps,
			'state' : np.array(state),
			'reward' : np.array(reward).astype(np.float),
			'action' : action,
			'terminal' : np.array(terminal).astype(np.float),
			'stateNext' : np.array(stateNext)
		}

		return batch

	def __len__(self):
		return len(self.buf)
