#coding=utf-8

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

    def add(self, step, state, terminal):
        tmp = {"step":step, "state":state, "terminal":terminal}
        self.buf.append(tmp)
        self.nowEpisode(tmp)

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

    def get(self, i):
        return self.buf[i]
