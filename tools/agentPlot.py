#!/usr/bin/python
from toolsComm import *

if __name__ == '__main__':
	opt = OptionParser()
	# opt.set('render', True)
	opt.set('device', '/cpu:0')
	player = players.Player(opt)

	savePath = opt.get('savePath')
	agent = player.agent
	buf = agent.evalBuf
	agent.load(savePath)

	observation, reward, terminal = player.reset(False)
	qAll = []
	r = []
	t = []
	for i in range(500):
		player.action, _ = agent.perceive(
				i, observation, reward, terminal, 1, True)

		r.append(reward)
		t.append(terminal)

		state = buf.getState()
		qAll.append(agent.q([state]).reshape(-1))
		observation, reward, terminal = player.oneStep(False)

	qAll = np.array(qAll)

	# plt.hold(False)
	for i in range(qAll.shape[1]):
		plt.plot(qAll[:, i], label=str(i))


	plt.plot(r, label='reward')
	plt.plot(np.array(t).astype(np.float), label='terminal')
	plt.legend()
	plt.show()
