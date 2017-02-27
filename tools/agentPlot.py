#!/usr/bin/python
from toolsComm import *

if __name__ == '__main__':
	opt = OptionParser()
	# opt.set('render', True)
	opt.set('trainFreq', 0)
	player = players.Player(opt)

	savePath = opt.get('savePath', 'best')
	agent = player.agent
	buf = agent.gameBuf
	agent.load(savePath)

	observation, reward, terminal = player.reset(True)
	qAll = []
	r = []
	t = []
	for i in range(1000):
		player.action, _ = agent.perceive(
				i, observation, reward, terminal, 0.05, False)

		r.append(reward)
		t.append(terminal)

		state = buf.getState()
		qAll.append(agent.q([state]).reshape(-1))
		observation, reward, terminal = player.oneStep(True)

	agent.report()
	qAll = np.array(qAll)

	# plt.hold(False)
	for i in range(qAll.shape[1]):
		plt.plot(qAll[:, i], label=str(i))


	plt.plot(r, label='reward')
	plt.plot(np.array(t).astype(np.float), label='terminal')
	plt.legend(loc='upper right')
	plt.show()
