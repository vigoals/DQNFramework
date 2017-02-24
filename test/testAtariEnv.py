import sys
sys.path.append('.')

import gameEnv
game = gameEnv.AtariEnv('Breakout-v0', 1, 30)
observation, reward, terminal = game.newGame()
game.render()

while True:
	a = input()
	if not terminal:
		observation, reward, terminal = game.step(a, True)
	else:
		observation, reward, terminal = game.nextRandomGame(True)
		print 'Terminal'
	game.render()
