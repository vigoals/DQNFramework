# 安装
	./installDep.sh
	
# 使用方法
	./main.py [options] <env>

	options:
		-c, --configure
			配置文件，具体见下方参数说明。

		-d, --device
			tensorflow使用设备(/cpu:0, /gpu:0等)

		-s, --savepath
			数据保存路径

	env:
		gym说运行的游戏环境，atari游戏只需要游戏名，不需要后面的版本号。

# 参数说明
	device
		tensorflow所使用的设备，以命令行参数为优先
	gameEnv
		使用的游戏环境
	agent
		使用的agent
	buf
		使用的memory replay
	convLayers
		卷积层设置
	linearLayers
		线性层设置
	render
		游戏是否显示
	steps
		运行的最大步数
	epsEnd
		最终epsilon
	epsEndT
		达到最终epsilon的步数
	epsTest
		测试时用的epsilon
	bufSize
		memory replay大小
	discount
	learnStart
		开始训练的步数
	clipDelta
	learningRate
	trainFreq
		每个多少步训练
	evalBatchSize
		评估时的batch大小
	targetFreq
		target网络更新频率
	reportFreq
		报告频率
	evalFreq
		评估频率
	evalMaxEpisode
		最大评估episode
	evalMaxSteps
		最大评估步数
	saveFreq
		保存频率
	savePath
		保存路径，以命令行参数为先

还有一些其他参数，可以参考具体的代码，及opts下的文件。
