#!/bin/bash
FILES="agents gameBuf gameEnv players test tools opts *.py *.json"

REMOTE_NAME="192.168.2.153" # cse.sysu.edu.cn
REMOTE_USER="vigoals"
REMOTE_DIR="/home/vigoals/remoteRun/DQNFramework"
REMOTE_PORT=22

ENV="breakout"
GPUID=0
DEVICE="/gpu:"$GPUID
SAVEPATH="./save-"$ENV"-"$GPUID"-"$RANDOM
OPT="opts/atari.json"

LOG=$SAVEPATH"/log.txt"

echo "run "$REMOTE_DIR" in "$REMOTE_USER@$REMOTE_NAME " by port " $REMOTE_PORT
echo "GPUID "$GPUID
echo "ENV "$ENV
echo "SAVEPATH "$SAVEPATH

echo '继续[y/n]'
read input
if [ $input != 'y' -a $input != 'Y' ]; then
	echo '结束'
	exit
fi

SSH_CMD="ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_NAME"

$SSH_CMD "mkdir -pv $REMOTE_DIR && cd $REMOTE_DIR && rm -rvf $FILES"
scp -P $REMOTE_PORT -r $FILES $REMOTE_USER@$REMOTE_NAME:$REMOTE_DIR

$SSH_CMD "cd $REMOTE_DIR && mkdir -pv $SAVEPATH "\
"&& nohup ./main.py -c $OPT -d $DEVICE -s $SAVEPATH $ENV >$LOG 2>&1 &"
