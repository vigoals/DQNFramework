#!/bin/sh

path_to_nvcc=$(which nvcc)
if [ -x "$path_to_nvcc" ]
then
	sudo pip install tensorflow-gpu
else
	sudo pip install tensorflow
fi

sudo pip install gym[all]
sudo pip install matplotlib

# 不按照python-tk matplotlib会出错，不知道为什么。
sudo apt install python-tk

echo '完成'
