#!/bin/sh

echo ' 不包括tensorflow'

sudo pip install gym[all]
sudo pip install matplotlib
sudo pip install deepdish
sudo apt install python-tk

echo '完成'
