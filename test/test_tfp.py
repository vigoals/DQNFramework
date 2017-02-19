#!/usr/bin/python
#coding=utf-8
import sys
sys.path.append(".")
import tfPack as tfp
import numpy as np


tfp.setDevice("/gpu:0")
net = tfp.Network()

l = net.add(tfp.Placeholder([None, 10]))
l = net.add(tfp.Linear(l, 10, 10))

tfp.initAllVariables()
print net.forward(np.random.rand(5, 10))
