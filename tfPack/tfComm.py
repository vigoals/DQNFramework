#coding=utf-8
import tensorflow as tf

SESS = tf.Session()
DEVICE = "/cpu:0"

def setDevice(device):
    DEVICE = device

def initAllVariables():
    SESS.run(tf.initialize_all_variables())
