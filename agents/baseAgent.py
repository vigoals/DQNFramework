#coding=utf-8

class BaseAgent(object):
    """Agent基类"""
    def __init__(self, opt):
		self.width = opt.get('width', 84)
		self.heigth = opt.get('heigth', 84)
		self.histLen = opt.get('histLen', 4)
