#coding=utf-8
import json

def byteify(self, input):
	if isinstance(input, dict):
		return {self.byteify(key): self.byteify(value)
				for key, value in input.iteritems()}
	elif isinstance(input, list):
		return [self.byteify(element) for element in input]
	elif isinstance(input, unicode):
		return input.encode('utf-8')
	else:
		return input

def loadJsonFromFile(self, file_):
	f = open(file_, 'r')
	tmp = json.load(f)
	tmp = self.byteify(conf)
	return tmp
