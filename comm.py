#coding=utf-8
import json

def byteify(input):
	if isinstance(input, dict):
		return {byteify(key): byteify(value)
				for key, value in input.iteritems()}
	elif isinstance(input, list):
		return [byteify(element) for element in input]
	elif isinstance(input, unicode):
		return input.encode('utf-8')
	else:
		return input

def loadJsonFromFile(file_):
	f = open(file_, 'r')
	tmp = json.load(f)
	tmp = byteify(tmp)
	return tmp
