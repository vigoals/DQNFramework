#coding=utf-8
import sys
import optparse
import json

class OptionParser:
	def __init__(self):
		parser = optparse.OptionParser(usage="usage:./main.py [optinos] env")
		parser.add_option('-c', '--configure',
				action = 'store',
				type = 'string',
				dest = "configure",
				default = None,
				help="配置文件"
				)
		parser.add_option('-s', '--savepath',
				action = "store",
				type = 'string',
				dest = 'savePath',
				default = None,
				help = '保存路径'
				)
		(options, args) = parser.parse_args(sys.argv[1:])
		self.options = options.__dict__
		# assert len(args) == 1, "usage:./main.py [optinos] env"
		if len(args) > 0:
			self.options['env'] = args[0]
		if self.options['configure']:
			self.load(self.options['configure'])
			self.options['configure'] = None

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

	def get(self, key, default=None):
		if self.options.has_key(key):
			return self.options[key]
		elif default is not None:
			self.options[key] = default
			return default
		else:
			return None

	def set(self, key, value):
		self.options[key] = value

	def load(self, configure):
		try:
			f = open(configure, 'r')
			conf = json.load(f)
			conf = self.byteify(conf)
			for k in conf.keys():
				if not self.options.has_key(k) or self.options[k] is None:
					self.options[k] = conf[k]
			f.close()
		except IOError:
			print "WARNING: 载入 %s 出错" % configure

	def save(self, path):
		path = path + '/opt.json'
		try:
			f = open(path, 'w')
			f.write(str(self))
			f.close()
		except IOError:
			print "WARNING: 保存opt到 %s 出错", path

	def __str__(self):
		return json.dumps(self.options, indent=4,
				sort_keys=False, ensure_ascii=False)
