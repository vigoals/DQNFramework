#coding=utf-8
import sys
import optparse
import json

class OptionParser:
    def __init__(self):
        parser = optparse.OptionParser(usage="usage:./main.py [optinos] env")
        parser.add_option("-c", "--configure",
                action = "store",
                type = 'string',
                dest = "configure",
                default = None,
                help="配置文件"
                )
        (options, args) = parser.parse_args(sys.argv[1:])
        self.options = options.__dict__
        assert len(args) == 1, "usage:./main.py [optinos] env"
        self.options['env'] = args[0]
        if self.options['configure']:
            self.load(self.options['configure'])
            self.options['configure'] = None

    def get(self, key, default=None):
        if self.options.has_key(key):
            return self.options[key]
        elif default is not None:
            self.options[key] = default
            return default
        else:
            return None

    def load(self, configure):
        try:
            f = open(configure, 'r')
            conf = json.load(f)
            for k in conf.keys():
                if not self.options.has_key(k.encode()):
                    self.options[k.encode()] = conf[k].encode()
        except IOError:
            pass

    def save(self):
        pass
