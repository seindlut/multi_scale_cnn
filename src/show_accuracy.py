#!/usr/bin/evn python

import cPickle as p
f = open('cost_error','rb')
a, b = p.load(f)
f.close()
print 1-min(b)
