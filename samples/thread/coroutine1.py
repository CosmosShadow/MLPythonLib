# coding: utf-8
import time
import gevent

def f(n):
	for i in range(n):
		gevent.sleep(1)
		print gevent.getcurrent(), i

g1 = gevent.spawn(f, 10)
g2 = gevent.spawn(f, 5)
g3 = gevent.spawn(f, 5)

g1.join()
g2.join()
g3.join()