# coding: utf-8
# 多线程: 生产、消费模式

from Queue import Queue
import random
import threading
import time

lock = threading.Lock()
running = True

#Producer thread
class Producer(threading.Thread):
	def __init__(self, t_name, queue):
		threading.Thread.__init__(self, name=t_name)
		self.data = queue

	def run(self):
		global lock, running
		while True:
			while self.data.qsize() > 3:
				time.sleep(2)
				with lock:
					state = running
				if not state:
					break

			with lock:
				state = running
			if not state:
				break

			print '生产一个\n'
			self.data.put(1)

		print '生产结束'


class Consumer(threading.Thread):
	def __init__(self, t_name, queue):
		threading.Thread.__init__(self, name=t_name)
		self.data=queue

	def run(self):
		global lock, running

		for i in range(5):
			val = self.data.get()
			print '消费一个: %s\n' % time.ctime()
			time.sleep(5)

		with lock:
			running = False
		print '消费结束'



if __name__ == '__main__':
	queue = Queue()
	producer = Producer('生产者', queue)
	consumer = Consumer('消费者', queue)
	producer.start()
	consumer.start()
	producer.join()
	consumer.join()

	print '所有都结束了'





