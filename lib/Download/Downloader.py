# coding: utf-8
# 多线程下载

import os
import sys
import requests
import xmltodict
import socket
from six.moves import queue as Queue
from threading import Thread
import re
import ssl
import urllib
import urllib2
 
ssl._create_default_https_context = ssl._create_unverified_context

from File.FilePath import *
from File.loop_file import *


class DownloadThread(Thread):
	def __init__(self, queue, timeout, retry):
		Thread.__init__(self)
		self.queue = queue
		self.timeout = timeout
		self.retry = retry

	def run(self):
		while True:
			url, save_path = self.queue.get()
			self.download(url, save_path)
			self.queue.task_done()

	def download(self, url, save_path):
		socket.setdefaulttimeout(self.timeout)

		if not os.path.isfile(save_path):
			print("Downloading %s from %s\n" % (save_path, url))
			retry_times = 0
			while retry_times < self.retry:
				try:
					urllib.urlretrieve(url, filename=save_path)
					if not os.path.isfile(save_path):
						data = urllib2.urlopen(url).read()
						with open(save_path,'wb') as f:
							f.write(data)
					break
				except Exception as e:
					print e
				retry_times += 1
			else:
				try:
					os.remove(save_path)
				except OSError:
					pass
				print("Failed to retrieve from %s\n" % (url))


class Downloader(object):
	# 参数: 
	# thread_count: 线程数
	# timeout: 等待超时
	# retry: 失败重试次数
	def __init__(self, thread_count = 10, timeout = 10, retry = 5):
		self.queue = Queue.Queue()
		self.thread_count = thread_count
		self.timeout = timeout
		self.retry = retry
		self.scheduling()

	def scheduling(self):
		for x in range(self.thread_count):
			worker = DownloadThread(self.queue, self.timeout, self.retry)
			worker.daemon = True
			worker.start()

	def download(self, download_list):
		# download_list: [[url, save_path], ...]
		for item in download_list:
			self.queue.put((item[0], item[1]))
		self.queue.join()

		print '队列完成'



