# coding: utf-8

# 队列栈: 长度小于一个数，压入的时候，自动把前面的给压掉
class QueueStack():
	def __init__(self, maxSize):
		self.maxSize = maxSize
		self.size = 0
		self.arrData = []

	def push(self, data):
		if self.size < self.maxSize:
			self.size += 1
		else:
			self.arrData.pop(0)
		self.arrData.append(data)

if __name__ == "__main__":
	queueStack = QueueStack(3)
	for i in xrange(1,100):
		queueStack.push(i)

	# 97, 98, 99
	print queueStack.arrData

	# 99
	print queueStack.arrData[-1]





