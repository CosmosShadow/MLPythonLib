# coding: utf-8

import sys
import csv

# 存csv文件
def savecsv(path, data):
	csvfile = file(path, 'wb')
	writer = csv.writer(csvfile)
	for row in data:
		writer.writerow(row)
	csvfile.close()

def savecsv_withdelimiter(path, data, delimiter_str):
	csvfile = file(path, 'wb')
	writer = csv.writer(csvfile, delimiter=delimiter_str)
	for row in data:
		writer.writerow(row)
	csvfile.close()

# 取csv文件
def loadcsv(path):
	# 设置field最大限制
	maxInt = sys.maxint
	decrement = True
	while decrement:
	    decrement = False
	    try:
	        csv.field_size_limit(maxInt)
	    except OverflowError:
	        maxInt = int(maxInt/10)
	        decrement = True

	data = []
	csvfile = file(path, 'rb')
	reader = csv.reader(csvfile)
	for line in reader:
		data.append(line)
	csvfile.close()
	return data