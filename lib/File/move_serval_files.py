#coding: utf-8

from File.loop_file import *
from File.FilePath import *
import shutil 
import sys
import time

dir_from = '/Users/lichen/Desktop/png/'
dir_to = '/Users/lichen/Desktop/to/'
types = ['.png']


lf = loop_file(dir_from, [], [], types)
for f in lf.start(lambda f: f):
	base_name = BaseNameOfPath(f)
	store_path = dir_to + 'M_*_**_' + str(int(time.time()*1000000)) + '.png'
	shutil.copy(f, store_path)
	# index += 1
	# print index
	# if index >= count:
	# 	break