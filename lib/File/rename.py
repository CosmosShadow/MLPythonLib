#coding: utf-8

from File.loop_file import *
from File.FilePath import *
import shutil 
import sys

if len(sys.argv) != 5:
	print 'usage: move_serval_files.py from_dir to_dir type count'
	exit()

dir_from = sys.argv[1]
dir_to = sys.argv[2]
file_type = sys.argv[3]
count = int(sys.argv[4])

print count

index = 0
lf = loop_file(dir_from, [], [], [''+file_type])
for f in lf.start(lambda f: f):
	base_name = BaseNameOfPath(f)
	if not base_name.startswith('.'):
		store_path = dir_to + '/' + base_name
		shutil.move(f, store_path)
		index += 1
		print index
		if index >= count:
			break