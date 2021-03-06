#!\urs\bin\env python
#encoding:utf-8       #设置编码方式  

# 作用: 循环读取文件
# 用法: 
# dataDirPath = './data/'
# lf = loop_file(dataDirPath, [], [], ['.txt'])
# for f in lf.start(lambda f: f):
#     print f

import os
import re
class loop_file:
	def __init__(self, root_dir, file_extend=[], short_exclude=[], long_exclude=[]):
		self.root_dir = root_dir
		self.short_exclude = short_exclude
		self.long_exclude = long_exclude
		self.file_extend = file_extend
	
	def __del__(self):
		pass
	
	def start(self, func):
		self.func = func
		return self.loop_file(self.root_dir)
	
	def loop_file(self, root_dir):
		t_sum = []
		sub_gen = os.listdir(root_dir)
		for sub in sub_gen:
			is_exclude = False
			for extends in self.short_exclude:  ##在不检查文件、目录范围中
				if extends in sub:              ##包含特定内容
					is_exclude = True
					break
				if re.search(extends, sub):     ##匹配指定正则
					is_exclude = True
					break                    
			if is_exclude:
				continue            
			abs_path = os.path.join(root_dir, sub)
			is_exclude = False
			for exclude in self.long_exclude:
				if exclude == abs_path[-len(exclude):]:
					is_exclude = True
					break
			if is_exclude:
				continue
			if os.path.isdir(abs_path):
				t_sum.extend(self.loop_file(abs_path))
			elif os.path.isfile(abs_path):
				if len(self.file_extend) > 0:
					if not "." + abs_path.rsplit(".", 1)[1] in self.file_extend:  ##不在后缀名 检查范围中
						continue
				t_sum.append(self.func(abs_path))
		return t_sum


def file_count(dirname, filter_types=[]):
	 '''Count the files in a directory includes its subfolder's files
		You can set the filter types to count specific types of file'''
	 count=0
	 filter_is_on=False
	 if filter_types!=[]: filter_is_on=True
	 for item in os.listdir(dirname):
		 abs_item=os.path.join(dirname,item)
		 #print item
		 if os.path.isdir(abs_item):
			 #Iteration for dir
			 count+=file_count(abs_item,filter_types)
		 elif os.path.isfile(abs_item):
			 if filter_is_on:
				 #Get file's extension name
				 extname=os.path.splitext(abs_item)[1]
				 if extname in filter_types:
					 count+=1
			 else:
				 count+=1
	 return count


def list_dir(dir_path, extension=[]):
	lf = loop_file(dir_path, extension)
	return lf.start(lambda f: f)


if '__main__'==__name__:
	root_dir = "."
	short_exclude = ['.svn', '.*_new.rb']     ###不包含检查的短目录、文件
	long_exclude = []                         ###不包含检查的长目录、文件
	file_extend = ['.py']                     ###包含检查的文件类型
	lf = loop_file(root_dir, file_extend, short_exclude, long_exclude)
	for f in lf.start(lambda f: f):
		print f

	print "count of file with .py extention: " + str(file_count(".", [".py"]))

	print list_dir('.')