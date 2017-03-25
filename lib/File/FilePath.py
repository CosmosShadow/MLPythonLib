# coding: utf-8

import os
import shutil

def FileNameOfPath(FilePath):
	return os.path.basename(FilePath).split('.')[0]

def BaseNameOfPath(FilePath):
	return os.path.basename(FilePath)

# 强制新建文件夹: 如果已有，则删除文件夹，然后再新建
def removeAndCreateDir(dirPath):
	if os.path.exists(dirPath):
		shutil.rmtree(dirPath, ignore_errors=True)
	os.makedirs(dirPath)

def NoExistsCreateDir(dirPath):
	if not os.path.exists(dirPath):
		os.makedirs(dirPath)

def check_dir(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)


def clear_dir(dir):
	if os.path.exists(dir):
		shutil.rmtree(dir, ignore_errors=True)
	os.makedirs(dir)