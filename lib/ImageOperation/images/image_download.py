# coding: utf-8
import ssl
import urllib2

def image_download(url, save_path):
	ssl._create_default_https_context = ssl._create_unverified_context
	image = urllib2.urlopen(url).read()
	with open(save_path, 'wb') as f:
		f.write(image)