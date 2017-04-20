# coding: utf-8
import json
import flickrapi
import account
import urllib

flickr = flickrapi.FlickrAPI(account.api_key, account.api_secret)
for photo in flickr.walk(
                         tag_mode='all',
                         tags='winter,yosemite,landscape',
                         extras='url_l'):
	print photo.attrib

# https://www.flickr.com/photos/62060901@N06/34098114936/

# 

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

# image = urllib.URLopener()
# image.retrieve('https://farm3.staticflickr.com/2824/34098114936_80e5b77724_b.jpg', 'b.jpg')