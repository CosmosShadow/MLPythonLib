# coding: utf-8
import json
import flickrapi
import account

flickr = flickrapi.FlickrAPI(account.api_key, account.api_secret)
for photo in flickr.walk(tag_mode='all',
        tags='winter,yosemite,landscape'):
    print photo.attrib
    break

# 