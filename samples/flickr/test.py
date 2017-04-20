# coding: utf-8
import json
import flickrapi

flickr = flickrapi.FlickrAPI(api_key, api_secret, format='json')
photos = flickr.photos.search(user_id='73509078@N00', per_page='10')
sets = flickr.photosets.getList(user_id='73509078@N00')

parsed = json.loads(sets.decode('utf-8'))
photoset = parsed['photosets']['photoset']
for photo in photoset:
	print photo


# print photos
# print sets