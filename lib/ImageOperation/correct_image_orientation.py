#coding: utf-8

from PIL import Image
from PIL import ExifTags

def correct_image_orientation(img):
	try :
		for orientation in ExifTags.TAGS.keys() : 
			if ExifTags.TAGS[orientation]=='Orientation' : break
		exif=dict(img._getexif().items())

		if exif[orientation] == 3 : 
			img=img.rotate(180, expand=True)
		elif exif[orientation] == 6 : 
			img=img.rotate(270, expand=True)
		elif exif[orientation] == 8 : 
			img=img.rotate(90, expand=True)

	except:
		print 'correct_image_orientation wrong'

	return img

if __name__ == '__main__':
	image = Image.open('images/irientation_wrong.jpg')
	image_correct = correct_image_orientation(image)
	image_correct.save('images/irientation_right.jpg')