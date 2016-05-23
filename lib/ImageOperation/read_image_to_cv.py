#coding: utf-8

from PIL import Image
from PIL import ExifTags
from correct_image_orientation import *
import numpy

def read_image_to_cv(image_path):
	image = Image.open(image_path)
	image_correct = correct_image_orientation(image)
	pil_image = image_correct.convert('RGB') 
	open_cv_image = numpy.array(pil_image) 
	open_cv_image = open_cv_image[:, :, ::-1].copy()
	return open_cv_image