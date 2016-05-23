#coding: utf-8

import cv2
import numpy as np

img = cv2.imread('1.jpg')
cv2.rectangle(img,(0, 0),(200, 100),(0,255,0),3)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 2
font_color = (255,255,255)
font_line_width = font_size
cv2.putText(img, 'OpenCV', (10, 500), font, font_size, font_color, font_line_width, 2)
cv2.imwrite('2.jpg', img)
