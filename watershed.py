#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 02:29:31 2019

@author: aylin
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('a1.jpg')
b,g,r = cv2.split(img)
rgb_img = cv2.merge([r,g,b])

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


# noise removal
kernel = np.ones((2,2),np.uint8)
#opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)

cv2.imshow('closing', closing)


# sure background area
sure_bg = cv2.dilate(closing,kernel,iterations=3)
cv2.imshow('sure_bg', sure_bg)



# Finding sure foreground area
dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)
cv2.imshow('dist_transform', dist_transform)
# Threshold
ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

cv2.imshow('unknown', unknown)


# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1




# mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]
img1 = img
cv2.imshow('img', img)


###########


fg = cv2.erode(thresh,None,iterations = 1)
cv2.imshow('fg', fg)



bgt = cv2.dilate(thresh,None,iterations = 5)
ret,bg = cv2.threshold(bgt,1,128,1)
cv2.imshow('bg', bg)


marker = cv2.add(fg,bg)
cv2.imshow('marker', marker)




marker32 = np.int32(marker)



cv2.watershed(img,marker32)
m = cv2.convertScaleAbs(marker32)
cv2.imshow('m', m)

ret,thresh = cv2.threshold(m,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
res = cv2.bitwise_and(img,img,mask = thresh)
cv2.imshow('res', res)

cv2.destroyAllWindows()
