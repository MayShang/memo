#!/usr/bin/env python
# coding=utf-8
from __future__ import division
import cv2 as cv
# import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load two images
img1 = cv.imread('lena.jpg')
img2 = cv.imread('opencv-logo.png')
img2small = cv.resize(img2, (200, 200), interpolation=cv.INTER_LINEAR)
# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2small.shape
print(rows, cols)

roi = img1[0:rows, 0:cols]
# Now create a mask of logo and create its inverse mask also
img2gray = cv.cvtColor(img2small,cv.COLOR_BGR2GRAY)
img1gray = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)

ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)
# Now black-out the area of logo in ROI
img1_bg = cv.bitwise_and(roi, roi, mask = mask_inv)
# img1_bg = cv.bitwise_and(img1,img1,mask = mask_inv)
# Take only region of logo from logo image.
img2_fg = cv.bitwise_and(img2small,img2small,mask = mask)
# Put logo in ROI and modify the main image
dst = cv.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst
cv.imshow('res',img1)
# cv.imshow('res',dst)
# cv.imshow('mask', roi)
# cv.imshow('mask', mask_inv)
cv.waitKey(0)
cv.destroyAllWindows()
