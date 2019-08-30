#!/usr/bin/env python
# coding=utf-8

from __future__ import division
# import cv2 as cv
import cv2
import numpy as np
import matplotlib.pyplot as plt

def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image
    print(x, y)
 
    # loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(250):
            if x + i >= rows or y + j >= cols:
                continue
            # alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
            alpha = 0.5
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src


lena = cv2.imread('lena.jpg')
txt_img = cv2.imread('transparent_img.png')

yoffset = 200
# lena[yoffset : yoffset + txt_img.shape[0], 0: txt_img.shape[1]] = txt_img
new_img = lena

# lena = transparentOverlay(lena, txt_img, pos=(0, 200))
# create mask
x_m = 150
y_m = 200
w_m = 250
h_m = 50
# mask = np.zeros(new_img.shape, np.uint8)
# mask[y_m : y_m + h_m, x_m : x_m + w_m] = new_img[y_m : y_m + h_m, x_m : x_m + w_m]
mask = np.zeros((50, 250, 1), np.uint8)
print(mask.shape[0], mask.shape[1])
print(new_img.shape[0], new_img.shape[1])
mask_inv = cv2.bitwise_not(mask)
print(mask_inv.shape[0], mask_inv.shape[1])

# bitwise operation
roi = new_img[y_m : y_m + h_m, x_m : x_m + w_m]
print(roi.shape[0], roi.shape[1])
fg = cv2.bitwise_and(roi, roi, mask = mask_inv)
print('fg w:{} h:{}'.format(fg.shape[0], fg.shape[1]))
# new_img[y_m : y_m + h_m, x_m : x_m + w_m] = tmp

roi_txt = txt_img[0: 50, 0 : w_m]
bg = cv2.bitwise_and(roi_txt, roi_txt, mask = mask)
txt_img[: h_m, x_m : x_m + w_m] = fg
print('bg w:{} h:{}'.format(bg.shape[0], bg.shape[1]))

# dst = cv2.add(fg, bg)
# new_img[y_m : y_m + h_m, x_m : x_m + w_m] = dst

# new_img[y_m : y_m + h_m, x_m : x_m + w_m] = txt_img
new_img[yoffset : yoffset + txt_img.shape[0], 0: txt_img.shape[1]] = txt_img

cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('image', new_img)
# cv2.imshow('image', txt_img)

cv2.waitKey()
