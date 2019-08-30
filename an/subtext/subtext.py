#!/usr/bin/env python
# coding=utf-8

from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(img):
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(0)

def create_text_plane(w, h):
    n_channels = 3
    txt_plane = np.zeros((h, w, n_channels), dtype=np.uint8)

    pos = [(0, 30), (50, 70), (60, 110), (40, 150), (80, 190),
           (100, 230), (120, 270), (0, 310), (42, 350), (40, 390),
           (0, 420), (42, 460), (40, 500)]
    strs = ["this function renders the specified text string",
            "symbols that cannot be rendered using the specified font",
           "fast way to draw a group of lines",
           "font scale factor that is multiplied by the font base size",
           "when true, the image data origin is at the bottom-left corner",
           "this function renders the specified text string",
            "symbols that cannot be rendered using the specified font",
           "fast way to draw a group of lines",
           "font scale factor that is multiplied by the font base size",
           "when true, the image data origin is at the bottom-left corner",
           "fast way to draw a group of lines",
           "font scale factor that is multiplied by the font base size",
           "when true, the image data origin is at the bottom-left corner"]

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, p in enumerate(pos):
        cv2.putText(txt_plane, strs[i], p, font, 1,
                    (255, 255, 255), 1, cv2.LINE_AA)
    
    return txt_plane

def sub_txt_bak():
    # load images
    lena = cv2.imread('lena.jpg')
    txt_img = cv2.imread('transparent_img.png')

    yoffset = 200
    new_img = lena

    # create mask
    x_m = 150
    y_m = 200
    w_m = 250
    h_m = 50

    mask = np.zeros((50, 250, 1), np.uint8)
    mask_inv = cv2.bitwise_not(mask)
    print('mask w:{} h:{}, mask_inv w:{} h:{}'.format(mask.shape[0], mask.shape[1],
                                                      mask_inv.shape[0],
                                                      mask_inv.shape[1]))

    # bitwise operation
    roi = new_img[y_m : y_m + h_m, x_m : x_m + w_m]
    fg = cv2.bitwise_and(roi, roi, mask = mask_inv)
    print('fg w:{} h:{}'.format(fg.shape[0], fg.shape[1]))

    txt_img[: h_m, x_m : x_m + w_m] = fg
    new_img[yoffset : yoffset + txt_img.shape[0], 0: txt_img.shape[1]] = txt_img

    return new_img

def sub_txt(txt_img):
    # load images
    lena = cv2.imread('lena.jpg')

    yoffset = 200
    new_img = lena

    # create mask
    x_m = 220
    y_m = 220
    w_m = 150
    h_m = 150

    mask = np.zeros((h_m, w_m, 1), np.uint8)
    mask_inv = cv2.bitwise_not(mask)
    print('mask w:{} h:{}, mask_inv w:{} h:{}'.format(mask.shape[0], mask.shape[1],
                                                      mask_inv.shape[0],
                                                      mask_inv.shape[1]))

    # bitwise operation
    roi = new_img[y_m : y_m + h_m, x_m : x_m + w_m]
    fg = cv2.bitwise_and(roi, roi, mask = mask)
    print('fg w:{} h:{}'.format(fg.shape[0], fg.shape[1]))

    txt_img[y_m : y_m + h_m, x_m : x_m + w_m] = fg
    final = cv2.add(new_img, txt_img)

    return final

def main():
    txt_plane = create_text_plane(512, 512)
    img = sub_txt(txt_plane)
    show(img)

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
