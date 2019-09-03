#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import time
import threading

IMG_WIDTH = 512
IMG_HEIGHT = 512

def create_text_plane(w, h, j):
    n_channels = 3
    txt_plane = np.zeros((h, w, n_channels), dtype=np.uint8)

    pos = [(0, 30), (30, 70), (60, 110), (90, 150), (120, 190),
           (150, 230), (120, 270), (180, 310), (42, 350), (140, 390),
           (0, 420), (82, 460), (300, 500)]
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
           "when true, the image data origin is at the bottom"]

    font = cv.FONT_HERSHEY_SIMPLEX
    for i, p in enumerate(pos):
        cv.putText(txt_plane, strs[i], (p[0] + w - j, p[1]), font, 1,
                   (255, 255, 255), 1, cv.LINE_AA)

    return txt_plane

def running_text_thread(name):
    j = 0
    while True:
        plane = create_text_plane(IMG_WIDTH, IMG_HEIGHT, j)
        show(plane)
        time.sleep(0.05)

        j += 1
        # todo: how to calc speed
        # print(j)
        if (j == 1500):
            j = 0

def start_running_txt_thread():
    running = threading.Thread(target=running_text_thread, args=(1,))
    running.start()

def show(img):
    # cv.namedWindow('image', cv.WINDOW_AUTOSIZE)
    cv.imshow('image', img)

    if cv.waitKey(5) == 27:
        return

if __name__ == '__main__':
    cv.namedWindow('image', cv.WINDOW_AUTOSIZE)
    start_running_txt_thread()

    for i in range(10):
        print(i)
    
    cv.destroyAllWindows()
