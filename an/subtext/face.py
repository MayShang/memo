#!/usr/bin/env python

'''
face detection using haar cascades

USAGE:
    facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import time
import threading

# local modules
from video import create_capture
from common import clock, draw_str

def show(img):
    cv.namedWindow('image', cv.WINDOW_AUTOSIZE)
    cv.imshow('image', img)
    cv.waitKey(0)

def create_text_plane(w, h, offset):
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

    font = cv.FONT_HERSHEY_SIMPLEX
    for i, p in enumerate(pos):
        cv.putText(txt_plane, strs[i], (p[0] + w - offset, p[1]), font, 1,
                    (255, 255, 255), 1, cv.LINE_AA)

    
    return txt_plane

def sub_txt(txt_img, bg_img, x_m, y_m, w_m, h_m):

    mask = np.zeros((h_m, w_m, 1), np.uint8)
    mask_inv = cv.bitwise_not(mask)
    print('mask w:{} h:{}, mask_inv w:{} h:{}'.format(mask.shape[0], mask.shape[1],
                                                      mask_inv.shape[0],
                                                      mask_inv.shape[1]))

    # bitwise operation
    roi = bg_img[y_m : y_m + h_m, x_m : x_m + w_m]
    fg = cv.bitwise_and(roi, roi, mask = mask)
    print('fg w:{} h:{}'.format(fg.shape[0], fg.shape[1]))

    txt_img[y_m : y_m + h_m, x_m : x_m + w_m] = fg
    print('bg_img shape:{} txt_img shape:{}'.format(bg_img.shape,
                                                    txt_img.shape))

    final = cv.add(bg_img, txt_img)

    return final


def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)

def detect_params(rects):
    if len(rects) == 0:
        return 0, 0, 0, 0
    
    x1 = rects[0][0]
    y1 = rects[0][1]
    x2 = rects[0][2]
    y2 = rects[0][3]
    w = x2 - x1
    h = y2 - y1
    print('rects params {} {} {} {}, w:{} y:{}'.format(x1, y1, x2, y2, w, h))
    return x1, y1, w, h

def draw_face_rects(img, rects):
    new_img = img.copy()
    draw_rects(new_img, rects, (0, 255, 0))
    return new_img

def draw_face_mask(img, rects, txt_plane):
    x, y, w, h = detect_params(rects)
    new_img = img
    if w and h : 
        new_img = sub_txt(txt_plane, img, x, y, w, h)
    else:
        new_img = cv.add(img, txt_plane)
    return new_img

def main():
    import sys, getopt
    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "data/haarcascades/haarcascade_frontalface_alt.xml")
    cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))
    cam = create_capture(video_src, fallback='synth:bg={}:noise=0.05'.format(cv.samples.findFile('lena.jpg')))

    offset = 0

    while True:
        ret, img = cam.read()
        txt_plane = create_text_plane(img.shape[1], img.shape[0], offset)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)

        t = clock()
        rects = detect(gray, cascade)
        final = draw_face_mask(img, rects, txt_plane)
        # final = draw_face_rects(img, rects)
        dt = clock() - t
        draw_str(final, (20, 20), 'time: %.1f ms' % (dt*1000))

        cv.imshow('facedetect', final)
        time.sleep(0.03)
        offset += 3

        if cv.waitKey(5) == 27:
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
