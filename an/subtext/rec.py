#!/usr/bin/env python
# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import time
import threading

# local modules
from video import create_capture
from common import clock, draw_str

class Person:
    def __init__(self, roi, vec, hit):
        self.roi = roi
        self.vec = vec
        self.hit = hit

def show(img):
    cv.namedWindow('image', cv.WINDOW_AUTOSIZE)
    cv.imshow('image', img)
    cv.waitKey(0)

netDet = cv.dnn.readNetFromCaffe('data/face_detector.prototxt', 'data/face_detector.caffemodel')
netRecogn = cv.dnn.readNetFromTorch('data/face_recognition.t7')
persons = []
index = 0
confThreshold = 0.5

def detectFaces_blob(img, netDet):
    faces = []
    frameWidth = img.shape[0]
    frameHeight = img.shape[1]

    # blob = cv.dnn.blobFromImage(img, 1.0, size=(frameWidth, frameHeight), mean=(104, 117, 123), swapRB=False, crop=False, ddepth=cv.CV_8U)
    blob = cv.dnn.blobFromImage(img, 1.0, size=(frameWidth, frameHeight), mean=(104, 117, 123), swapRB=False, crop=False)
    netDet.setInput(blob)
    outs = netDet.forward()

    layerNames = netDet.getLayerNames()
    lastLayerId = netDet.getLayerId(layerNames[-1])
    lastLayer = netDet.getLayer(lastLayerId)
    # print(layerNames, lastLayerId, lastLayer)
    print(lastLayer.type, outs.shape)

    # Network produces output blob with a shape 1x1xNx7 where N is a number of
    # detections and an every detection is a vector of values
    # [batchId, classId, confidence, left, top, right, bottom]
    # print(outs)
    for out in outs:
        # print(out[0, 0], out[0])
        for detection in out[0]:
            # print(detection)
            confidence = detection[2]
            if (confidence > confThreshold):
                print(confidence)
                left = int(detection[3])
                top = int(detection[4])
                right = int(detection[5])
                bottom = int(detection[6])
                width = right - left + 1
                height = bottom - top + 1
                if width * height <= 1:
                    left = int(detection[3] * frameWidth)
                    top = int(detection[4] * frameHeight)
                    right = int(detection[5] * frameWidth)
                    bottom = int(detection[6] * frameHeight)
                    width = right - left + 1
                    height = bottom - top + 1
                    # classIds.append(int(detection[1]) - 1)  # Skip background label
                    # confidences.append(float(confidence))
                    print([left, top, width, height])
                    faces.append([left, top, width, height])

    return faces

def detectFaces_cascade(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def face2vec(face, netRecogn):
    blob = cv.dnn.blobFromImage(face, 1.0 / 255, size=(96, 96), mean=(0, 0, 0), swapRB=True, crop=False)
    netRecogn.setInput(blob)
    vec = netRecogn.forward()
    # print(vec)
    return vec

def draw_rects(img, rects, color):
    face_roi = []
    for x1, y1, x2, y2 in rects:
        # casca
        cv.rectangle(img, (x1, y1), (x2, y2), color)
        # blob
        # cv.rectangle(img, (x1, y1), (x1 + x2, y1 + y2), color)

        roi = img[y1:y2, x1:x2]

        vec = face2vec(roi, netRecogn)
        found = False
        if len(persons):
            for i, p in enumerate(persons):
                score = np.dot(vec, p.vec.T)
                if (score > 0.35):
                    found = True
                    p.hit += 1
                    print('{} found {} times'.format(i, p.hit))

        if (found == False):
            newOne = Person(roi, vec, 0)
            persons.append(newOne)
            # show(roi)

        face_roi.append(roi)
        # show(roi)
    # print(len(face_roi))
    return face_roi

def showResult():
    xoffset = 30
    yoffset = 30
    result_plane = np.zeros((200, 600, 3), dtype=np.uint8)
    lastx = 0

    for i, p in enumerate(persons):
        imgr = cv.resize(p.roi, (100, 100))
        y = yoffset
        h = 100

        w = 100
        x = lastx + xoffset
        result_plane[y : y + h, x : x + w] = imgr
        lastx = x + w
        draw_str(result_plane, (x, 20), 'hit:%d' % p.hit)

    cv.imshow("s", result_plane)
    
    cv.waitKey(0)

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

    # netDet = cv.dnn.readNetFromCaffe('data/face_detector.prototxt', 'data/face_detector.caffemodel')
    # netRecogn = cv.dnn.readNetFromTorch('data/face_recognition.t7')

    cam = create_capture(video_src, fallback='synth:bg={}:noise=0.05'.format(cv.samples.findFile('lena.jpg')))

    while True:
        ret, img = cam.read()
        if img is None:
            break

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)

        # rects = detectFaces_blob(img, netDet)
        rects = detectFaces_cascade(gray, cascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))

        # print('now {} persons'.format(len(persons)))
        cv.imshow('facedetect', vis)
        time.sleep(0.03)

        if cv.waitKey(5) == 27:
            break

    print('total {} persons'.format(len(persons)))
    showResult()

if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()


