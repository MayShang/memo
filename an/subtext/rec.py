#!/usr/bin/env python
# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import time
import threading
import sys, getopt

# local modules
from video import create_capture
from common import clock, draw_str

class Person:
    def __init__(self, roi, vec, hit):
        self.roi = roi
        self.vec = vec
        self.hit = hit
        self.frames = []

def show(img):
    cv.namedWindow('image', cv.WINDOW_AUTOSIZE)
    cv.imshow('image', img)
    cv.waitKey(0)

netDet = cv.dnn.readNetFromCaffe('data/face_detector.prototxt', 'data/face_detector.caffemodel')
netRecogn = cv.dnn.readNetFromTorch('data/face_recognition.t7')
persons = []
index = 0
confThreshold = 0.5
recogMatchThreshold = 0.5

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

def face_recog(img, rects, frm_num):
    face_roi = []
    for x1, y1, x2, y2 in rects:
        roi = img[y1:y2, x1:x2]

        vec = face2vec(roi, netRecogn)
        found = False
        if len(persons):
            for i, p in enumerate(persons):
                score = np.dot(vec, p.vec.T)
                if (score > recogMatchThreshold):
                    found = True
                    p.hit += 1
                    p.frames.append(frm_num)
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
        # for frm in p.frames:
        #     print(frm)

    cv.imshow("result", result_plane)
    cv.moveWindow("result", 100, 700)

    if cv.waitKey(5) == 27:
        return

def get_frmnum(selected_idx):
    if selected_idx >= len(persons) or selected_idx < 0:
        return []
    for i, p in enumerate(persons):
        if i == selected_idx:
            return p.frames
    return []

def calc_frames(vsrc, casrc):
    cascade = cv.CascadeClassifier(cv.samples.findFile(casrc))

    # netDet = cv.dnn.readNetFromCaffe('data/face_detector.prototxt', 'data/face_detector.caffemodel')
    # netRecogn = cv.dnn.readNetFromTorch('data/face_recognition.t7')

    cam = create_capture(vsrc, fallback='synth:bg={}:noise=0.05'.format(cv.samples.findFile('lena.jpg')))
    frame_num = 0

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
        face_recog(vis, rects, frame_num)
        frame_num += 1

        # print('now {} persons'.format(len(persons)))
        cv.imshow('facedetect', vis)
        cv.moveWindow("facedetect", 100, 100)
        # time.sleep(0.03)

        if cv.waitKey(5) == 27:
            break
    del cam

def disp_video(src, expect_frames):
    expect_cam = create_capture(src)
    frm_idx = 0

    while True:
        ret, imgx = expect_cam.read()
        if imgx is None:
            break

        v = imgx.copy()
        if frm_idx in expect_frames:
            # print('idx in list {}'.format(frm_idx))
            cv.imshow("expect", v)
            cv.moveWindow("expect", 200, 500)
            time.sleep(0.03)
            if cv.waitKey(5) == 27:
                break

        frm_idx += 1

def main():
    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "data/haarcascades/haarcascade_frontalface_alt.xml")

    calc_frames(video_src, cascade_fn)
    
    # print('total {} persons'.format(len(persons)))
    showResult()

    while True:
        msg = input("which people? enter num ")
        if msg == 'x':
            break
        num = int(msg)
        print('you select #{} '.format(num))

        expect_frms = get_frmnum(num)
        print(expect_frms)
        # expect_frms0 = [2, 3, 4, 5, 9, 10, 11, 13, 
        #                33, 34, 35, 37, 39, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 
        #                76, 87, 88, 90, 91, 92, 93, 94, 95, 96, 97, 
        #                144, 154, 156, 158, 167, 168, 174, 175, 177, 178, 179, 180, 183, 231, 232, 234, 235]
        # expect_frms1 = [18, 21, 22, 24, 25, 26, 33, 34, 35, 37, 39, 
        #                 44, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 
        #                 76, 87, 88, 91, 92, 93, 94, 95, 96, 97, 154, 156, 174, 175, 230, 231, 232, 236]
        # expect_frms2 = [66, 69, 70, 71, 74, 75, 76, 82, 83, 84, 87, 88, 88, 89, 
        #                 90, 93, 94, 146, 150, 200, 201, 202, 203, 204, 205, 206, 
        #                 207, 208, 209, 210, 211, 212, 213, 230, 231, 232, 234, 236,
        #                 239, 240, 241, 242, 247, 248, 249, 250, 251, 252, 253, 255, 256, 258, 260]
        # expect_frms3 = [144, 146, 149, 150, 183, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 
        #                 210, 211, 212, 213, 230, 232, 234, 236, 239, 240, 241, 242,
        #                 247, 248, 249, 250, 251, 252, 253, 255, 256, 258, 260]
        
        disp_video(video_src, expect_frms)

    
if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()




