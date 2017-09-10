#!/usr/bin/env python
# coding=utf-8

import os
import cv2
import numpy as np
# visualise car error
from matplotlib import pyplot as plt

res = '../carvana/features/VGG_16/test/fc8/'
#res = '/media/Disk/wangfuyu/Carvana/train_hq/SegmentationClass/'



list = open('test_list.txt', 'w')



for root, dirnames, filenames in os.walk(res):
    filenames.sort()
    for filename in filenames:
        img = cv2.imread(res + filename)
        #img *= 255
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, 0)
        
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        ans_x = 0
        ans_y = 0
        ans_w = 0
        ans_h = 0
        ans_area = 0

        for index, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            if (w*h < 30000):
                continue

            if (w * h > ans_area):
                ans_area = w * h
                ans_x = x
                ans_y = y
                ans_w = w
                ans_h = h

        x1 = max(ans_x - 20, 1)
        y1 = max(ans_y - 20, 1)
        x2 = min(x1 + ans_w + 40, 1918)
        y2 = min(y1 + ans_h + 40, 1280)
        list.write(filename[0:-4] + '.jpg ' + str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + '\n')


# plt.imshow(img)
# plt.show()
                               
