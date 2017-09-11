#!/usr/bin/env python
# coding=utf-8

import os
from skimage import io
import numpy as np
from skimage import measure, morphology

gt_dir = '/media/Disk/wangfuyu/Carvana/train_hq/SegmentationClass'
pred_dir = '../carvana/features/VGG_16/val/fc8'

i = 0
dice = 0

for root, dirnames, filenames in os.walk(pred_dir):
    filenames.sort()
    l = len(filenames)
    print l
    for filename in filenames:
        img_fn = filename[0:-4]
        gt = io.imread(os.path.join(gt_dir, filename))
        pred = io.imread(os.path.join(pred_dir, filename))
        pred /= 255
        '''
        limg = measure.label(pred)
        props = measure.regionprops(limg)
        props = sorted(props, key=lambda p: -p.area)
        # mask operation
        mask = np.uint64(limg == props[0].label)
        pred = np.uint64(pred) & mask
        '''

        sum = pred.sum() + gt.sum()
        dice += 2.0*(pred * gt).sum() / sum

        i += 1
        if (i%100 == 0):
            print filename

    dice /= l
    print dice
