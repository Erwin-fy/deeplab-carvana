#!/usr/bin/env python
# coding=utf-8

import os
from skimage import io
import numpy as np
import csv
from skimage import measure, morphology


def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of 
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask, 
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)





res = '../carvana/features/VGG_16/test/fc8'

sample_submission = open('sample_submission.csv', 'w')
writer = csv.writer(sample_submission)
writer.writerow(['img', 'rle_mask'])

i = 0

for root, dirnames, filenames in os.walk(res):
    filenames.sort()
    print len(filenames)
    for filename in filenames:
        img_fn = filename[0:-4]
        img = io.imread(os.path.join(res, filename))
        
        limg = measure.label(img)
        props = measure.regionprops(limg)
        props = sorted(props, key=lambda p: -p.area)
        # mask operation
        mask = np.uint64(limg == props[0].label)
        img = np.uint64(img) & mask

        '''
        height = img.shape[0]
        width = img.shape[1]
        
        h = 0
        w = 0
        seq = ''
        while(h < height):
            while(w < width):
                if (img[h][w] > 0):
                    index = h*width + w
                    count = 1
                    w += 1
                    while((w < width) and (img[h][w] > 0)):
                        count += 1
                        w += 1
                    s = str(index) + ' ' + str(count) + ' '
                    seq += s
                else:
                    w += 1
            h += 1
            w = 1
        '''
        rle_str = rle_to_string(rle_encode(img))
        writer.writerow([img_fn + '.jpg', rle_str])
        i += 1
        if (i%100 == 0):
            print filename
