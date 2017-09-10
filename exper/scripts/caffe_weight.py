#!/usr/bin/env python
# coding=utf-8

import numpy as np
import caffe
'''
import _init_paths
import collections
from collections import OrderedDict
'''

caffe.set_mode_cpu
net0 = caffe.Net('../carvana/config/VGG_16/train_weight.prototxt',\
                 '../carvana/model/VGG_16/init.caffemodel',caffe.TEST) #TEST/TRAIN

# net1 = caffe.Net('../carvana/config/VGG_16/res075_weight.prototxt', \
#                 '../carvana/model/VGG_16/init_res075.caffemodel', caffe.TEST)

# net2 = caffe.Net('../carvana/config/VGG_16/res05_weight.prototxt', \
#                 '../carvana/model/VGG_16/init_res05.caffemodel', caffe.TEST)

net1 = caffe.Net('../carvana/config/VGG_16/multi_weight.prototxt', \
                '../carvana/model/VGG_16/multi_train_705/multi_train_iter_16000.caffemodel', caffe.TEST)

#模型参数都存在了net.params这个有序字典里，对这就是python里的那个字典，所以对模型参数的操作和对python字典操作一样。['conv1_1/conv']是键名，[0]是权的维度

keys0 = net0.params.keys()
print net0.params.keys(), len(keys0)
 

for key0 in keys0:   # 输出所有层名，参数
    
    print key0
    print 'before'
    #print net1.params[key0 + '_res05'][0].data[:] 
    net1.params[key0][0].data[:] = np.copy(net0.params[key0][0].data[:])
    net1.params[key0][1].data[:] = np.copy(net0.params[key0][1].data[:])
    net1.params[key0 + '_res075'][0].data[:] = np.copy(net0.params[key0][0].data[:])
    net1.params[key0 + '_res075'][1].data[:] = np.copy(net0.params[key0][1].data[:])
    net1.params[key0 + '_res05'][0].data[:] = np.copy(net0.params[key0][0].data[:])
    net1.params[key0 + '_res05'][1].data[:] = np.copy(net0.params[key0][1].data[:])
    print 'after'
    #print net1.params[key0 + '_res05'][0].data[:] 

    print 'ini'
    #print net0.params[key0][0].data[:]

keys1 = net1.params.keys()
for key1 in keys1:
    print key1
    print net1.params[key1][0].data[:]
print net1.params.keys()
net1.save('init_multi.caffemodel')
