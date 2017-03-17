#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:35:34 2016

@author: wangs
"""

nRounds = 1  #测试次数
tr_num = 60  #训练样本数
patchn = 1360  #图片总数


#将图片路径写入txt文件
import imgfile_to_txt 
img_dir = 'images/oxfordflower17'  #相对路径
txtfile = 'patchlist.txt'
imgfile_to_txt.imageFileToTxt(img_dir, txtfile)


import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import sys 
import os
import random
import caffe


# Make sure that caffe is on the python path:
caffe_root = '/home/wangs/caffe/'  
code_root = '/home/wangs/Work/Flower_CNN_Train(python)/'
img_root = '/home/wangs/Work/Flower_CNN_Train(python)/images/'
feat_root = '/home/wangs/Work/Flower_CNN_Train(python)/features/'


#读文件内容，存入链表imglist
flist = open(txtfile, 'r')
imglist = [i for i in range(patchn)]
for i in range(patchn):
    imglist[i] = flist.readline().rstrip('\n')
flist.close()

'''
sys.path.insert(0, caffe_root + 'python')

#设置默认显示参数
plt.rcParams['figure.figsize'] = (10, 10)  #图像显示大小
plt.rcParams['image.interpolation'] = 'nearest'   #最近邻差值: 像素为正方形
plt.rcParams['image.cmap'] = 'gray'  #使用灰度输出而不是彩色输出

caffe.set_mode_cpu()

net = caffe.Net(caffe_root + 'models/bvlc_alexnet/deploy.prototxt',
                caffe_root + 'models/bvlc_alexnet/bvlc_alexnet.caffemodel',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  # 对输入数据进行变换
transformer.set_transpose('data', (2,0,1))  #将图像的通道数设置为outermost的维数
#transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# set net to batch size of 10
net.blobs['data'].reshape(10,3,227,227)
#imgscale = ['32', '64', '128']

#提取特征
print('Extracting features...')
stri = imglist[0]
imgname = stri[0: stri.find('/')]  # rose
feat_img_path =  feat_root+imgname  # /home/wangs/Work/AlexNet/alexnet_features/rose/
if not os.path.exists(feat_img_path):                       
    os.mkdir(feat_img_path)  #创建存放特征目录
    
for i in range(patchn):
    stri = imglist[i]
    start = stri.find('/')+1
    end = len(stri)
    #查找第一根斜线和第二根斜线之间的字符串
    per_class_name = stri[start: stri.find('/', start, end)] 
    subpath = stri[0: stri.find('/', start, end)]
    feat_sub_path = feat_root + subpath
    
    if not os.path.exists(feat_sub_path):
        os.mkdir(feat_sub_path)
    
    patch_path = img_root+imglist[i]
    feat_path = feat_root+imglist[i]
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(patch_path))
    t1 = time.time()
    out = net.forward()  # call once for allocation
    t2 = time.time()
    print "Extract features from image %d cost %f sec:%s" % (i, t2 - t1, imglist[i])
    fea = net.blobs['fc7'].data[0]
    sio.savemat(feat_path + '.mat', {'fea':fea, 'label':int(per_class_name)})
'''

#训练测试数据
tr_file_path = code_root + 'train.txt'
ts_file_path = code_root + 'test.txt'
accuracy = [] #准确度

print('Testing...')
for n in range(nRounds):
    print('Rounds: %d...'%(n+1))
    
    #统计每类样本数目
    per_class_num = [] #每类样本数
    stri = imglist[0]
    start = stri.find('/')+1
    end = len(stri)
    curr_class = stri[start: stri.find('/', start, end)] 
    cnt = 1
    for i in range(1, patchn): 
        stri = imglist[i]
        start = stri.find('/')+1
        end = len(stri)
        per_class_name = stri[start: stri.find('/', start, end)] 
        if curr_class == per_class_name:
            cnt += 1
        else:
            per_class_num.append(cnt)
            curr_class = per_class_name
            cnt = 1
    per_class_num.append(cnt)
    categories = len(per_class_num)
    
    #生成liblinear数据格式的数据
    with open(tr_file_path, 'w') as trfile, open(ts_file_path, 'w') as tsfile:      
        per_class_start = 0
        for i in range(categories):          
            rand = random.sample(range(per_class_start, 
                                     per_class_start+per_class_num[i]), 
                                     per_class_num[i])
            per_class_start += per_class_num[i]
            #生成liblinear数据格式的训练数据
            for j in rand[0: tr_num]:
                feat_path = feat_root + imglist[j] + '.mat'
                data = sio.loadmat(feat_path)  
                fea = data['fea']
                label = data['label']
                tr_svm_format = "%s "%(label[0][0])  
                for k in range(4096):
                    if fea[0][k] != 0:    
                        tr_svm_format += "%d:%s "%((k+1), fea[0][k])
                tr_svm_format += "\n"
                trfile.write(tr_svm_format)
                
            #生成liblinear数据格式的测试数据
            for j in rand[tr_num: ]:
                feat_path = feat_root + imglist[j] + '.mat'
                data = sio.loadmat(feat_path)  
                fea = data['fea']
                label = data['label']
                ts_svm_format = "%s "%(label[0][0])
                for k in range(4096):
                    if fea[0][k] != 0:            
                        ts_svm_format += "%d:%s "%((k+1), fea[0][k])
                ts_svm_format += "\n"
                tsfile.write(ts_svm_format)  
                    
    #用liblinear分类器分类
    sys.path.insert(0, code_root+'liblinear/python')
    from liblinearutil import *
    y, x = svm_read_problem(tr_file_path)#读入训练数据
    prob = problem(y, x)
    param = parameter('-c 10')
    m = train(prob, param)
    save_model('heart_scale.model', m) 
    #m = load_model('heart_scale.model') 
    yt, xt = svm_read_problem(ts_file_path)#训练测试数据
    p_labels, p_acc, p_vals = predict(yt, xt, m)
    accuracy.append(p_acc[0])
print('Average classification accuracy: %0.2f%%'%(np.mean(accuracy)))


'''
#用libsvm分类器分类
sys.path.insert(0, code_root+'libsvm/python')
from svmutil import *
y, x = svm_read_problem(code_root + 'train.txt')#读入训练数据
yt, xt = svm_read_problem(code_root + 'test.txt')#训练测试数据
m = svm_train(y, x )#训练
p_label, p_acc, p_val = svm_predict(yt, xt, m)#测试
'''