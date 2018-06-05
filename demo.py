#encoding:utf-8
# from __future__ import absolute_import
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms

import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math

from fusemodel.argsoftmax_cpn_hgv2 import CPN_hg_v2_argmax
from fusemodel.dataset import part_name
from fusemodel.image import flip_util

from crop.argsoftmax_cpn_hgv2 import CPN_hg_v2_argmax as CPN_hg_v2_argmax_crop

trousers_net = CPN_hg_v2_argmax(num_classes=len(part_name['trousers']))
checkpoint = torch.load('./checkpoint/argmax-trousers.pth')
trousers_net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
best_NE = checkpoint['NE']
start_epoch = checkpoint['epoch']
trousers_net.cuda()
trousers_net.eval()
print('load trousers model done!')
print('best_loss{} best_NE{} epoch{}'.format(best_acc,best_NE,start_epoch))

trousers_net_crop = CPN_hg_v2_argmax_crop(num_classes=len(part_name['trousers']))
checkpoint = torch.load('./crop/checkpoint/argmax-crop13-trousers.pth')
trousers_net_crop.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
best_NE = checkpoint['NE']
start_epoch = checkpoint['epoch']
trousers_net_crop.cuda()
trousers_net_crop.eval()
print('load trousers model done!')
print('best_loss{} best_NE{} epoch{}'.format(best_acc,best_NE,start_epoch))

# net = {'trousers':trousers_net,'blouse':blouse_net,'outwear':outwear_net,'skirt':skirt_net,'dress':dress_net}
net = {'trousers':trousers_net}


transform=[transforms.ToTensor()]
testDataRoot = './data/test_round2/'
trainDataRoot = './data/train/'
annoRoot = './data/train/Annotations/'
col_default = ['-1_-1_-1']*26
sample_df = pd.read_csv(annoRoot+'valid.csv')
csv_file = testDataRoot+'test.csv'
df = pd.read_csv(annoRoot+'valid.csv')
# df = pd.read_csv(annoRoot+'train_round2.csv')
# df = pd.read_csv(annoRoot+'fashionAI_key_points_test_b_answer_20180426.csv')
# df = pd.read_csv(csv_file)

for i in range(len(df)):
    anno = df.ix[i]
    image_id = anno['image_id']
    category = anno['image_category']
    if category != 'trousers':
        continue

    img_src = cv2.imread(trainDataRoot+image_id)
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(img_src)

    ############### first step predict #######################
    H,W,C = img_src.shape
    x_delta = int((512-W)/2)
    x_delta_right = 512-W-x_delta
    img_pad = np.zeros((512,512,3),dtype=np.uint8) + 255
    img_pad[0:H,x_delta:W+x_delta,:C] = img_src #put image in x center
    img = img_pad
    for t in transform:
        img = t(img)
    img = img.unsqueeze(0)
    img = Variable(img.cuda(), volatile=True)
    tempOut, px,py,px1,py1,px2,py2= net[category](img)
    C,H,W = (3,128,128)
    px = (px.cpu().data).numpy()*W; py = (py.cpu().data).numpy()*H
    pts = np.zeros((len(part_name[category]),2),dtype=np.int32)
    pts[:,0] = px; pts[:,1] = py
    pts[:,:] = pts[:,:]*(img_pad.shape[1])/H
    pts[:,0] -= x_delta

    if True: # predict flip and compute average
        img = cv2.flip(img_pad, 1)
        H,W,C = img.shape
        for t in transform:
            img = t(img)
        img = img.unsqueeze(0)
        img = Variable(img.cuda(), volatile=True)
        tempOut, px,py,px1,py1,px2,py2= net[category](img)#net[category](img)
        C,H,W = (3,128,128)
        px = (px.cpu().data).numpy()*W; py = (py.cpu().data).numpy()*H
        pts_flip = np.zeros((len(part_name[category]), 2), dtype=np.int32)
        pts_flip[:,0] = px; pts_flip[:,1] = py
        pts_flip[:,:] = pts_flip[:,:]*(img_pad.shape[1])/H # *512/128
        pts_flip[:,0] -= x_delta_right #
        pts_flip[:,0] = img_src.shape[1] - pts_flip[:,0]
        pts_flip = flip_util(pts_flip,category) #翻转点
        pts = (pts+pts_flip)/2.0
    pts_step1 = np.round(pts).astype(np.int32)
    ############### first step predict #######################


    ############## second step predict, use croped image #######################
    min_x = int(np.min(pts[:,0])); max_x = int(np.max(pts[:,0]))
    min_y = int(np.min(pts[:,1])); max_y = int(np.max(pts[:,1]))
    H,W,C = img_src.shape
    min_x = max(0,min_x-30); min_y=max(0,min_y-30); max_x=min(max_x+30,W); max_y=min(max_y+30,H)
    img = img_src[min_y:max_y,min_x:max_x,:]#截取区域
    H,W,C = img.shape; flip_w = W
    long_bian = max(H,W)
    scale = 512./long_bian

    img = cv2.resize(img,dsize=None,fx=scale,fy=scale)
    H,W,C = img.shape
    x_delta = int((512-W)/2.)
    x_delta_right = 512-W-x_delta
    img_pad = np.zeros((512,512,3),dtype=np.uint8) + 255
    img_pad[0:H,x_delta:W+x_delta,:C] = img #put image in x center
    img = img_pad
    for t in transform:
        img = t(img)
    img = img.unsqueeze(0)
    img = Variable(img.cuda(), volatile=True)
    tempOut, px, py,_,_,_,_= net[category](img)
    C,H,W = (3,128,128)
    px = (px.cpu().data).numpy()*W; py = (py.cpu().data).numpy()*H
    pts[:,0] = px; pts[:,1] = py
    pts[:,1] = pts[:,1]*(img_pad.shape[0])/float(H)
    pts[:,0] = pts[:,0]*(img_pad.shape[1])/float(W)
    pts[:,0] -= x_delta
    pts[:,1] = pts[:,1]/scale
    pts[:,0] = pts[:,0]/scale
    pts[:,0] += min_x #回到截取前的坐标
    pts[:,1] += min_y

    if True: # predict flip and compute average
        img = cv2.flip(img_pad, 1)
        H,W,C = img.shape
        for t in transform:
            img = t(img)
        img = img.unsqueeze(0)
        img = Variable(img.cuda(), volatile=True)
        tempOut, px, py,_,_,_,_= net[category](img)#net[category](img)
        finalOut = tempOut[-1]
        pts_flip = np.zeros((len(part_name[category]), 2), dtype=np.int32)
        C,H,W = (3,128,128)
        px = (px.cpu().data).numpy()*W; py = (py.cpu().data).numpy()*H
        pts_flip[:,0] = px; pts_flip[:,1] = py
        pts_flip[:,1] = pts_flip[:,1]*(img_pad.shape[0])/float(H)
        pts_flip[:,0] = pts_flip[:,0]*(img_pad.shape[1])/float(W)
        pts_flip[:,0] = img_pad.shape[1] - pts_flip[:,0]
        pts_flip[:,0] -= x_delta #
        pts_flip[:,1] = pts_flip[:,1]/scale
        pts_flip[:,0] = pts_flip[:,0]/scale
        pts_flip[:,0] += min_x
        pts_flip[:,1] += min_y
        pts_flip = flip_util(pts_flip,category) #翻转点
        pts = (pts+pts_flip)/2.0
    pts_step2 = np.round(pts).astype(np.int32)
    ############## second step predict, use croped image #######################

    ############# imshow #########
    #1. final result
    #2. first step result
    #3. second step result
    #from left to right
    img_src = cv2.cvtColor(img_src,cv2.COLOR_BGR2RGB)
    img_src1 = img_src.copy()
    img_src2 = img_src.copy()
    pts = np.round((0.35*pts_step1+0.65*pts_step2)).astype(np.int32)
    for index,pt in enumerate(pts):
        cv2.circle(img_src,(pt[0],pt[1]),7,(0,255,0),-1)
    for index,pt in enumerate(pts_step1):
        cv2.circle(img_src1,(pt[0],pt[1]),7,(0,255,0),-1)
    for index,pt in enumerate(pts_step2):
        cv2.circle(img_src2,(pt[0],pt[1]),7,(0,255,0),-1)
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img_src)
    plt.subplot(1,3,2)
    plt.imshow(img_src1)
    plt.subplot(1,3,3)
    plt.imshow(img_src2)
    plt.show()