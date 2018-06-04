#encoding:utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms

from dataset_crop2 import part_name
from image import *
from argsoftmax_cpn_hgv2 import CPN_hg_v2_argmax

import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math

from argsoftmax import SoftArgmax

# argmax = SoftArgmax(128,128,len(part_name['trousers']))
# argmax = argmax.cuda()

def distance(x1,y1,x2,y2):
    diss = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return diss
# net = hourglass(len(blouse_part))
# trousers_net_flip = HourglassNet_old(block=Bottleneck,num_blocks=4,num_classes=len(part_name['trousers']),image_c=3)
# checkpoint = torch.load('./checkpoint/ckpt512bestNE-s2n1-128-trousers.pth')
# trousers_net_flip.load_state_dict(checkpoint['net'])
# best_acc = checkpoint['acc']
# start_epoch = checkpoint['epoch']
# trousers_net_flip.cuda()
# trousers_net_flip.eval()
# print('load trousers model done!')
# print('best_loss{} epoch{}'.format(best_acc,start_epoch))

trousers_net = CPN_hg_v2_argmax(num_classes=len(part_name['trousers']))
checkpoint = torch.load('./checkpoint/argmax-crop13-trousers.pth')
trousers_net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
best_NE = checkpoint['NE']
start_epoch = checkpoint['epoch']
trousers_net.cuda()
trousers_net.eval()
print('load trousers model done!')
print('best_loss{} best_NE{} epoch{}'.format(best_acc,best_NE,start_epoch))

outwear_net = CPN_hg_v2_argmax(num_classes=len(part_name['outwear']))
checkpoint = torch.load('./checkpoint/argmax-crop13-outwear.pth')
outwear_net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
best_NE = checkpoint['NE']
start_epoch = checkpoint['epoch']
outwear_net.cuda()
outwear_net.eval()
print('load outwear model done!')
print('best_loss{} best_NE{} epoch{}'.format(best_acc,best_NE,start_epoch))

blouse_net = CPN_hg_v2_argmax(num_classes=len(part_name['blouse']))
checkpoint = torch.load('./checkpoint/argmax-crop13-blouse.pth')
blouse_net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
best_NE = checkpoint['NE']
start_epoch = checkpoint['epoch']
blouse_net.cuda()
blouse_net.eval()
print('load blouse model done!')
print('best_loss{} best_NE{} epoch{}'.format(best_acc,best_NE,start_epoch))


skirt_net = CPN_hg_v2_argmax(num_classes=len(part_name['skirt']))
checkpoint = torch.load('./checkpoint/argmax-crop13-skirt.pth')
skirt_net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
best_NE = checkpoint['NE']
start_epoch = checkpoint['epoch']
skirt_net.cuda()
skirt_net.eval()
print('load skirt model done!')
print('best_loss{} best_NE{} epoch{}'.format(best_acc,best_NE,start_epoch))

dress_net = CPN_hg_v2_argmax(num_classes=len(part_name['dress']))
checkpoint = torch.load('./checkpoint/argmax-crop13-dress.pth')
dress_net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
best_NE = checkpoint['NE']
start_epoch = checkpoint['epoch']
dress_net.cuda()
dress_net.eval()
print('load dress model done!')
print('best_loss{} best_NE{} epoch{}'.format(best_acc,best_NE,start_epoch))
net = {'trousers':trousers_net,'blouse':blouse_net,'outwear':outwear_net,'skirt':skirt_net,'dress':dress_net}
# net = {'skirt':skirt_net,'blouse':blouse_net,'outwear':outwear_net,'trousers':trousers_net}
# transform=[transforms.ToTensor(),transforms.Normalize(mean=[0.625,0.638,0.67],std=[0.252,0.254,0.245])]
transform=[transforms.ToTensor()]
testDataRoot = '../data/test_round2/'
# trainDataRoot = '/media/xiong/449C8E929C8E7DE4/fashionAI/train/'
# annoRoot = '/media/xiong/449C8E929C8E7DE4/fashionAI/train/Annotations/'
trainDataRoot = '../data/train/'
annoRoot = '../data/train/Annotations/'
col_default = ['-1_-1_-1']*26
sample_df = pd.read_csv(annoRoot+'valid.csv')
csv_file = testDataRoot+'test.csv'
df = pd.read_csv(annoRoot+'valid-argmax-cpnv2.csv')
# df = pd.read_csv('../round2_argmax.csv')
# df = pd.read_csv(annoRoot+'valid.csv')
# df = pd.read_csv(annoRoot+'fashionAI_key_points_test_b_answer_20180426.csv')
# df = pd.read_csv(csv_file)
submit_df = pd.DataFrame(columns=sample_df.columns)
col_index = 0
for i in range(len(df)):
    # if i==40:
    #     break
    # pts = np.zeros((len(part_name['trousers']),2),dtype=np.int32)
    anno = df.ix[i]
    
    image_id = anno['image_id']
    category = anno['image_category']
    # if category != 'dress':
    #     continue
    submit_df.loc[col_index]=col_default
    submit_df.loc[col_index]['image_id']=image_id
    submit_df.loc[col_index]['image_category']=category
    print(image_id)
    pts_t = []
    for part in part_name[category]:
        x,y,visiable = anno[part].split('_')
        # if int(visiable) == 1:
        #     pts_t.append((int(x),int(y)))
        # else:
        #     pts_t.append((int(x),int(y)))
        pts_t.append((int(x),int(y)))
    pts_t = np.array(pts_t).astype(np.int32)
    min_x=9999; min_y=9999; max_x=0; max_y=0
    for part in part_name[category]:
        x,y,visiable = anno[part].split('_')
        if int(x)!=-1:
            min_x = min(min_x,int(x));max_x=max(max_x,int(x))
        if int(y)!=-1:
            min_y=min(min_y,int(y));max_y=max(max_y,int(y))

    img_src = cv2.imread(trainDataRoot+image_id)
    H,W,C = img_src.shape
    min_x = max(0,min_x-30); min_y=max(0,min_y-30); max_x=min(max_x+30,W); max_y=min(max_y+30,H)
    img = img_src[min_y:max_y,min_x:max_x,:]#截取区域
    # cv2.imwrite('imgpred/'+image_id.split('/')[-1],img) ########
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
    # img = cv2.resize(img,(256,256))
    for t in transform:
        img = t(img)
    img = img.unsqueeze(0)
    img = Variable(img.cuda(), volatile=True)
    # img = Variable(img, volatile=True)
    tempOut, px, py,_,_,_,_= net[category](img)
    finalOut = tempOut[-1]
    pts = np.zeros((len(part_name[category]),2),dtype=np.int32)
    pred_heatmap = (finalOut.cpu().data.squeeze(0)).numpy()
    pred_heatmap[pred_heatmap<0]=0
    pred_heatmap[pred_heatmap>255]=255
    # px, py = argmax(finalOut) #1,num_part
    C,H,W = pred_heatmap.shape
    px = (px.cpu().data).numpy()*W; py = (py.cpu().data).numpy()*H
    
    # for j in range(pred_heatmap.shape[0]):
    #     one_pt_map = pred_heatmap[j,:,:]
    #     maxi = np.argmax(one_pt_map)
    #     pts[j,:] = np.unravel_index(maxi,(H,W))
    #         # one_pt_map[np.unravel_index(maxi,(H,W))] = 0 # mask bigest
    #         # maxi = np.argmax(one_pt_map)
    #         # pts_two[j,:] = np.unravel_index(maxi,(H,W))
    #         # pts[j,:] = np.around(pts[j,:]*0.75 + pts_two[j,:]*0.25).astype(np.int32)
    #     # print((img_pad.shape[1])/H)
    pts[:,0] = px; pts[:,1] = py
    pts[:,1] = pts[:,1]*(img_pad.shape[0])/float(H)
    pts[:,0] = pts[:,0]*(img_pad.shape[1])/float(W)
    # pts[:,[0,1]] = pts[:,[1,0]]
    pts[:,0] -= x_delta
    pts[:,1] = pts[:,1]/scale
    pts[:,0] = pts[:,0]/scale
    pts[:,0] += min_x #回到截取前的坐标
    pts[:,1] += min_y

    if True: # predict twice and compute average
        img = cv2.flip(img_pad, 1)
        H,W,C = img.shape
        # img_pad_flip = np.zeros((512,512,3),dtype=np.uint8) + 255
        # img_pad_flip[0:H,0:W,:C] = img
        # img = img_pad_flip
        for t in transform:
            img = t(img)
        img = img.unsqueeze(0)
        img = Variable(img.cuda(), volatile=True)
        tempOut, px, py,_,_,_,_= net[category](img)#net[category](img)
        finalOut = tempOut[-1]
        pts_flip = np.zeros((len(part_name[category]), 2), dtype=np.int32)
        pred_heatmap = (finalOut.cpu().data.squeeze(0)).numpy()
        pred_heatmap[pred_heatmap<0] = 0
        pred_heatmap[pred_heatmap>255] = 255
        # px, py = argmax(finalOut) #1,num_part
        C,H,W = pred_heatmap.shape
        px = (px.cpu().data).numpy()*W; py = (py.cpu().data).numpy()*H
        
        # for j in range(pred_heatmap.shape[0]):
        #     one_pt_map = pred_heatmap[j,:,:]
        #     maxi = np.argmax(one_pt_map)
        #     pts_flip[j,:] = np.unravel_index(maxi, (H,W))
        # # import pdb; pdb.set_trace()
        # pts_flip[:,[0,1]] = pts_flip[:,[1,0]] # x,y
        pts_flip[:,0] = px; pts_flip[:,1] = py
        pts_flip[:,1] = pts_flip[:,1]*(img_pad.shape[0])/float(H)
        pts_flip[:,0] = pts_flip[:,0]*(img_pad.shape[1])/float(W)
        pts_flip[:,0] = img_pad.shape[1] - pts_flip[:,0]
        pts_flip[:,0] -= x_delta #
        pts_flip[:,1] = pts_flip[:,1]/scale
        pts_flip[:,0] = pts_flip[:,0]/scale
        # pts_flip[:,0] = flip_w - pts_flip[:,0]

        pts_flip[:,0] += min_x
        pts_flip[:,1] += min_y
        
        # pts_flip[[0,1], :] = pts_flip[[1,0], :]
        # pts_flip[[3,5], :] = pts_flip[[5,3], :]
        # pts_flip[[4,6], :] = pts_flip[[6,4], :]
        pts_flip = flip_util(pts_flip,category) #翻转点
        pts = (pts+pts_flip)/2.0
    # pts[:,0] = pts[:,0]-2
    # pts[:,1] = pts[:,1]-2
    pts = np.round(pts).astype(np.int32)
    

    
    for index,pt in enumerate(pts):
        # print(pt)
        submit_df.loc[col_index][part_name[category][index]] = str(pt[0])+'_'+str(pt[1])+'_1'
        # cv2.circle(img_src,(pt[0],pt[1]),3,(0,255,0),-1)
        # cv2.circle(img_src,(pts_t[index][0],pts_t[index][1]),3,(0,0,255),-1)
        # cv2.line(img_src, (pt[0],pt[1]), (pts_t[index][0],pts_t[index][1]), color=(255,0,0), thickness=1)
    # corners = np.int0(corners)
    # # print(corners)  
    # for i in corners:  
    #     x,y = i.ravel()  
    #     cv2.circle(img_pad,(x,y),3,255,-1)
    # for pt in pts:
    #     cv2.circle(img_pad,(pt[0],pt[1]),3,(0,0,255),-1)
    # heatmap_show = np.sum(pred_heatmap,axis=0)/C
    # print(pred_heatmap[0,:,:],np.mean(pred_heatmap))
    heatmap_show = (pred_heatmap[0,:,:]*255).astype(np.uint8)

    # t = torch.from_numpy(heatmap)
    # t = t.unsqueeze(0)
    # t = Variable(t)
    # l = loss(tempOut, finalOut, t)
    # print('loss {}'.format(l.data[0]))

    # heatmap = (np.sum(heatmap,0)*25).astype(np.uint8)
    heatmap_show = (heatmap_show).astype(np.uint8)
    # print(np.mean(heatmap), np.mean(heatmap_show))
    # if random.random() < 0.5:
    #     cv2.imwrite('img/'+image_id.split('/')[-1],img_src)
    col_index += 1
    # plt.figure()
    # img_pad = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB)
    # plt.subplot(1,3,1)
    # plt.imshow(img_pad)
    # plt.subplot(1,3,2)
    # plt.imshow(cv2.cvtColor(heatmap_show, cv2.COLOR_GRAY2RGB))
    # plt.subplot(1,3,3)
    # plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB))
    # plt.show()
submit_df.to_csv('../argmax-cpnv2-crop.csv',index=False)
