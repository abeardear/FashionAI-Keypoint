#encoding:utf-8
import os
import numpy as np
import torch
import pandas as pd
import cv2
import torchvision.transforms as transforms
from torch.autograd import Variable
from dataset_crop2 import part_name
from image import *
import math
from argsoftmax import SoftArgmax

def get_preds(scores):
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:,:,0] = (preds[:,:,0] - 1) % scores.size(3) + 1
    preds[:,:,1] = torch.floor((preds[:,:,1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds

def calc_dists(preds, target, normalize):
    preds = preds.float()
    target = target.float()
    dists = torch.zeros(preds.size(1), preds.size(0))
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
            if target[n,c,0] > 1 and target[n, c, 1] > 1:
                dists[c, n] = torch.dist(preds[n,c,:], target[n,c,:])/normalize[n]
            else:
                dists[c, n] = -1
    return dists

def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    if dists.ne(-1).sum() > 0:
        return dists.le(thr).eq(dists.ne(-1)).sum()*1.0 / dists.ne(-1).sum()
    else:
        return -1

def accuracy(output, target, idxs, thr=0.5):
    ''' Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs', followed by individual accuracies
    '''
    preds   = get_preds(output)
    gts     = get_preds(target)
    norm    = torch.ones(preds.size(0))*output.size(3)/10
    dists   = calc_dists(preds, gts, norm)

    acc = torch.zeros(idxs+1)
    avg_acc = 0
    cnt = 0

    for i in range(idxs):
        acc[i+1] = dist_acc(dists[i])
        if acc[i+1] >= 0: 
            avg_acc = avg_acc + acc[i+1]
            cnt += 1
            
    if cnt != 0:  
        acc[0] = avg_acc / cnt
    return acc[0]

def ne_evaluate(score_map, target_map, part, select_part):
    '''
    approximate normalized error
    '''
    preds = get_preds(score_map)*4
    gts = get_preds(target_map)*4
    preds = preds.numpy()
    gts = gts.numpy()
    batch_size = gts.shape[0]
    # print(gts)
    if select_part == 'trousers' or select_part == 'skirt':
        a = part.index('waistband_left')
        b = part.index('waistband_right')
    if select_part == 'blouse' or select_part == 'outwear' or select_part == 'dress':
        a = part.index('armpit_left')
        b = part.index('armpit_right')
    left = gts[:,a]
    right = gts[:,b] #batch,2
    diss = np.sqrt(np.sum((left-right)**2,1))#b,
    diss[diss==0] = 100.0
    visiable = (gts!=0) #b,13,2
    # gts = gts[visiable].reshape(-1,2)
    # preds = preds[visiable].reshape(-1,2) #b*13,2
    preds[~visiable] = 0
    visiable = visiable.reshape(batch_size,-1) #b,26
    error = np.sqrt(np.sum((preds-gts)**2,2)) #b,13
    visiable_sum = (np.sum(visiable,1)/2) #b,
    visiable_sum[visiable_sum==0] = 1000
    error = np.sum(error,1) / visiable_sum #b,
    error = error/diss
    NE = np.sum(error)/batch_size
    return NE

def pred_csv(net,select_part,name=''):    
    transform=[transforms.ToTensor()]
    testDataRoot = '../data/testb/'
    # trainDataRoot = '/media/xiong/449C8E929C8E7DE4/fashionAI/train/'
    # annoRoot = '/media/xiong/449C8E929C8E7DE4/fashionAI/train/Annotations/'
    trainDataRoot = '../data/train/'
    annoRoot = '../data/train/Annotations/'
    col_default = ['-1_-1_-1']*26
    sample_df = pd.read_csv(annoRoot+'valid-argmax-cpnv2.csv')
    csv_file = testDataRoot+'test.csv'
    df = pd.read_csv(annoRoot+'valid-argmax-cpnv2.csv')
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
        if category != select_part:
            continue
        submit_df.loc[col_index]=col_default
        submit_df.loc[col_index]['image_id']=image_id
        submit_df.loc[col_index]['image_category']=category
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
        H,W,C = img.shape; flip_w = W
        long_bian = max(H,W)
        scale = 512./long_bian
        # x_delta = int((512-W)/2)
        # x_delta_right = 512-W-x_delta
        # img_pad = np.zeros((512,512,3),dtype=np.uint8) + 255
        # img_pad[0:H,x_delta:W+x_delta,:C] = img_src #put image in x center
        # img_pad[0:H,:W,:C] = img_src
        # img = img_pad
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
        # img = Variable(img, volatile=True)
        tempOut, px, py,_,_,_,_= net(img)
        finalOut = tempOut[-1]
        pts = np.zeros((len(part_name[category]),2))
        pred_heatmap = (finalOut.cpu().data.squeeze(0)).numpy()
        pred_heatmap[pred_heatmap<0]=0
        pred_heatmap[pred_heatmap>255]=255
        C,H,W = pred_heatmap.shape
        # px, py = argmax(finalOut) #1,num_part
        px = (px.cpu().data).numpy()*W; py = (py.cpu().data).numpy()*H
        # for j in range(pred_heatmap.shape[0]):
        #     one_pt_map = pred_heatmap[j,:,:]
        #     maxi = np.argmax(one_pt_map)
        #     pts[j,:] = np.unravel_index(maxi,(H,W))
                # one_pt_map[np.unravel_index(maxi,(H,W))] = 0 # mask bigest
                # maxi = np.argmax(one_pt_map)
                # pts_two[j,:] = np.unravel_index(maxi,(H,W))
                # pts[j,:] = np.around(pts[j,:]*0.75 + pts_two[j,:]*0.25).astype(np.int32)
            # print((img_pad.shape[1])/H)
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
            # img_pad = np.zeros((512,512,3),dtype=np.uint8) + 255
            # img_pad[0:H,0:W,:C] = img
            # img = img_pad
            # img = cv2.resize(img_src,(384,384))
            for t in transform:
                img = t(img)
            img = img.unsqueeze(0)
            img = Variable(img.cuda(), volatile=True)
            tempOut, px, py ,_,_,_,_= net(img)#net[category](img)
            finalOut = tempOut[-1]
            pts_flip = np.zeros((len(part_name[category]), 2))
            pred_heatmap = (finalOut.cpu().data.squeeze(0)).numpy()
            pred_heatmap[pred_heatmap<0] = 0
            pred_heatmap[pred_heatmap>255] = 255
            C,H,W = pred_heatmap.shape
            # px, py = argmax(finalOut) #1,num_part
            px = (px.cpu().data).numpy()*W; py = (py.cpu().data).numpy()*H
            # for j in range(pred_heatmap.shape[0]):
            #     one_pt_map = pred_heatmap[j,:,:]
            #     maxi = np.argmax(one_pt_map)
            #     pts_flip[j,:] = np.unravel_index(maxi, (H,W))
            # import pdb; pdb.set_trace()
            # pts_flip[:,[0,1]] = pts_flip[:,[1,0]] # x,y
            pts_flip[:,0] = px; pts_flip[:,1] = py
            pts_flip[:,1] = pts_flip[:,1]*(img_pad.shape[0])/float(H)
            pts_flip[:,0] = pts_flip[:,0]*(img_pad.shape[1])/float(W)
            pts_flip[:,0] -= x_delta_right #
            pts_flip[:,1] = pts_flip[:,1]/scale
            pts_flip[:,0] = pts_flip[:,0]/scale
            pts_flip[:,0] = flip_w - pts_flip[:,0]

            pts_flip[:,0] += min_x
            pts_flip[:,1] += min_y
            # pts_flip[[0,1], :] = pts_flip[[1,0], :]
            # pts_flip[[3,5], :] = pts_flip[[5,3], :]
            # pts_flip[[4,6], :] = pts_flip[[6,4], :]
            pts_flip = flip_util(pts_flip,category) #翻转点
            pts = (pts+pts_flip)/2.0
        # pts[:,0] = pts[:,0]
        # pts[:,1] = pts[:,1]-2
        pts = np.round(pts).astype(np.int32)
        

        
        for index,pt in enumerate(pts):
            submit_df.loc[col_index][part_name[category][index]] = str(pt[0])+'_'+str(pt[1])+'_1'
        col_index += 1
    submit_df.to_csv('pred'+select_part+name+'.csv',index=False)

def distance(x1,y1,x2,y2):
    l1 = math.pow((x1-x2),2)
    l2 = math.pow((y1-y2),2)
    return math.sqrt(l1+l2)

def normalized_error(select_part,name=''):
    target = pd.read_csv('../valid.csv')
    pred = pd.read_csv('pred'+select_part+name+'.csv')
    numAnno = len(target)
    NEsum = 0
    count_vis = 0
    count = 0
    diffMean = np.zeros(2)
    j = 0
    for i in range(numAnno):
        anno = target.ix[i]
        class_name = anno['image_category']
        if class_name != select_part: #如果只测试一种类别,把这里取消注释
            continue
        result = pred.ix[j]
        # print(anno['image_id'])
        assert anno['image_id'] == result['image_id'] #出错一般是编号没有对其,检查i j
        # print(anno['image_id'])
        if class_name == 'trousers' or class_name == 'skirt':
            # print('waistband')
            x1,y1,_ = anno['waistband_left'].split('_')
            x2,y2,_ = anno['waistband_right'].split('_')
        if class_name == 'blouse' or class_name == 'outwear' or class_name == 'dress':
            # print('armpit')
            x1,y1,_ = anno['armpit_left'].split('_')
            x2,y2,_ = anno['armpit_right'].split('_')

        x1,y1,x2,y2 = float(x1), float(y1), float(x2), float(y2)
        normalize = distance(x1,y1,x2,y2)
        if(normalize==0):
            # print('WTF')
            normalize = 100.0
        pts = []
        pts_pred = []
        visiable_count = 0
        for part in part_name[class_name]:
            xt,yt,visiable = anno[part].split('_')
            visiable = int(visiable)
            visiable_count += visiable
            xp,yp,_ = result[part].split('_')
            if visiable == 1:
                pts.append((float(xt),float(yt)))
                # pts_pred.append((float(xp)-0,float(yp)-2))
                pts_pred.append((float(xp)-0,float(yp)-0)) #弥补统计偏差x_offset后的效果?
        if visiable_count==0:
            j += 1
            continue
        pts = np.array(pts)
        pts_pred = np.array(pts_pred)
        diff_mean = pts-pts_pred
        diffMean += np.mean(diff_mean,0)#统计偏差x_offset,y_offset用的
        diff = (pts-pts_pred) ** 2
        # print(diff)
        diss = np.sqrt( np.sum(diff,1) )
        total_diss = np.sum(diss)
        NE = total_diss/normalize#/len(pts)
        NEsum += NE
        count_vis += len(pts)
        count += 1
        j += 1
    NEsum = NEsum/count_vis
    diffMean = diffMean/count
    return NEsum