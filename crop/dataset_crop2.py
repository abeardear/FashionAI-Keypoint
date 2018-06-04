#encoding:utf-8
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import random

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from image import *

dataRoot = '../data/train/'
annoRoot = '../data/train/Annotations/'
blouse_part = ['neckline_left','neckline_right','center_front','shoulder_left','shoulder_right',
    'armpit_left','armpit_right','cuff_left_in','cuff_left_out','cuff_right_in',
    'cuff_right_out','top_hem_left','top_hem_right']
trousers_part = ['waistband_left','waistband_right','crotch','bottom_left_in','bottom_left_out','bottom_right_in','bottom_right_out']

outwear_part = ['neckline_left','neckline_right','shoulder_left','shoulder_right','armpit_left','armpit_right',
    'waistline_left','waistline_right','cuff_left_in','cuff_left_out','cuff_right_in','cuff_right_out','top_hem_left','top_hem_right']

skirt_part = ['waistband_left','waistband_right','hemline_left','hemline_right']

dress_part = ['neckline_left','neckline_right','shoulder_left','shoulder_right','center_front','armpit_left','armpit_right',
    'waistline_left','waistline_right','cuff_left_in','cuff_left_out','cuff_right_in','cuff_right_out','hemline_left','hemline_right']

part_name = {'trousers':trousers_part,'blouse':blouse_part,'outwear':outwear_part,'skirt':skirt_part,'dress':dress_part}

class fashionDataset(data.Dataset):
    def __init__(self,csv_file,pred_csv,dataRoot,train=True,transform=None,select_part='trousers',gan=False,ingore_unvis=True,ksize=7,color_jitter=False):
        print('data init')
        print(select_part)
        self.csv_file=csv_file
        self.pred_csv=pred_csv
        self.dataRoot = dataRoot
        self.ingore_unvis = ingore_unvis
        self.ksize = ksize
        self.color_jitter = color_jitter
        df = pd.read_csv(self.csv_file)
        df_pred = pd.read_csv(self.pred_csv)
        self.numAnno = len(df)
        self.numAnno_pred = len(df_pred)
        self.outputRes = 128
        self.scale = 4
        self.gan = gan
        self.train = train
        self.transform = transform
        self.pts_list = []
        self.image_id_list = []
        self.visiable_list = []
        self.min_xs = []; self.min_ys=[]; self.max_xs=[]; self.max_ys=[]
        #预测点
        self.pts_list_pred=[]
        for i in range(self.numAnno_pred):
            pts=[]
            anno = df_pred.ix[i]
            if (anno['image_category'] != select_part):
                continue
            for part in part_name[select_part]:
                x,y,visiable = anno[part].split('_')
                pts.append((int(x),int(y)))
            pts=np.array(pts)
            self.pts_list_pred.append(pts)
        for i in range(self.numAnno):
            pts = []
            visiables = []
            anno = df.ix[i]
            min_x=9999; min_y=9999; max_x=0; max_y=0
            if (anno['image_category'] != select_part):
                continue
            for part in part_name[select_part]:
                x,y,visiable = anno[part].split('_')
                pts.append((int(x),int(y)))
                visiables.append(int(visiable))
                if int(x)!=-1:
                    min_x = min(min_x,int(x));max_x=max(max_x,int(x))
                if int(y)!=-1:
                    min_y=min(min_y,int(y));max_y=max(max_y,int(y))
            pts = np.array(pts)
            visiables = np.array(visiables)
            image_id = anno['image_id']
            self.pts_list.append(pts)
            self.image_id_list.append(image_id)
            self.visiable_list.append(visiables)
            self.min_xs.append(min_x); self.min_ys.append(min_y); self.max_xs.append(max_x); self.max_ys.append(max_y)

    def __getitem__(self,idx):
        image_id = self.dataRoot + self.image_id_list[idx]
        # print(image_id)
        img = cv2.imread(image_id)
        # plt.figure()
        # plt.subplot(1,3,1)
        # plt.imshow(img)
        pts = self.pts_list[idx]
        pts_pred = self.pts_list_pred[idx]
        visiables = self.visiable_list[idx]
        # min_x=self.min_xs[idx];min_y=self.min_ys[idx];max_x=self.max_xs[idx];max_y=self.max_ys[idx]
        visiables = torch.ByteTensor(visiables)
        visiables_for_crop = (visiables!=-1)
        visiables = (visiables==1)
        
        H,W,C = img.shape
        # print(H,W)
        # min_x = max(0,min_x-40); min_y=max(0,min_y-40); max_x=min(max_x+40,W); max_y=min(max_y+40,H)
        # img = img[min_y:max_y,min_x:max_x,:]
        # pts[:,0] -= min_x; pts[:,1] -= min_y
        # H,W,C = img.shape
        # print(H,W)
        # long_bian = max(H,W)
        # scale = 384./long_bian
        x_delta = int((512-W)/2)
        img_pad = np.zeros((512,512,3),dtype=np.uint8) + 255
        img_pad[0:H,x_delta:W+x_delta,:C] = img #put image in x center
        img = img_pad
        # # img_pad[0:H,:W,:C] = img
        pts[:,0] = pts[:,0]+x_delta #cord transform
        pts_pred[:,0] = pts_pred[:,0]+x_delta
        if self.train:
            # if random.random() < 0.5:
            #     img,pts = flip(img,pts)
            if random.random() < 0.5:
                rotate_angle = np.random.uniform(-30,30)
                img,pts,pts_pred = rot(img,pts,pts_pred,rotate_angle,[512,512])
            if self.color_jitter:
                img = random_color_distortion(img)
        # plt.figure()
        # for index,pt in enumerate(pts):
        #     if (visiables[index]==0):
        #         continue
        #     cv2.circle(img,(int(pt[0]),int(pt[1])),5,(0,255,0),-1)
        # for index,pt in enumerate(pts_pred):
        #     if (visiables[index]==0):
        #         continue
        #     cv2.circle(img,(int(pt[0]),int(pt[1])),5,(0,0,255),-1)
        # plt.subplot(1,3,2)
        # plt.imshow(img)
        # plt.show()
        #在这里截取好点
        H,W,C = img.shape
        xs = pts_pred[:,0].copy(); ys = pts_pred[:,1].copy()
        xs = xs[visiables_for_crop.numpy().astype(np.bool)]; ys=ys[visiables_for_crop.numpy().astype(np.bool)]
        min_x = min(xs); max_x = max(xs); min_y = min(ys); max_y = max(ys)
        min_x = max(0,min_x-30); min_y=max(0,min_y-30); max_x=min(max_x+30,W); max_y=min(max_y+30,H)
        min_x = int(min_x); min_y=int(min_y); max_x = int(max_x); max_y=int(max_y)

        img = img[min_y:max_y,min_x:max_x,:]
        pts[:,0] -= min_x; pts[:,1] -= min_y
        H,W,C = img.shape
        long_bian = max(H,W)
        scale = 512./long_bian
        img = cv2.resize(img,dsize=None,fx=scale,fy=scale)
        H,W,C = img.shape
        x_delta = int((512-W)/2.)
        img_pad = np.zeros((512,512,3),dtype=np.uint8) + 255
        img_pad[0:H,x_delta:W+x_delta,:C] = img #put image in x center
        img = img_pad
        pts[:,0] = pts[:,0]*scale#.astype(np.int32)
        pts[:,1] = pts[:,1]*scale#.astype(np.int32)
        pts[:,0] = pts[:,0]+x_delta #cord transform
        #有的点经过旋转后转出去了,可能影响训练
        temp_x = pts[:,0].copy();temp_y = pts[:,1].copy()
        temp_x=torch.Tensor(temp_x); temp_y=torch.Tensor(temp_y)
        x_bool = (temp_x<0) | (temp_y>512); y_bool = (temp_y<0) | (temp_y>512)
        visiables[x_bool]=0; visiables[y_bool]=0


        pts = np.around(pts/float(self.scale)).astype(np.int32) # replace //4
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # for index,pt in enumerate(pts):
        #     if (visiables[index]==0):
        #         continue
        #     cv2.circle(img,(pt[0]*self.scale,pt[1]*self.scale),5,(0,255,0),-1)
        heatmap = np.zeros((len(pts),self.outputRes, self.outputRes),dtype=np.float32)
        # heatmap_show = np.zeros((64,64))
        ksize = self.ksize
        gaussian = gauss2D(ksize)
        for i in range(len(pts)):
            if (visiables[i]==0): #or (visiables[i] != 1 and self.ingore_unvis): #ingore unvisiable
                continue
            x = pts[i,0]-int(ksize/2) +1
            xp = pts[i,0]+int(ksize/2)+2
            y = pts[i,1]-int(ksize/2)+1
            yp = pts[i,1]+int(ksize/2)+2
            l = np.clip(x, 0, self.outputRes)
            r = np.clip(xp, 0, self.outputRes)
            u = np.clip(y, 0, self.outputRes)
            d = np.clip(yp, 0, self.outputRes)
            # print(pts[i])
            # print(x,xp,y,yp,l,r,u,d)
            clipped = gaussian[u-y:ksize-(yp-d), l-x:ksize-(xp-r)]
            # print(clipped)
            heatmap[i,u:d,l:r] = clipped
            # heatmap_show[u:d,l:r] = clipped
        # plt.figure()
        # plt.subplot(1,3,3)
        # plt.imshow(img)
        # heatmap_show = (heatmap[0]).astype(np.uint8)
        # plt.subplot(1,2,2)
        # plt.imshow(cv2.cvtColor(heatmap_show[:,:],cv2.COLOR_GRAY2RGB))
        # plt.show()
        for t in self.transform:
            img = t(img) #convert numpy to tensor and /255
            # gan_img = t(gan_img)
            # salmap = torch.from_numpy(salmap)
        # img = torch.cat((img,salmap),0)
        heatmap = torch.from_numpy(heatmap)
        # visiables = visiables.unsqueeze(1).unsqueeze(2).expand_as(heatmap)
        # if self.gan:
        #     return img,gan_img,heatmap
        # else:
        x_label = (pts[:,0]/float(self.outputRes)).astype(np.float32)
        y_label = (pts[:,1]/float(self.outputRes)).astype(np.float32)
        return img, heatmap, x_label, y_label, visiables



    def __len__(self):
        return len(self.image_id_list)
            
def main():
    dataset = fashionDataset(annoRoot+'train_round2.csv',annoRoot+'train-argmax-cpnv2.csv',dataRoot,select_part='outwear',train=True,transform=[transforms.ToTensor()])
    train_loader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)
    batch = iter(train_loader)
    for i in range(len(train_loader)):
        img,heatmap,x_label,y_label,visiables = next(batch)
        # print(img,heatmap,visiables)
        print(x_label)
        if i==30:
            break
if __name__ == '__main__':
    main()        