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
    def __init__(self,csv_file,dataRoot,train=True,transform=None,select_part='trousers',gan=False,ingore_unvis=True,ksize=7,color_jitter=False):
        print('data init')
        print(select_part)
        self.csv_file=csv_file
        self.dataRoot = dataRoot
        self.ingore_unvis = ingore_unvis
        self.ksize = ksize
        self.color_jitter = color_jitter
        df = pd.read_csv(self.csv_file)
        self.numAnno = len(df)
        self.outputRes = 128
        self.scale = 4
        self.gan = gan
        self.train = train
        self.transform = transform
        self.pts_list = []
        self.image_id_list = []
        self.visiable_list = []
        for i in range(self.numAnno):
            pts = []
            visiables = []
            anno = df.ix[i]
            if (anno['image_category'] != select_part):
                continue
            for part in part_name[select_part]:
                x,y,visiable = anno[part].split('_')
                pts.append((int(x),int(y)))
                visiables.append(int(visiable))
            pts = np.array(pts)
            visiables = np.array(visiables)
            image_id = anno['image_id']
            self.pts_list.append(pts)
            self.image_id_list.append(image_id)
            self.visiable_list.append(visiables)

    def __getitem__(self,idx):
        image_id = self.dataRoot + self.image_id_list[idx]
        # print(image_id)
        img = cv2.imread(image_id)
        pts = self.pts_list[idx]
        visiables = self.visiable_list[idx]
        visiables = torch.ByteTensor(visiables)
        visiables = (visiables==1)
        H,W,C = img.shape
        x_delta = int((512-W)/2)
        img_pad = np.zeros((512,512,3),dtype=np.uint8) + 255
        img_pad[:H,x_delta:W+x_delta,:C] = img #put image in x center
        # img_pad[0:H,:W,:C] = img
        pts[:,0] = pts[:,0]+x_delta #cord transform
        img = img_pad
        if self.train:
            # if random.random() < 0.5:
            #     img,pts = flip(img,pts)
            if random.random() < 0.5:
                rotate_angle = np.random.uniform(-30,30)
                img,pts = rot(img,pts,rotate_angle,[512,512])
            if self.color_jitter:
                img = random_color_distortion(img)
        
        #有的点经过旋转后转出去了,可能影响训练
        temp_x = pts[:,0].copy();temp_y = pts[:,1].copy()
        temp_x=torch.Tensor(temp_x); temp_y=torch.Tensor(temp_y)
        x_bool = (temp_x<0) | (temp_y>512); y_bool = (temp_y<0) | (temp_y>512)
        visiables[x_bool]=0; visiables[y_bool]=0

        pts = np.around(pts/self.scale).astype(np.int32) # replace //4
        # ## debug
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # for index,pt in enumerate(pts):
        #     if (visiables[index]==0):
        #         continue
        #     cv2.circle(img,(pt[0]*self.scale,pt[1]*self.scale),5,(0,255,0),-1)
        # ## debug
        heatmap = np.zeros((len(pts),self.outputRes, self.outputRes),dtype=np.float32)
        # heatmap_show = np.zeros((64,64))
        ksize = self.ksize
        gaussian = gauss2D(ksize)
        for i in range(len(pts)):
            if (pts[i,0] == -1): #or (visiables[i] != 1 and self.ingore_unvis): #ingore unvisiable
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
        # plt.subplot(1,2,1)
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

        # img = cv2.cvtColor(img_pad,cv2.COLOR_BGR2RGB)
        # img_flip,pts_flip = flip(img,pts)
        # rotate_angle = np.random.uniform(-30,30)
        # img_rotate, pts_rotate = rot(img,pts,rotate_angle,[512,512])
        # for pt in pts:
        #     cv2.circle(img,(pt[0],pt[1]),5,(0,255,0),-1)
        # for pt in pts_flip:
        #     cv2.circle(img_flip,(pt[0],pt[1]),5,(0,255,0),-1)
        # for pt in pts_rotate:
        #     cv2.circle(img_rotate,(pt[0],pt[1]),5,(0,255,0),-1)
        # heatmap = np.zeros((128,128,len(pts)))
        # heatmap_show = np.zeros((128,128))
        # ksize = 7
        # gaussian = gauss2D(ksize)
        # for i in range(len(pts)):
        #     x = pts[i,0]//4-int(ksize/2)
        #     xp = pts[i,0]//4+int(ksize/2)+1
        #     y = pts[i,1]//4-int(ksize/2)
        #     yp = pts[i,1]//4+int(ksize/2)+1
        #     l = np.clip(x, 0, self.outputRes)
        #     r = np.clip(xp, 0, self.outputRes)
        #     u = np.clip(y, 0, self.outputRes)
        #     d = np.clip(yp, 0, self.outputRes)
        #     print(pts[i])
        #     print(x,xp,y,yp,l,r,u,d)
        #     clipped = gaussian[u-y:ksize-(yp-d), l-x:ksize-(xp-r)]
        #     print(clipped)
        #     heatmap[u:d,l:r,i] = clipped
        #     heatmap_show[u:d,l:r] = clipped
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(img)
        # # plt.subplot(1,4,2)
        # # plt.imshow(img_flip)
        # # plt.subplot(1,4,3)
        # # plt.imshow(img_rotate)
        # heatmap_show = (heatmap_show*255).astype(np.uint8)
        # plt.subplot(1,2,2)
        # plt.imshow(cv2.cvtColor(heatmap_show[:,:],cv2.COLOR_GRAY2RGB))
        # plt.show()
        # print(heatmap[:,:,0], np.max(heatmap[:,:,0]))
        # return img


    def __len__(self):
        return len(self.image_id_list)
            
def main():
    dataset = fashionDataset(annoRoot+'train.csv',dataRoot,train=True,transform=[transforms.ToTensor()],select_part='outwear')
    train_loader = DataLoader(dataset,batch_size=1,shuffle=True,num_workers=0)
    batch = iter(train_loader)
    for i in range(len(train_loader)):
        img,heatmap,x_label,y_label,visiables = next(batch)
        # print(img,heatmap,visiables)
        print(x_label)
        if i==10:
            break
if __name__ == '__main__':
    main()        
