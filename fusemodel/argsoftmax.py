#encoding:utf-8
#
#created by xiongzihua
#
import torch    
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SoftArgmax(nn.Module):
    def __init__(self, H, W, num_part):
        super(SoftArgmax,self).__init__()
        self.conv_x = nn.Conv2d(in_channels=num_part,out_channels=num_part,kernel_size=(H,W), \
        groups=num_part,bias=False)
        self.conv_y = nn.Conv2d(in_channels=num_part,out_channels=num_part,kernel_size=(H,W), \
        groups=num_part,bias=False)
        self.num_part = num_part
        #权重初始化
        x_weight = np.arange(0,W)/float(W)
        x_weight = x_weight.reshape(-1,W)
        x_weight = np.repeat(x_weight,H,0)
        x_weight = np.expand_dims(x_weight,0); x_weight = np.expand_dims(x_weight,0)
        x_weight = np.repeat(x_weight,num_part,0)
        x_weight = torch.FloatTensor(x_weight)
        self.conv_x.weight.data = x_weight

        y_weight = np.arange(0,W)/float(W)
        y_weight = y_weight.reshape(H,1)
        y_weight = np.repeat(y_weight,W,1)
        y_weight = np.expand_dims(y_weight,0); y_weight = np.expand_dims(y_weight,0)
        y_weight = np.repeat(y_weight,num_part,0)
        y_weight = torch.FloatTensor(y_weight)
        self.conv_y.weight.data = y_weight
        # print(self.conv_y.weight.data)

    def forward(self,f):
        N,C,H,W = f.size()
        f = f.view(N,C,-1)
        f = F.softmax(f,-1).view(N,C,H,W)
        x = self.conv_x(f)
        y = self.conv_y(f)
        x = x.view(-1,self.num_part)
        y = y.view(-1,self.num_part)
        # print(x.size())
        return x,y

def main():
    from torch.autograd import Variable
    net = SoftArgmax(64,64,16)
    x = torch.rand(1,16,64,64)
    x = Variable(x)
    xp,yp = net(x)
    # print(y)

if __name__ == '__main__':
    main()
        