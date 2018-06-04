#encoding:utf-8
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch    
import torch.nn as nn
import torch.nn.functional as F
import fpn
from argsoftmax import SoftArgmax

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        # self.downsample = nn.Sequential(
        #         nn.BatchNorm2d(inplanes),
        #         nn.ReLU(),
        #         nn.Conv2d(inplanes, planes * 2,
        #                   kernel_size=1, stride=stride, bias=True),
        #     )
        # self.selayer = SELayer(planes*2) #input channels equal to residual channels
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        # out = self.gn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        # out = self.gn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        # out = self.gn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        # residual = self.selayer(residual)
        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.upsample = nn.Upsample(scale_factor=2)
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)
        self.down_feature = []

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x, outs):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1 ,outs)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = self.upsample(low3)
        out = up1 + up2
        # if(n==4):
        #     outs.append(out)
        # else:
        #     outs.append(up2)
        outs.append(out)
        return out

    def forward(self, x):
        outs = []
        self._hour_glass_forward(self.depth, x, outs)
        return outs #(1/8, 1/4, 1/2, 1)

class CPN_hg_v2_argmax(nn.Module):
    inplanes = 256

    def __init__(self, num_classes):
        super(CPN_hg_v2_argmax, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fpn = fpn.FPN50(outchannel=self.inplanes)
        # globalNet
        self.global_layers = nn.ModuleList()
        self.temp_layers = nn.ModuleList()
        for i in range(4):
            self.temp_layers.append(nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1, stride=1, padding=1))
            self.global_layers.append(nn.Conv2d(self.inplanes, num_classes, kernel_size=3, stride=1, padding=1))

        # refineNet
        self.refine_block4 = nn.Sequential(  # 256
            fpn.Bottleneck_2(self.inplanes, self.inplanes//2), 
            fpn.Bottleneck_2(self.inplanes, self.inplanes//2),
            fpn.Bottleneck_2(self.inplanes, self.inplanes//2), # in 256, out 256
        )
        
        self.refine_block3 = nn.Sequential(
            fpn.Bottleneck_2(self.inplanes, self.inplanes//2),
            fpn.Bottleneck_2(self.inplanes, self.inplanes//2), # 256
        )

        self.refine_block2 = nn.Sequential(
            fpn.Bottleneck_2(self.inplanes, self.inplanes//2), # in 256, out 256
        )
        # self.refine_conv = nn.Conv2d(512*3+256, num_classes, kernel_size=3, stride=1, padding=1)  

        # self.deconv = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final_block = nn.Sequential(
            fpn.Bottleneck_2(self.inplanes*4, self.inplanes//2), # in 256*4, out 256
        )
        self.refine_conv = nn.Conv2d(self.inplanes, num_classes, kernel_size=3, stride=1, padding=1)  
        
        ###hourglass part
        block = Bottleneck
        ch = 256
        num_blocks=1
        self.num_stacks=2
        self.inplanes = 64
        self.num_feats = 128 #reduce channels
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True) #stride=1 to make output size=256
        self.bn1 = nn.BatchNorm2d(self.inplanes) 
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        hg_feature_pyramid = []
        for i in range(2): # stack 3
            for j in range(3):
                #三个尺度特征的1x1通道压缩卷积
                hg_feature_pyramid.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            # se.append(SELayer(ch))
            print(ch,num_classes)
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            if i < self.num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        # self.se = nn.ModuleList(se)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_) 
        self.score_ = nn.ModuleList(score_)
        self.score_hg_feature_pyramid = nn.ModuleList(hg_feature_pyramid)
        self.score_mutil_reg = nn.Conv2d(num_classes*2, num_classes, kernel_size=1, bias=True)

        #fuse part
        self.fuse_conv = nn.Conv2d(num_classes*2, num_classes, kernel_size=1, bias=True)

        #soft argmax
        self.argmax = SoftArgmax(128,128,num_classes)
    
    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )
        
    def forward(self, xin):
        xs = self.fpn(xin)
        outs = []
        # outs.append(self.global_layers[0](xs[0]))
        # globalout = []
        for i, x in enumerate(xs):
            outs.append(self.global_layers[i](self.temp_layers[i](x)))
        # import pdb; pdb.set_trace()

        ys = []
        ys.append(xs[0])
        _,_,h,w = xs[0].size()
        ys.append(F.upsample(self.refine_block2(xs[1]), size=(h,w), mode='bilinear'))
        ys.append(F.upsample(self.refine_block3(xs[2]), size=(h,w), mode='bilinear'))
        ys.append(F.upsample(self.refine_block4(xs[3]), size=(h,w), mode='bilinear'))
        y = torch.cat(ys, dim=1)
        
        y = self.final_block(y)
        refineout = self.refine_conv(y)
        outs.append(refineout)

        ###hourglass part
        x = self.conv1(xin)
        x = self.bn1(x)
        # x = self.gn1(x)
        x = self.relu(x) 

        x = self.layer1(x)  
        x = self.maxpool(x)
        x = self.layer2(x)  
        x = self.layer3(x)
        for i in range(2):
            y8,y4,y2,y = self.hg[i](x) #分别表示1/8,1/4,1/2和原始大小的输出
            ys = [y8,y4,y2]
            for j in range(2,3):#只要1/2这张特征图
                outs.append(self.score_hg_feature_pyramid[i*3+j](ys[j]))
            # out.append(self.score_hg_feature_pyramid[i](y2))
            y = self.res[i](y)
            y = self.fc[i](y)
            # y = self.se[i](y)
            score = self.score[i](y)
            outs.append(score) #128
            if i == self.num_stacks-1: #mutil-scale-regression-network
                score_end = torch.cat([outs[-1], F.upsample(outs[-2],scale_factor=2)],1)
                score_end = self.score_mutil_reg(score_end)
                outs.append(score_end)
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_
        #合并
        fuse_feature = torch.cat([refineout,score_end],1)
        fuse_out = self.fuse_conv(fuse_feature)
        outs.append(fuse_out)
        px,py = self.argmax(fuse_out) #batch,num_part
        px1,py1 = self.argmax(refineout)
        px2,py2 = self.argmax(score_end)

        

        return outs, px, py, px1, py1, px2, py2

def main():
    from torch.autograd import Variable
    net = CPN_hg_v2_argmax(num_classes=5)
    # x = torch.rand(1,3,512,512)
    # x = Variable(x)
    # y = net(x)
    # print(y)
    params_dict = dict(net.named_parameters())
    for key, value in params_dict.items():
        print(key)
    # print(net.parameters())

if __name__ == '__main__':
    # import torch
    # from torch.autograd import Variable
    # x = Variable(torch.randn(1,3,512,512))
    # # x = x.cuda()
    # net = CPN(num_classes=5)
    # # net.cuda()
    # import pdb; pdb.set_trace()
    
    # gouts, rout = net(x)
    # print(rout.size())
    # for out in gouts:
    #     print(out.size())
    main()
