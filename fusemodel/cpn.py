# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch    
import torch.nn as nn
import torch.nn.functional as F
import fpn

class CPN(nn.Module):
    inplanes = 256

    def __init__(self, num_classes,upconv=False):
        super(CPN, self).__init__()
        self.fpn = fpn.FPN50(outchannel=self.inplanes,upconv=upconv)
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
        # self.refine_conv1 = nn.Conv2d(512*3+256, 512, kernel_size=1, stride=1)#, padding=1)  
        # self.refine_bn1 = nn.BatchNorm2d(512)
        # self.refine_conv2 = nn.Conv2d(512, num_classes*4, kernel_size=(5, 3), stride=1, padding=(2,1))  
        
        # self.pixelshuffle = nn.PixelShuffle(2)
        # self.convout = nn.Conv2d(num_classes, num_classes, kernel_size=3, stride=1, padding=1)
        
    def forward(self, xs):
        xs = self.fpn(xs)
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
        # refineout = F.relu(self.refine_bn1(self.refine_conv1(y)), inplace=True)
        # refineout = self.refine_conv2(refineout)
        
        # import pdb; pdb.set_trace()
        # refineout = self.deconv(refineout)
        # refineout = self.pixelshuffle(refineout)
        # refineout = self.convout(refineout)

        outs.append(refineout)

        return outs

def main():
    from torch.autograd import Variable
    net = CPN(num_classes=5)
    # x = torch.rand(1,3,512,512)
    # x = Variable(x)
    # y = net(x)
    # print(y)
    params_dict = dict(net.named_parameters())
    for key, value in params_dict.items():
        print(key)
    print(net.parameters())

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