import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, outplanes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != outplanes*self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes*self.expansion),
            )

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.downsample is not None:
            residual = self.downsample(x)
        
        x += residual
        x = F.relu(x)
        
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, outplanes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.conv3 = nn.Conv2d(outplanes, outplanes*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes*self.expansion)

        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != outplanes*self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes*self.expansion),
            )

    def forward(self, x):
        # import pdb; pdb.set_trace()
        xf = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.bn3(self.conv3(x))
        x += self.downsample(xf)
        x = F.relu(x, inplace=True)
        return x

class Bottleneck_2(nn.Module):
    expansion = 2

    def __init__(self, inplanes, outplanes, stride=1):
        super(Bottleneck_2, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.conv3 = nn.Conv2d(outplanes, outplanes*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes*self.expansion)

        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != outplanes*self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes*self.expansion),
            )

    def forward(self, x):
        # import pdb; pdb.set_trace()
        xf = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.bn3(self.conv3(x))
        x += self.downsample(xf)
        x = F.relu(x, inplace=True)
        return x


class FPN(nn.Module):
    
    def __init__(self, block, numblocks, outchannel, upconv=False):
        super(FPN, self).__init__()
        self.inplane = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, numblocks[0], stride=1) # out 64*4layers
        self.layer2 = self._make_layer(block, 128, numblocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, numblocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, numblocks[3], stride=2) # stride=16*maxpool2d, out:512*2layers

        self.skip2 = nn.Conv2d(256, outchannel, kernel_size=1, stride=1)
        self.skip3 = nn.Conv2d(512, outchannel, kernel_size=1, stride=1)
        self.skip4 = nn.Conv2d(1024, outchannel, kernel_size=1, stride=1)
        self.skip5 = nn.Conv2d(2048, outchannel, kernel_size=1, stride=1)

        if upconv:
            self.upconv2 = nn.Conv2d(outchannel, outchannel, kernel_size=1, stride=1)
            self.upconv3 = nn.Conv2d(outchannel, outchannel, kernel_size=1, stride=1)
            self.upconv4 = nn.Conv2d(outchannel, outchannel, kernel_size=1, stride=1)
        
        # conv before element-size
        self.sum2 = nn.Conv2d(outchannel, outchannel, kernel_size=1, stride=1)
        self.sum3 = nn.Conv2d(outchannel, outchannel, kernel_size=1, stride=1)
        self.sum4 = nn.Conv2d(outchannel, outchannel, kernel_size=1, stride=1)
        # self.sum5 = nn.Conv2d(1024, outchannel, kernel_size=1, stride=1)

        self.initParameter()
        
    def _make_layer(self, block, outplane, numblock, stride):
        strides = [stride] + [1] * (numblock-1) #list
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, outplane, stride))
            self.inplane = outplane * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _,_,h,w = y.size()
        return F.upsample(x, size=(h,w), mode='bilinear') + y

    def _upsample(self, x, y):
        _,_,h,w = y.size()
        return F.upsample(x, size=(h,w), mode='bilinear')

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.skip5(c5)
        c4 = self.skip4(c4)
        p4 = self._upsample_add(self.sum4(p5), c4)
        c3 = self.skip3(c3)
        p3 = self._upsample_add(self.sum3(p4), c3)
        c2 = self.skip2(c2)
        p2 = self._upsample_add(self.sum2(p3), c2)

        return [p2,p3,p4,p5] # p2 is biggest
    
    def initParameter(self):
        res = torchvision.models.resnet50(pretrained=True)
        res_dict = res.state_dict()
        fpn_dict = self.state_dict()
        
        for k in res_dict.keys():
            if not k.startswith('fc'):
                fpn_dict[k] = res_dict[k]
                
        self.load_state_dict(fpn_dict)

def FPN18(outchannel=256):
    return FPN(BasicBlock, [2,2,2,2], outchannel)

def FPN50(outchannel=256,upconv=False):
    return FPN(Bottleneck, [3,4,6,3], outchannel,upconv=upconv)


if __name__ == "__main__":
    import torch
    from torch.autograd import Variable
    x = torch.randn(1,3,512,512)
    x = Variable(x, volatile=True)
    net = FPN50()
    outs = net(x)
    for out in outs:
        print(out.size())