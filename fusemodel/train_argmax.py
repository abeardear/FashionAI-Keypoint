#encoding:utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

from argsoftmax_cpn_hgv2 import CPN_hg_v2_argmax
from mutilloss import MultiLoss, cordLoss
from dataset import fashionDataset,part_name
from evaluate import *
print(torch.cuda.device_count())
parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument(
      "--select_part",
      "--s",
      type=str,
      default='trousers',
      help="Which class to train.")
args = parser.parse_args()
select_part = args.select_part
trainDataRoot = '../data/train/'
annoRoot = '../data/train/Annotations/'
assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_loss = float('inf')  # best test loss
best_acc = 0
best_NE = float('inf')
start_epoch = 0  # start from epoch 0 or last epoch
learning_rate = 1e-3
resume = True   
# select_part = 'trousers'
# Data
print('==> Preparing data..')

train_dataset = fashionDataset(annoRoot+'train_round2.csv',trainDataRoot,train=True,transform=[transforms.ToTensor()], \
    select_part=select_part,ingore_unvis=False, ksize=13,color_jitter=True)
train_loader = DataLoader(train_dataset,batch_size=4,shuffle=True,num_workers=8)
test_dataset = fashionDataset(annoRoot+'valid.csv',trainDataRoot,train=False,transform=[transforms.ToTensor()], \
    select_part=select_part,ingore_unvis=False, ksize=13)
test_loader = DataLoader(test_dataset,batch_size=4,shuffle=False,num_workers=8)

# Model
# net = HourglassNet_mssa(block=Bottleneck, num_stacks = 8,num_blocks=1, num_classes=len(part_name[select_part]),image_c=3)
net = CPN_hg_v2_argmax(num_classes=len(part_name[select_part]))
# checkpoint = torch.load('checkpoint/cpn-hg-v2-' +select_part +'.pth')
# start_epoch = checkpoint['epoch']
# net.load_state_dict(checkpoint['net'])
# net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
# for name,m in net.named_parameters():
#     print(name)
# net.load_state_dict(torch.load('model/net_init.pth'))
if resume:
    print('==> Resuming from checkpoint..')
    # checkpoint = torch.load('./checkpoint/ckpt'+select_part+'.pth')
    checkpoint = torch.load('../model_best.pth.tar')
    # original saved file with DataParallel
    state_dict = checkpoint['state_dict']
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        # name = k[:]
        new_state_dict[name] = v
    # load params
    dd = net.state_dict()
    for k in new_state_dict.keys():
        # if not k.startswith('score'):  # skip layers
        if k.find('score') <0: #if don't have score, we load the weight
            dd[k] = new_state_dict[k]
    net.load_state_dict(dd)
    # # for k,v in checkpoint['state_dict'].items():
    # #     print(k)
    # net.load_state_dict(checkpoint['net'])
    # best_loss = checkpoint['loss']
    # best_acc = checkpoint['acc']
    # start_epoch = checkpoint['epoch']
# net.load_state_dict(torch.load('../checkpoint/argmax-blouse.pth')['net'])
net.cuda()
# class MultiLoss(nn.Module):
#     def __init__(self, size_average):
#         super(MultiLoss, self).__init__()
#         self.mseloss = nn.MSELoss(size_average=size_average)
        
#     def forward(self, outs, heatmap, visiables):
#         # global net loss
#         # import pdb; pdb.set_trace()
#         global_loss = Variable(torch.Tensor([0])).cuda()
#         for item in outs:
#             _,_,h,w = item.shape
#             heatmap_tmp = F.upsample(heatmap, size=(h,w), mode='bilinear')
#             # visiables_mask = visiables.unsqueeze(2).unsqueeze(3).expand_as(heatmap_tmp)
#             # masked_out = item[visiables_mask]
#             # masked_heatmap = heatmap_tmp[visiables_mask]
#             # global_loss += self.mseloss(masked_out, masked_heatmap)
#             global_loss += self.mseloss(item, heatmap_tmp)
#         global_loss /= len(outs)
#         return global_loss
# class valLoss(nn.Module):
#     def __init__(self, size_average):
#         super(valLoss, self).__init__()
#         self.mseloss = nn.MSELoss(size_average=size_average)
        
#     def forward(self, out, heatmap, visiables):
#         visiables_mask = visiables.unsqueeze(2).unsqueeze(3).expand_as(heatmap)
#         masked_out = out[visiables_mask]
#         masked_heatmap = heatmap[visiables_mask]
#         loss = self.mseloss(masked_out, masked_heatmap)
#         return loss

# criterion_train = MultiLoss(size_average=False).cuda()
# criterion = torch.nn.MSELoss(size_average=False).cuda()
# criterion = valLoss(size_average=False).cuda()
criterion_train = MultiLoss(size_average=True).cuda()
criterion_cord = cordLoss(size_average=True).cuda()
criterion = torch.nn.MSELoss(size_average=True).cuda()
# optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
# 指定各层的lr
params = []
skip_layer=['fpn.skip2.weight',
'fpn.skip2.bias',
'fpn.skip3.weight',
'fpn.skip3.bias',
'fpn.skip4.weight',
'fpn.skip4.bias',
'fpn.skip5.weight',
'fpn.skip5.bias',
'fpn.sum2.weight',
'fpn.sum2.bias',
'fpn.sum3.weight',
'fpn.sum3.bias',
'fpn.sum4.weight',
'fpn.sum4.bias']
params_dict = dict(net.named_parameters())
for key, value in params_dict.items():
    if key.startswith('fpn') and (key not in skip_layer):#fpn中学习率小一点?
        params += [{'params':[value],'lr':learning_rate*0.1}]
    elif key.startswith('argmax'):
        params += [{'params':[value],'lr':0}] #不能训练argmax的固定卷积核 
    else:
        params += [{'params':[value],'lr':learning_rate}]
optimizer = optim.Adam(params, lr=learning_rate, weight_decay=1e-5)
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
# vis = visualize.Visualizer(env='CPN', title_name=select_part+'loss')
# acc_vis = visualize.Visualizer(env='CPN', title_name='NE')
logfile = open('argmax-'+select_part+'.txt','w')
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    traincord_loss = 0
    # acc_average = 0
    # NE_average = 0
    max_iters = len(train_loader)
    for batch_idx, (img, heatmap, x_label, y_label, visiables) in enumerate(train_loader):
        optimizer.zero_grad()
        img = Variable(img.cuda())
        heatmap = Variable(heatmap.cuda())
        visiables = Variable(visiables.cuda())
        x_label = Variable(x_label.cuda())
        y_label = Variable(y_label.cuda())

        output, x_pred, y_pred,x_pred1, y_pred1,x_pred2, y_pred2 = net(img)
        # score_map = output[-1].data.cpu()
        # target_map = heatmap.data.cpu()
        # acc = accuracy(score_map, target_map, len(part_name[select_part]))
        # NE = ne_evaluate(score_map,target_map,part_name[select_part],select_part)
        loss = criterion_train(output[:], heatmap, visiables)
        cord_loss = criterion_cord(x_pred, x_label, y_pred, y_label, visiables)
        cord_loss1 = criterion_cord(x_pred1, x_label, y_pred1, y_label, visiables)
        cord_loss2 = criterion_cord(x_pred2, x_label, y_pred2, y_label, visiables)
        # for i in range(1,len(output)/4):
        #     loss += criterion_train(output[4*i:4*i+4], heatmap, visiables)
        # loss += criterion_train(output[-1:],heatmap, visiables) #mutil-scale-regression-loss
        # acc_average += acc
        # NE_average += NE
        loss_total = loss + cord_loss + cord_loss1 + cord_loss2 #平衡损失大小?
        loss_total.backward()
        optimizer.step()

        train_loss += loss.data[0]
        traincord_loss += cord_loss.data[0]
        print('epoch: %d | iter: %d/%d  | train_loss: %.3f | avg_loss: %.3f' 
            % (epoch, batch_idx, max_iters,traincord_loss/(batch_idx+1), train_loss/(batch_idx+1)))
        # vis.plot_train_val(loss_train=train_loss/(batch_idx+1))
        # acc_vis.plot('acc',acc_average)
        # acc_vis.plot_train_val(loss_train=NE_average/(batch_idx+1))

# Test
def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    acc_average = 0
    NE_average = 0
    pred_csv(net,select_part,name='argmax') #get pred.csv
    for batch_idx, (img, heatmap, x_label, y_label, visiables) in enumerate(test_loader):
        img = Variable(img.cuda(), volatile=True)
        # img = Variable(img.cuda())
        # print(img.size())
        heatmap = Variable(heatmap.cuda(),volatile=True)
        visiables = Variable(visiables.cuda())
        # heatmap = Variable(heatmap.cuda())

        output,px,py,_,_,_,_ = net(img)
        score_map = output[-1].data.cpu()
        target_map = heatmap.data.cpu()
        acc = accuracy(score_map, target_map, len(part_name[select_part]))
        # NE = ne_evaluate(score_map,target_map,part_name[select_part],select_part)
        loss = criterion(output[-1], heatmap)

        acc_average += acc
        # NE_average += NE
        test_loss += loss.data[0]
        #print('test_loss: %.3f | avg_loss: %.3f' % (loss.data[0], test_loss/(batch_idx+1)))
    NE_average = normalized_error(select_part,'argmax') #get NE from pred.csv and valid.csv
    # vis.plot_train_val(loss_val=test_loss/len(test_loader))
    acc_average  = acc_average/len(test_loader)
    # NE_average = NE_average/len(test_loader)
    print(NE_average)
    # acc_vis.plot(select_part+'NE',NE_average)
    logfile.writelines(str(epoch) + '\t' + str(NE_average) + '\n')
    logfile.flush()

    # Save checkpoint
    global best_loss
    global best_acc
    global best_NE
    test_loss /= len(test_loader)
    if best_NE > NE_average:
        print('Saving bestNE..')
        # state = {
        #     'net': net.module.state_dict(),
        #     'loss': test_loss,
        #     'epoch': epoch,
        # }
        state = {
            'net': net.state_dict(),
            'loss': test_loss,
            'acc' : acc_average,
            'NE' : NE_average,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state,'./checkpoint/argmax-' +select_part +'.pth')
        best_NE = NE_average
    # if epoch in [17,21]:
    #     state = {
    #         'net': net.state_dict(),
    #         'loss': test_loss,
    #         'acc' : acc_average,
    #         'NE' : NE_average,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state,'./checkpoint/' +select_part +str(epoch)+'.pth')
    # if acc_average > best_acc:
    #     print('Saving bestacc..')
    #     # state = {
    #     #     'net': net.module.state_dict(),
    #     #     'loss': test_loss,
    #     #     'epoch': epoch,
    #     # }
    #     state = {
    #         'net': net.state_dict(),
    #         'loss': test_loss,
    #         'acc' : acc_average,
    #         'NE' : NE_average,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state,'checkpoint/ckpt512bestacc-s2n1-128-' +select_part +'.pth')
    #     best_acc = acc_average
# def freeze_bn():
#     net.eval()
#     params=[]
#     for key, value in params_dict.items():
#         if key.find('bn')>0:
#             params += [{'params':[value],'lr':0}]
#         elif key.startswith('fpn') and (key not in skip_layer):
#             params += [{'params':[value],'lr':learning_rate*0.1}]
#         else:
#             params += [{'params':[value],'lr':learning_rate}]
#     optimizer = optim.Adam(params, weight_decay=1e-5)
#     return optimizer
timefile = open('time.txt','w')    
test(start_epoch)
for epoch in range(start_epoch, start_epoch+30):

    # adjust learning rate
    # if epoch == 20:

    # if epoch ==5:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = learning_rate #5轮之后,学习率一致
    if epoch == 10 or epoch==20:
        learning_rate *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1 #learning_rate
    # if epoch ==20:
    #     optimizer = freeze_bn() #將bn层的学习率设置为0

    start = time.time()
    
    train(epoch)
    test(epoch)

    end = time.time()
    train_time = end-start
    timefile.writelines(str(train_time)+ '\n')
    timefile.flush()

    # if epoch == 15:
    #     optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    #     learning_rate *= 0.1
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = learning_rate
