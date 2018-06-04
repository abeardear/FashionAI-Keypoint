import torch
import torch.nn as nn
import torch.nn.functional as F
from evaluate import get_preds
from torch.autograd import Variable

# class MultiLoss(nn.Module):
#     def __init__(self, size_average):
#         super(MultiLoss, self).__init__()
#         self.mseloss = nn.MSELoss(size_average=size_average)
        
#     def forward(self, outs, heatmap, visiable):
#         N, C, W, H = heatmap.shape
#         # import pdb; pdb.set_trace()
#         # global net loss
#         global_loss = Variable(torch.Tensor([0])).cuda()
#         for item in outs[:-1]:
#             _,_,h,w = item.shape
#             heatmap_tmp = F.upsample(heatmap, size=(h,w), mode='bilinear')
#             global_loss += self.mseloss(item, heatmap_tmp)
#         global_loss /= len(outs[:-1]) #140
#         # print(global_loss)


#         hard_channel = int(C*0.5)
#         aver = N*hard_channel*W*H
#         refine_loss = F.mse_loss(outs[-1], heatmap,reduce=False)
#         refine_loss = refine_loss.sum(dim=2).sum(dim=2)  #148
#         # import pdb; pdb.set_trace()
        
#         # hard example mining msemseloss
#         refine_loss, _ = refine_loss.sort(dim=1, descending=True)
#         refine_loss = refine_loss[:,:hard_channel]

#         # hard example mining distance
#         # preds = get_preds(outs[-1].data)
#         # gts = get_preds(heatmap.data)
#         # dist = (preds-gts)**2
#         # dist = dist.sum(2)
#         # _, idx = dist.sort(descending=True)
#         # # variable must use with variable index
#         # refine_loss = refine_loss.gather(1, Variable(idx[:,:hard_channel]))
        
#         # visiable desert,  doesn't work
#         # refine_loss = refine_loss[visiable]
#         # refine_loss,_ = refine_loss.sort(descending=True)
#         # if refine_loss.size(0) > hard_channel*N:
#         #     refine_loss = refine_loss[:hard_channel*N]
#         refine_loss = refine_loss.sum().div(aver)
        
#         return global_loss + refine_loss


class MultiLoss(nn.Module):
    def __init__(self, size_average):
        super(MultiLoss, self).__init__()
        self.mseloss = nn.MSELoss(size_average=size_average)
        
    def forward(self, outs, heatmap, visiable):
        N, C, H, W = heatmap.shape
        # import pdb; pdb.set_trace()
        # global net loss
        global_loss = Variable(torch.Tensor([0])).cuda()
        for item in outs[:]:
            _,_,h,w = item.shape
            heatmap_tmp = F.upsample(heatmap, size=(h,w), mode='bilinear')
            visiables_mask = visiable.unsqueeze(2).unsqueeze(3).expand_as(heatmap_tmp)
            masked_out = item[visiables_mask]
            masked_heatmap = heatmap_tmp[visiables_mask]
            global_loss += self.mseloss(masked_out, masked_heatmap)
            # global_loss += self.mseloss(item, heatmap_tmp)
        global_loss /= len(outs[:]) #140
        # print(global_loss)


        hard_channel = int(N*C*0.5)
        aver = N*hard_channel*W*H
        visiables_mask = visiable.unsqueeze(2).unsqueeze(3).expand_as(heatmap)
        masked_out = outs[-1][visiables_mask].view(-1,H,W)
        masked_heatmap = heatmap[visiables_mask].view(-1,H,W)
        refine_loss = F.mse_loss(masked_out, masked_heatmap, size_average=False ,reduce=False)
        refine_loss = refine_loss.sum(dim=1).sum(dim=1)  #148
        # refine_loss = F.mse_loss(outs[:-1], heatmap, size_average=False ,reduce=False)
        # import pdb; pdb.set_trace()
        
        # hard example mining mse
        refine_loss, _ = refine_loss.sort(dim=0, descending=True)
        refine_loss = refine_loss[:hard_channel]

        # hard example mining distance
        # preds = get_preds(outs[-1].data)
        # gts = get_preds(heatmap.data)
        # dist = (preds-gts)**2
        # dist = dist.sum(2)
        # _, idx = dist.sort(descending=True)
        # # variable must use with variable index
        # refine_loss = refine_loss.gather(1, Variable(idx[:,:hard_channel]))
        
        # visiable desert,  doesn't work
        # refine_loss = refine_loss[visiable]
        # refine_loss,_ = refine_loss.sort(descending=True)
        # if refine_loss.size(0) > hard_channel*N:
        #     refine_loss = refine_loss[:hard_channel*N]
        refine_loss = refine_loss.sum().div(aver)
        
        return global_loss + refine_loss

def GetLabel(out, heatmap):
        # import pdb; pdb.set_trace()
        preds = get_preds(out.data)
        gts = get_preds(heatmap.data)
        dist = (preds-gts)**2
        dist = dist.sum(2)
        label = dist < 4 # 100->50->18->9
        return label

class cordLoss(nn.Module):
    def __init__(self, size_average):
        super(cordLoss, self).__init__()
        self.mseloss = nn.MSELoss(size_average=size_average)
    
    def forward(self, x_pred, x_label, y_pred, y_label, visiable):
        masked_xpred = x_pred[visiable]
        masked_xlabel = x_label[visiable]
        loss1 = self.mseloss(masked_xpred, masked_xlabel)
        
        masked_ypred = y_pred[visiable]
        masked_ylabel = y_label[visiable]
        loss2 = self.mseloss(masked_ypred, masked_ylabel)

        loss = loss1+loss2
        loss = loss*10
        return loss
