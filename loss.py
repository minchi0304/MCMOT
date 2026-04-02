import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Loss(nn.Module):

    def __init__(self, loss_select, k, p):
        super().__init__()
        self.loss_select = loss_select
        self.p = p
        self.k = k

    def kldiv(self, s_map, gt):
        batch_size = s_map.size(0)
        w = s_map.size(1)
        h = s_map.size(2)
        sum_s_map = torch.sum(s_map.view(batch_size, -1), 1)
        expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, w, h)
    
        assert expand_s_map.size() == s_map.size()

        sum_gt = torch.sum(gt.view(batch_size, -1), 1)
        expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, w, h)
        
        assert expand_gt.size() == gt.size()

        s_map = s_map/(expand_s_map*1.0)
        gt = gt / (expand_gt*1.0)

        s_map = s_map.view(batch_size, -1)
        gt = gt.view(batch_size, -1)

        eps = 2.2204e-16
        result = gt * torch.log(eps + gt/(s_map + eps))
        # print(torch.log(eps + gt/(s_map + eps))   )
        return torch.mean(torch.sum(result, 1))

    def cc(self, s_map, gt):
        batch_size = s_map.size(0)
        w = s_map.size(1)
        h = s_map.size(2)
    
        mean_s_map = torch.mean(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
        std_s_map = torch.std(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    
        mean_gt = torch.mean(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
        std_gt = torch.std(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    
        s_map = (s_map - mean_s_map) / std_s_map
        gt = (gt - mean_gt) / std_gt
    
        ab = torch.sum((s_map * gt).view(batch_size, -1), 1)
        aa = torch.sum((s_map * s_map).view(batch_size, -1), 1)
        bb = torch.sum((gt * gt).view(batch_size, -1), 1)
    
        return torch.mean(ab / (torch.sqrt(aa*bb)))


    def forward(self, x, target, kernel):
        # target = self.target_transform_(x, target, kernel)
        if self.loss_select == 'klcc':
            #### KL div + cc ####
            return (self.k*self.kldiv(x[0], target[0])) - (self.p*self.cc(x[0], target[0]))
            ################################################
        elif self.loss_select == 'mse':
            return F.mse_loss(x, target)


    def target_transform_(self, x, target, kernel):
        target = F.adaptive_max_pool2d(target, x.shape[2:])
        with torch.no_grad():
            if self.loss_select == 'klcc':
                #### Use Sigmoid for KLDiv+cc ####
                target = torch.sigmoid(F.conv2d(target, kernel.float().to(target.device), padding=int((kernel.shape[-1] - 1) / 2)))
                ############################################
            elif self.loss_select == 'mse':
                target = F.conv2d(target, kernel.float().to(target.device), padding=int((kernel.shape[-1] - 1) / 2))
        return target

class FocalLoss(torch.nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self, use_distance_weight=False):
        super(FocalLoss, self).__init__()
        self.use_distance_weight = use_distance_weight

    def forward(self, pred, gt):
        """ Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
            Arguments:
                pred (batch x c x h x w)
                gt_regr (batch x c x h x w)
        """
        # find pos indices and neg indices
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        distance_weight = torch.ones_like(gt)
        if self.use_distance_weight:
            w, h = gt.shape[-2:]
            xs = torch.linspace(-1, 1, steps=h, device=gt.device)
            ys = torch.linspace(-1, 1, steps=w, device=gt.device)
            x, y = torch.meshgrid(xs, ys, indexing='xy')
            distance_weight = 9 * torch.sin(torch.sqrt(x * x + y * y)) + 1

        # following paper alpha 2, beta 4
        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds * distance_weight
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds * distance_weight

        num_pos = pos_inds.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss



