import torch
import torch.nn.functional as F

from utils import _sigmoid, _transpose_and_gather_feat

def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)
    loss = 0
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(torch.nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target) 


class RegL1Loss(torch.nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred*mask, target*mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class CtdetLoss(torch.nn.Module):
    def __init__(self, cfg):
        super(CtdetLoss, self).__init__()
        self.cfg = cfg
        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss()

    def forward(self, outputs, batch):
        hm_loss, wh_loss, off_loss = 0, 0, 0
        for s in range(self.cfg['num_stacks']):
            output = outputs[s]
            if not self.cfg['mse_loss']:
                output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / self.cfg['num_stacks']
            if self.cfg['wh_weight'] > 0:
                wh_loss += self.crit_reg(output['wh'], batch['reg_mask'], batch['ind'], batch['wh']) / self.cfg['num_stacks']
 
            if self.cfg['reg_offset'] and self.cfg['off_weight'] > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'], batch['ind'], batch['reg']) / self.cfg['num_stacks']
 
            loss = self.cfg['hm_weight'] * hm_loss + self.cfg['wh_weight'] * wh_loss + self.cfg['off_weight'] * off_loss
            loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss, 'off_loss': off_loss}
        return loss, loss_stats