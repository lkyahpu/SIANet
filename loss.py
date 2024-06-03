import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()
    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                smooth = 1
                intersection = pred * gt_masks
                loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
                loss = 1 - loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = preds
            smooth = 1
            intersection = pred * gt_masks
            loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
            loss = 1 - loss.mean()
            return loss

class ISNetLoss(nn.Module):
    def __init__(self):
        super(ISNetLoss, self).__init__()
        self.softiou = SoftIoULoss()
        self.bce = nn.BCELoss()
        self.grad = Get_gradient_nopadding()
        
    def forward(self, preds, gt_masks):
        edge_gt = self.grad(gt_masks.clone())
        
        ### images loss
        loss_img = self.softiou(preds[0], gt_masks)
        
        ### edge loss
        loss_edge = 10 * self.bce(preds[1], edge_gt)+ self.softiou(preds[1].sigmoid(), edge_gt)
        
        return loss_img + loss_edge

def dice_loss(pred,target):
    # pred = pred.sigmoid()
    smooth = 0.00

    intersection = pred * target

    intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
    pred_sum = torch.sum(pred, dim=(1, 2, 3))
    target_sum = torch.sum(target, dim=(1, 2, 3))
    loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)

    loss = 1 - torch.mean(loss)
    return loss
class WeightedBCELoss(nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()
    def forward(self, inputs, targets, threshord=0.4):
        # 计算目标像素和背景像素的占比
        gt = (targets > threshord).float()
        pt = (inputs > threshord).float()
        Mt = (targets > 0.5).float()
        loss_dice = dice_loss(inputs, Mt)
        pos_ratio = torch.mean(Mt)
        neg_ratio = 1 - pos_ratio

        squared_difference = np.array(torch.abs(pt-gt).cpu(), dtype=np.float32)
        loss_ou = np.sum(squared_difference) / gt.size(0)
        weight = neg_ratio / np.maximum(pos_ratio.cpu(), 1e-6)
        # 创建BCELoss
        bce_loss = nn.BCELoss(weight=weight)

        # 计算带权重调整的Loss
        loss1 = bce_loss(inputs, Mt)

        loss = loss1 + loss_dice*10 + 0.2*loss_ou
        return loss
class MaskWeightedHeatmapLoss(torch.nn.Module):
    def __init__(self, delta=0.5, lambda_val=0.25):
        super(MaskWeightedHeatmapLoss, self).__init__()
        self.delta = delta
        self.lambda_val = lambda_val

    def forward(self, predicted_heatmap, target_heatmap):
        squared_difference = torch.pow(predicted_heatmap - target_heatmap, 2)

        # 生成mask-weighted mask
        Mt = (predicted_heatmap > self.delta).float()

        # 计算最终的损失函数
        loss = torch.sum(squared_difference*(Mt + self.lambda_val * (1 - Mt)))/predicted_heatmap.shape[0]
        return torch.mean(loss)
