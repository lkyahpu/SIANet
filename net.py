from math import sqrt
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import ml_collections


from utils import *
import os
from loss import *
from model import *
# from skimage.feature.tests.test_orb import img

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 768  # KV_size = Q1 + Q2 + Q3 + Q4 = 960
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.expand_ratio = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 64  # base channel of U-Net
    config.n_classes = 1
    return config
class Net(nn.Module):
    def __init__(self, model_name, mode):
        super(Net, self).__init__()
        self.model_name = model_name
        
        self.cal_loss = SoftIoULoss()
        if model_name == 'DNANet':
            if mode == 'train':
                self.model = DNANet(mode='train')
            else:
                self.model = DNANet(mode='test')  
        # elif model_name == 'DNANet_BY':
        #     if mode == 'train':
        #         self.model = DNAnet_BY(mode='train')
        #     else:
        #         self.model = DNAnet_BY(mode='test')
        elif model_name == 'ACM':
            self.model = ACM()
        elif model_name == 'ALCNet':
            self.model = ALCNet()
        # elif model_name == 'ISNet':
        #     if mode == 'train':
        #         self.model = ISNet(mode='train')
        #     else:
        #         self.model = ISNet(mode='test')
            self.cal_loss = ISNetLoss()
        # elif model_name == 'RISTDnet':
        #     self.model = RISTDnet()
        elif model_name == 'UIUNet':
            if mode == 'train':
                self.model = UIUNet(mode='train')
            else:
                self.model = UIUNet(mode='test')
        elif model_name == 'U-Net':
            self.model = Unet()
        elif model_name == 'ISTDU-Net':
            self.model = ISTDU_Net()
        elif model_name == 'RDIAN':
            self.model = RDIAN()
        elif model_name == 'SIANet':
            self.model = SIANet()
            self.cal_loss = MaskWeightedHeatmapLoss()
        elif model_name == 'FTCNet':
            config_vit = get_CTranS_config()
            self.model = USwinT(config_vit, n_channels=3, n_classes=1)
            self.cal_loss = MaskWeightedHeatmapLoss()
        
    def forward(self, img):
        return self.model(img)

    def loss(self, pred, gt_mask):
        loss = self.cal_loss(pred, gt_mask)
        return loss


