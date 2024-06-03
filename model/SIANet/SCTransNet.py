# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# @Author  : Shuai Yuan
# @File    : SCTransNet.py
# @Software: PyCharm
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import math
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import torch.nn as nn
import torch
import torch.nn.functional as F
import ml_collections
from einops import rearrange
import numbers
# from model.DNANet.model_DNANet import Res_CBAM_block

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 224  # KV channel dimension size = C1 + C2 + C3 + C4
    config.transformer.num_layers = 3  # the number of SCTB
    config.expand_ratio = 2.66  # CFN channel dimension expand ratio
    config.base_channel = 32  # base channel of SCTransNet
    config.patch_sizes = [16, 8, 4]
    # config.patch_sizes = [32, 16, 8, 4]
    config.n_classes = 1
    return config

# 定义深度可分离卷积层
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Channel_Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, patchsize, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)
        return x


class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None
        x = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear')(x)
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


# spatial-embedded Single-head Channel-cross Attention (SSCA)
class SSCA(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(SSCA, self).__init__()
        self.vis = vis
        self.KV_size = config.KV_size
        self.num_attention_heads = 1
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        # self.mhead1 = nn.Conv2d(channel_num[0], channel_num[0] * self.num_attention_heads, kernel_size=1, bias=False)
        # self.mhead2 = nn.Conv2d(channel_num[1], channel_num[1] * self.num_attention_heads, kernel_size=1, bias=False)
        # self.mhead3 = nn.Conv2d(channel_num[2], channel_num[2] * self.num_attention_heads, kernel_size=1, bias=False)
        # self.mhead4 = nn.Conv2d(channel_num[3], channel_num[3] * self.num_attention_heads, kernel_size=1, bias=False)
        # self.mheadk = nn.Conv2d(self.KV_size, self.KV_size * self.num_attention_heads, kernel_size=1, bias=False)
        # self.mheadv = nn.Conv2d(self.KV_size, self.KV_size * self.num_attention_heads, kernel_size=1, bias=False)

        # self.q1 = nn.Conv2d(channel_num[0] * self.num_attention_heads, channel_num[0] * self.num_attention_heads,
        #                     kernel_size=3, stride=1,
        #                     padding=1,
        #                     groups=channel_num[0] * self.num_attention_heads // 2, bias=False)
        # self.q2 = nn.Conv2d(channel_num[1] * self.num_attention_heads, channel_num[1] * self.num_attention_heads,
        #                     kernel_size=3, stride=1,
        #                     padding=1,
        #                     groups=channel_num[1] * self.num_attention_heads // 2, bias=False)
        # self.q3 = nn.Conv2d(channel_num[2] * self.num_attention_heads, channel_num[2] * self.num_attention_heads,
        #                     kernel_size=3, stride=1,
        #                     padding=1,
        #                     groups=channel_num[2] * self.num_attention_heads // 2, bias=False)
        # # self.q4 = nn.Conv2d(channel_num[3] * self.num_attention_heads, channel_num[3] * self.num_attention_heads,
        # #                     kernel_size=3, stride=1,
        # #                     padding=1,
        # #                     groups=channel_num[3] * self.num_attention_heads // 2, bias=False)
        # self.k = nn.Conv2d(self.KV_size * self.num_attention_heads, self.KV_size * self.num_attention_heads,
        #                    kernel_size=3, stride=1,
        #                    padding=1, groups=self.KV_size * self.num_attention_heads, bias=False)
        # self.v = nn.Conv2d(self.KV_size * self.num_attention_heads, self.KV_size * self.num_attention_heads,
        #                    kernel_size=3, stride=1,
        #                    padding=1, groups=self.KV_size * self.num_attention_heads, bias=False)
        self.q1 = DepthwiseSeparableConv(channel_num[0], channel_num[0] * self.num_attention_heads)
        self.q2 = DepthwiseSeparableConv(channel_num[1], channel_num[1] * self.num_attention_heads)
        self.q3 = DepthwiseSeparableConv(channel_num[2], channel_num[2] * self.num_attention_heads)

        self.k = DepthwiseSeparableConv(self.KV_size, self.KV_size * self.num_attention_heads)
        self.v = DepthwiseSeparableConv(self.KV_size, self.KV_size * self.num_attention_heads)
        self.project_out1 = nn.Conv2d(channel_num[0], channel_num[0], kernel_size=1, bias=False)
        self.project_out2 = nn.Conv2d(channel_num[1], channel_num[1], kernel_size=1, bias=False)
        self.project_out3 = nn.Conv2d(channel_num[2], channel_num[2], kernel_size=1, bias=False)
        # self.project_out4 = nn.Conv2d(channel_num[3], channel_num[3], kernel_size=1, bias=False)

    def forward(self, emb1, emb2, emb3, emb_all):
        b, c, h, w = emb1.shape

        #  spatial-embedded
        q1 = self.q1(emb1)
        q2 = self.q2(emb2)
        q3 = self.q3(emb3)
        # q4 = self.q4(self.mhead4(emb4))
        k = self.k(emb_all)
        v = self.v(emb_all)

        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        q3 = rearrange(q3, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        # q4 = rearrange(q4, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        q3 = torch.nn.functional.normalize(q3, dim=-1)
        # q4 = torch.nn.functional.normalize(q4, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, c1, _ = q1.shape
        _, _, c2, _ = q2.shape
        _, _, c3, _ = q3.shape
        # _, _, c4, _ = q4.shape
        _, _, c, _ = k.shape

        # Attention
        attn1 = (q1 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)
        attn2 = (q2 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)
        attn3 = (q3 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)
        # attn4 = (q4 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)

        attention_probs1 = self.softmax(self.psi(attn1))
        attention_probs2 = self.softmax(self.psi(attn2))
        attention_probs3 = self.softmax(self.psi(attn3))
        # attention_probs4 = self.softmax(self.psi(attn4))

        out1 = (attention_probs1 @ v)
        out2 = (attention_probs2 @ v)
        out3 = (attention_probs3 @ v)
        # out4 = (attention_probs4 @ v)

        out_1 = out1.mean(dim=1)
        out_2 = out2.mean(dim=1)
        out_3 = out3.mean(dim=1)
        # out_4 = out4.mean(dim=1)

        out_1 = rearrange(out_1, 'b  c (h w) -> b c h w', h=h, w=w)
        out_2 = rearrange(out_2, 'b  c (h w) -> b c h w', h=h, w=w)
        out_3 = rearrange(out_3, 'b  c (h w) -> b c h w', h=h, w=w)
        # out_4 = rearrange(out_4, 'b  c (h w) -> b c h w', h=h, w=w)

        O1 = self.project_out1(out_1)
        O2 = self.project_out2(out_2)
        O3 = self.project_out3(out_3)
        # O4 = self.project_out4(out_4)
        weights = None

        return O1, O2, O3, weights


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm3d(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm3d, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class GSLC(nn.Module):

    def __init__(self, channel, k_size=3):
        super(GSLC, self).__init__()
        padding = k_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        out = self.avg_pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x


# Complementary Feed-forward Network (CFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                   groups=hidden_features,
                                   bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2,
                                   groups=hidden_features,
                                   bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)
        self.eca = GSLC(dim)

    def forward(self, x):
        x_3, x_5 = self.project_in(x).chunk(2, dim=1)
        x1_3 = self.relu3(self.dwconv3x3(x_3))
        x1_5 = self.relu5(self.dwconv5x5(x_5))
        x = torch.cat([x1_3, x1_5], dim=1)
        x = self.project_out(x)
        x = self.eca(x)

        return x

#  Spatial-channel Cross Transformer Block (SCTB)
class SCTB(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(SCTB, self).__init__()
        expand_ratio = config.expand_ratio
        self.attn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.attn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.attn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        # self.attn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')
        self.attn_norm = LayerNorm3d(config.KV_size, LayerNorm_type='WithBias')

        self.channel_attn = SSCA(config, vis, channel_num)

        self.ffn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.ffn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.ffn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        # self.ffn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')

        # self.ffn1 = FeedForward(channel_num[0], ffn_expansion_factor=expand_ratio, bias=False)
        # self.ffn2 = FeedForward(channel_num[1], ffn_expansion_factor=expand_ratio, bias=False)
        # self.ffn3 = FeedForward(channel_num[2], ffn_expansion_factor=expand_ratio, bias=False)
        # self.ffn4 = FeedForward(channel_num[3], ffn_expansion_factor=expand_ratio, bias=False)

        self.ffn1 = Res_CBAM_block(channel_num[0], channel_num[0])
        self.ffn2 = Res_CBAM_block(channel_num[1], channel_num[1])
        self.ffn3 = Res_CBAM_block(channel_num[2], channel_num[2])

    def forward(self, emb1, emb2, emb3):
        embcat = []
        org1 = emb1
        org2 = emb2
        org3 = emb3
        # org4 = emb4
        for i in range(3):
            var_name = "emb" + str(i + 1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)
        emb_all = torch.cat(embcat, dim=1)
        cx1 = self.attn_norm1(emb1) if emb1 is not None else None
        cx2 = self.attn_norm2(emb2) if emb2 is not None else None
        cx3 = self.attn_norm3(emb3) if emb3 is not None else None
        # cx4 = self.attn_norm4(emb4) if emb4 is not None else None
        emb_all = self.attn_norm(emb_all)  # 1 196 960
        cx1, cx2, cx3, weights = self.channel_attn(cx1, cx2, cx3, emb_all)
        cx1 = org1 + cx1 if emb1 is not None else None
        cx2 = org2 + cx2 if emb2 is not None else None
        cx3 = org3 + cx3 if emb3 is not None else None
        # cx4 = org4 + cx4 if emb4 is not None else None

        org1 = cx1
        org2 = cx2
        org3 = cx3
        # org4 = cx4
        x1 = self.ffn_norm1(cx1) if emb1 is not None else None
        x2 = self.ffn_norm2(cx2) if emb2 is not None else None
        x3 = self.ffn_norm3(cx3) if emb3 is not None else None
        # x4 = self.ffn_norm4(cx4) if emb4 is not None else None
        x1 = self.ffn1(x1) if emb1 is not None else None
        x2 = self.ffn2(x2) if emb2 is not None else None
        x3 = self.ffn3(x3) if emb3 is not None else None
        # x4 = self.ffn4(x4) if emb4 is not None else None
        x1 = x1 + org1 if emb1 is not None else None
        x2 = x2 + org2 if emb2 is not None else None
        x3 = x3 + org3 if emb3 is not None else None
        # x4 = x4 + org4 if emb4 is not None else None

        return x1, x2, x3, weights


class Encoder(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.encoder_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.encoder_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        # self.encoder_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')
        for _ in range(config.transformer["num_layers"]):
            layer = SCTB(config, vis, channel_num)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1, emb2, emb3):
        attn_weights = []
        for layer_block in self.layer:
            emb1, emb2, emb3, weights = layer_block(emb1, emb2, emb3)
            if self.vis:
                attn_weights.append(weights)
        emb1 = self.encoder_norm1(emb1) if emb1 is not None else None
        emb2 = self.encoder_norm2(emb2) if emb2 is not None else None
        emb3 = self.encoder_norm3(emb3) if emb3 is not None else None
        # emb4 = self.encoder_norm4(emb4) if emb4 is not None else None
        return emb1, emb2, emb3, attn_weights


class ChannelTransformer(nn.Module):
    def __init__(self, config, vis, img_size, channel_num, patchSize):
        super().__init__()

        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]
        self.patchSize_3 = patchSize[2]
        # self.patchSize_4 = patchSize[3]
        self.embeddings_1 = Channel_Embeddings(config, self.patchSize_1, img_size=img_size, in_channels=channel_num[0])
        self.embeddings_2 = Channel_Embeddings(config, self.patchSize_2, img_size=img_size // 2,
                                               in_channels=channel_num[1])
        self.embeddings_3 = Channel_Embeddings(config, self.patchSize_3, img_size=img_size // 4,
                                               in_channels=channel_num[2])
        # self.embeddings_4 = Channel_Embeddings(config, self.patchSize_4, img_size=img_size // 8,
        #                                        in_channels=channel_num[3])
        self.encoder = Encoder(config, vis, channel_num)

        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1,
                                         scale_factor=(self.patchSize_1, self.patchSize_1))
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1,
                                         scale_factor=(self.patchSize_2, self.patchSize_2))
        self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1,
                                         scale_factor=(self.patchSize_3, self.patchSize_3))
        # self.reconstruct_4 = Reconstruct(channel_num[3], channel_num[3], kernel_size=1,
        #                                  scale_factor=(self.patchSize_4, self.patchSize_4))

    def forward(self, en1, en2, en3):
        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)
        # emb4 = self.embeddings_4(en4)

        encoded1, encoded2, encoded3, attn_weights = self.encoder(emb1, emb2, emb3)  # (B, n_patch, hidden)

        x1 = self.reconstruct_1(encoded1) if en1 is not None else None
        x2 = self.reconstruct_2(encoded2) if en2 is not None else None
        x3 = self.reconstruct_3(encoded3) if en3 is not None else None
        # x4 = self.reconstruct_4(encoded4) if en4 is not None else None

        x1 = x1 + en1 if en1 is not None else None
        x2 = x2 + en2 if en2 is not None else None
        x3 = x3 + en3 if en3 is not None else None
        # x4 = x4 + en4 if en4 is not None else None

        return x1, x2, x3, attn_weights


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(CBN(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(CBN(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class CBN(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(CBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CCA(nn.Module):
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out


class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=in_channels // 2, F_x=in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class SCTransNet(nn.Module):
    def __init__(self, config, n_channels, n_classes=1, img_size=512, vis=False, mode='train', deepsuper=True):
        super().__init__()
        self.vis = vis
        self.deepsuper = deepsuper
        print('Deep-Supervision:', deepsuper)
        self.mode = mode
        in_channels = config.base_channel  # 32
        block = Res_CBAM_block
        self.pool = nn.MaxPool2d(2, 2)
        self.inc = self._make_layer(block, n_channels, in_channels)
        self.down_encoder1 = self._make_layer(block, in_channels, in_channels * 2, 1)  # 64  128
        self.down_encoder2 = self._make_layer(block, in_channels * 2, in_channels * 4, 1)  # 64  128
        self.down_encoder3 = self._make_layer(block, in_channels * 4, in_channels * 8, 1)  # 64  128
        # self.down_encoder4 = self._make_layer(block, in_channels * 8, in_channels * 8, 1)  # 64  128


        self.SCT = ChannelTransformer(config, vis, img_size,
                                       channel_num=[in_channels, in_channels * 2, in_channels * 4],
                                       patchSize=config.patch_sizes)
        self.neck = self._make_layer(block, in_channels * 8, in_channels * 8, 1)
        # CCA
        # self.up_decoder4 = UpBlock_attention(in_channels * 16, in_channels * 4, nb_Conv=2)
        # self.up3 = nn.ConvTranspose2d(in_channels * 8, in_channels * 4, kernel_size=4, stride=2, padding=1,
        #                                   bias=False)
        # self.up2 = nn.ConvTranspose2d(in_channels * 4, in_channels*2, kernel_size=4, stride=2, padding=1,
        #                                   bias=False)
        # self.up1 = nn.ConvTranspose2d(in_channels * 2, in_channels, kernel_size=4, stride=2, padding=1,
        #                                   bias=False)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.decoder3 = self._make_layer(block, in_channels * 12, in_channels * 4, 1)
        self.decoder2 = self._make_layer(block, in_channels * 6, in_channels * 2, 1)
        self.decoder1 = self._make_layer(block, in_channels * 3, in_channels, 1)
        # self.up_decoder3 = self._make_layer(block, in_channels * 12, in_channels * 4, 1)
        # self.up_decoder2 = self._make_layer(block, in_channels * 6, in_channels * 2, 1)
        # self.up_decoder1 = self._make_layer(block, in_channels * 3, in_channels * 1, 1)

        # self.outc = nn.Conv2d(in_channels, n_channels, kernel_size=(1, 1), stride=(1, 1))
        self.final = nn.Conv2d(n_channels, 1, kernel_size=(1, 1), stride=(1, 1))

        if self.deepsuper:
            # self.gt_conv5 = nn.Sequential(nn.Conv2d(in_channels * 8, 1, 1))
            # self.gt_conv4 = nn.Sequential(nn.Conv2d(in_channels * 4, n_channels, 1))
            # self.gt_conv3 = nn.Sequential(nn.Conv2d(in_channels * 2, n_channels, 1))
            # self.gt_conv2 = nn.Sequential(nn.Conv2d(in_channels * 1, n_channels, 1))
            self.outconv = Res_CBAM_block(15 * in_channels, n_channels)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.inc(x)  # 64 224 224
        x2 = self.down_encoder1(self.pool(x1))  # 128 112 112
        x3 = self.down_encoder2(self.pool(x2))  # 256 56  56
        d4 = self.down_encoder3(self.pool(x3))  # 512 28  28
        # d5 = self.down_encoder4(self.pool(x4))  # 512 14  14

        # Skip Connection
        f1 = x1
        f2 = x2
        f3 = x3
        # f4 = x4

        x1, x2, x3, att_weights = self.SCT(x1, x2, x3)

        x1 = x1 + f1
        x2 = x2 + f2
        x3 = x3 + f3
        # x4 = x4 + f4
        #
        #  Feature fusion
        # d4 = self.up_decoder4(d5, x4)

        d3 = self.decoder3(torch.cat([self.up(self.neck(d4)), x3], 1))
        d2 = self.decoder2(torch.cat([self.up(d3), x2], 1))
        d1 = self.decoder1(torch.cat([self.up(d2), x1], 1))
        # out = self.outc(d1)

        # Deep Supervision
        if self.deepsuper:
            # gt_5 = self.gt_conv5(d5)
            # gt_4 = self.gt_conv4(d4)
            # gt_3 = self.gt_conv3(d3)
            # gt_2 = self.gt_conv2(d2)

            # gt5 = F.interpolate(gt_5, scale_factor=16, mode='bilinear', align_corners=True)
            gt4 = self.up_8(d4)
            gt3 = self.up_4(d3)
            gt2 = self.up(d2)
            d0 = self.outconv(torch.cat((gt2, gt3, gt4, d1), 1))

            if self.mode == 'train':
                # return torch.sigmoid(d0)
                d0 = self.final(d0)
                return d0
            else:
                # return torch.sigmoid(out)
                out = self.final(d0)
                return out
        else:
            # return torch.sigmoid(out)
            out = self.final(d1)
            return out


if __name__ == '__main__':
    config_vit = get_CTranS_config()
    model = SCTransNet(config_vit, mode='train', deepsuper=True)
    # model = model.cuda()
    # inputs = torch.rand(2, 1, 256, 256).cuda()
    inputs = torch.rand(2, 1, 256, 256)
    output = model(inputs)