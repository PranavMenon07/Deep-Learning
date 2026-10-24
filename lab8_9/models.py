import os
import sys
import time
import json
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models, transforms

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ---------------------------------------------------------
# Set random seeds for reproducibility (Section 3.2, 34)
# ---------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

BASE_DIR = r"C:\Users\Prana\.gemini\antigravity-ide\scratch\lab8_9_attention_resnet50"
DATA_DIR = os.path.join(BASE_DIR, "dataset", "flower_photos")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------
# Attention Modules Definitions
# ---------------------------------------------------------

class ChannelAttention(nn.Module):
    """
    Channel Attention (Squeeze-and-Excitation style):
    z = GAP(F) in R^C
    Mc(F) = sigma(W2 * delta(W1 * z)) with reduction ratio r=16
    Fc = F * Mc(F)
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        reduced_channels = max(1, in_channels // reduction_ratio)
        self.fc1 = nn.Linear(in_channels, reduced_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduced_channels, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.last_weights = None

    def forward(self, x):
        b, c, h, w = x.size()
        z = x.view(b, c, -1).mean(dim=2)  # GAP: [B, C]
        w_att = self.sigmoid(self.fc2(self.relu(self.fc1(z))))  # [B, C]
        self.last_weights = w_att.detach().cpu()
        w_att = w_att.view(b, c, 1, 1)
        return x * w_att


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module:
    Favg = Mean_c(F)
    Fmax = Max_c(F)
    Fs = [Favg || Fmax] in R^(H x W x 2)
    Ms(F) = sigma(Conv_kxk(Fs))
    F's = F * Ms(F)
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 5, 7), "Kernel size must be 3, 5, or 7"
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.last_map = None

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        s_cat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        s_att = self.sigmoid(self.conv(s_cat))  # [B, 1, H, W]
        self.last_map = s_att.detach().cpu()
        return x * s_att


class CBAMAttention(nn.Module):
    """
    Sequential Channel-Spatial Attention (CBAM):
    F -> Channel Attention -> Spatial Attention -> Fcs
    """
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7, order='channel_first'):
        super().__init__()
        self.order = order
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        if self.order == 'channel_first':
            x = self.ca(x)
            x = self.sa(x)
        else:
            x = self.sa(x)
            x = self.ca(x)
        return x


class LightweightSelfAttention(nn.Module):
    """
    Self-Attention for Convolutional Feature Maps:
    Flatten spatial HW to N, project to Q, K, V
    A = softmax(Q K^T / sqrt(d))
    Y = A V
    F_SA = F + gamma * reshape(Y)
    """
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = max(1, in_channels // reduction_ratio)
        self.q_conv = nn.Conv2d(in_channels, self.inter_channels, 1, bias=False)
        self.k_conv = nn.Conv2d(in_channels, self.inter_channels, 1, bias=False)
        self.v_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.last_attn_matrix = None

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w
        q = self.q_conv(x).view(b, self.inter_channels, n).permute(0, 2, 1)  # [B, N, d]
        k = self.k_conv(x).view(b, self.inter_channels, n)                   # [B, d, N]
        scale = self.inter_channels ** 0.5
        energy = torch.bmm(q, k) / scale                                     # [B, N, N]
        attn = F.softmax(energy, dim=-1)                                     # [B, N, N]
        self.last_attn_matrix = attn.detach().cpu()
        v = self.v_conv(x).view(b, c, n)                                     # [B, C, N]
        out = torch.bmm(v, attn.permute(0, 2, 1))                            # [B, C, N]
        out = out.view(b, c, h, w)
        return x + self.gamma * out


class ECAModule(nn.Module):
    """
    Efficient Channel Attention (ECA-Net):
    Captures cross-channel interaction with adaptive 1D convolution
    No dimensionality reduction!
    """
    def __init__(self, in_channels, gamma=2, b=1):
        super().__init__()
        k = int(abs((np.log2(in_channels) / gamma) + (b / gamma)))
        kernel_size = k if k % 2 == 1 else k + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.last_weights = None

    def forward(self, x):
        b, c, h, w = x.size()
        z = x.view(b, c, -1).mean(dim=2)  # [B, C]
        y = self.conv(z.unsqueeze(1))     # [B, 1, C]
        att = self.sigmoid(y).squeeze(1)  # [B, C]
        self.last_weights = att.detach().cpu()
        return x * att.view(b, c, 1, 1)


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention for Efficient Feature Representation:
    Encodes spatial coordinate information into channel representations
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.pool_h(x)                                 # [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)             # [B, C, W, 1]
        y = torch.cat([x_h, x_w], dim=2)                     # [B, C, H+W, 1]
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return x * a_h * a_w

# ---------------------------------------------------------
# ResNet50 with Configurable Attention Modules and Positions
# ---------------------------------------------------------
class ResNet50WithAttention(nn.Module):
    def __init__(self, attention_type=None, position='P3', num_classes=5, dropout_rate=0.5,
                 spatial_kernel=7, attention_order='channel_first', multiple_positions=None):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)

        # ResNet50 sub-stages:
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1  # conv2_x: 256 ch, 56x56 (P1)
        self.layer2 = resnet.layer2  # conv3_x: 512 ch, 28x28 (P2)
        self.layer3 = resnet.layer3  # conv4_x: 1024 ch, 14x14 (P3)
        self.layer4 = resnet.layer4  # conv5_x: 2048 ch, 7x7 (P4)

        self.attention_type = attention_type
        self.position = position
        self.multiple_positions = multiple_positions

        stage_channels = {'P1': 256, 'P2': 512, 'P3': 1024, 'P4': 2048}

        def build_att(att_type, ch):
            if att_type is None or att_type == 'none':
                return nn.Identity()
            elif att_type == 'channel':
                return ChannelAttention(ch, reduction_ratio=16)
            elif att_type == 'spatial':
                return SpatialAttention(kernel_size=spatial_kernel)
            elif att_type == 'cbam' or att_type == 'channel_spatial':
                return CBAMAttention(ch, reduction_ratio=16, kernel_size=spatial_kernel, order=attention_order)
            elif att_type == 'self_attention':
                return LightweightSelfAttention(ch)
            elif att_type == 'combined':
                return nn.Sequential(
                    CBAMAttention(ch, reduction_ratio=16, kernel_size=spatial_kernel),
                    LightweightSelfAttention(ch)
                )
            elif att_type == 'eca':
                return ECAModule(ch)
            elif att_type == 'coordinate':
                return CoordinateAttention(ch)
            else:
                return nn.Identity()

        self.att_P1 = nn.Identity()
        self.att_P2 = nn.Identity()
        self.att_P3 = nn.Identity()
        self.att_P4 = nn.Identity()

        if multiple_positions:
            for p in multiple_positions:
                att_mod = build_att(attention_type, stage_channels[p])
                setattr(self, f"att_{p}", att_mod)
        elif attention_type and attention_type != 'none':
            att_mod = build_att(attention_type, stage_channels[position])
            setattr(self, f"att_{position}", att_mod)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(2048, num_classes)

        # Hook storage for Grad-CAM
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, capture_gradcam=False):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.att_P1(x)

        x = self.layer2(x)
        x = self.att_P2(x)

        x = self.layer3(x)
        x = self.att_P3(x)

        x = self.layer4(x)
        x = self.att_P4(x)

        if capture_gradcam:
            self.activations = x
            h = x.register_hook(self.activations_hook)

        x_pool = self.gap(x)
        x_flat = torch.flatten(x_pool, 1)
        x_drop = self.dropout(x_flat)
        out = self.classifier(x_drop)
        return out

    def get_last_conv_weights(self):
        return self.classifier.weight

print("Model architecture classes defined successfully.")
