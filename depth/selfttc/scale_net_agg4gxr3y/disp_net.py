import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.matching import (scale_correlation_softmax, local_scale_correlation)
from ..modules.attention import SelfAttnPropagation
from ..modules.utils import upsample_flow_with_mask
from ..modules.reg_refine import BasicUpdateBlock


class DispNet(nn.Module):
    def __init__(self,
                 num_scales=2,
                 feature_channels=128,
                 upsample_factor=8,
                 reg_refine=False,  # optional local regression refinement
                 ):
        super(DispNet, self).__init__()

        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.upsample_factor = upsample_factor
        self.reg_refine = reg_refine

        self.disp_encoder = CorrEncoder(dim_in=feature_channels, dim_out=feature_channels)
        self.ini_cor_encoder = DispHead(input_dim=128, hidden_dim=256, output_dim=1)
        self.feature_flow_attn = SelfAttnPropagation(in_channels=feature_channels)

    def forward(self, feature0, scale_idx, prop_radius_list=None, disp0=None):

        b, c, h, w = feature0.size()
        prop_radius = prop_radius_list[scale_idx]
        feature0_d = self.disp_encoder(feature0)
        ini_disp = self.ini_cor_encoder(feature0_d)
        if disp0 is not None:
            ini_disp = disp0 + ini_disp
        disp = self.feature_flow_attn(feature0_d, ini_disp,
                                        local_window_attn=prop_radius > 0,
                                        local_window_radius=prop_radius,
                                        )
        return disp


class CorrEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(CorrEncoder, self).__init__()
        self.convc1 = nn.Conv2d(dim_in, 256, 3, padding=1)
        self.convc2 = nn.Conv2d(256, dim_out, 3, padding=1)

    def forward(self, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        return cor

class DispHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1):
        super(DispHead, self).__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, self.output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))