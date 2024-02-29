import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.matching import (global_correlation_softmax, local_correlation_softmax)
from ..utils.attention import SelfAttnPropagation

class TextureNet(nn.Module):
    def __init__(self,
                 num_scales=2,
                 feature_channels=128,
                 upsample_factor=8,
                 reg_refine=False,  # optional local regression refinement
                 ):
        super(TextureNet, self).__init__()

        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.upsample_factor = upsample_factor
        self.reg_refine = reg_refine

        self.feature_attn = SelfAttnPropagation(in_channels=feature_channels)


    def forward(self, feature0, feature1,
                scale_idx,
                corr_radius_list=None,
                prop_radius_list=None,
                num_reg_refine=6,
                pred_bidir_flow=False,
                flow = None,
                ):

        flow_preds = []
        mlvl_flows = []

        feature0_ori, feature1_ori = feature0, feature1

        upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))

        corr_radius = corr_radius_list[scale_idx]
        prop_radius = prop_radius_list[scale_idx]


        # correlation and softmax
        if corr_radius == -1:  # global matching
            flow_pred = global_correlation_softmax(feature0, feature1, pred_bidir_flow)[0]
        else:  # local matching
            flow_pred = local_correlation_softmax(feature0, feature1, corr_radius)[0]


        if flow is not None:
            flow = flow + flow_pred
            flow_preds.append(flow)

        else:
            flow_preds.append(flow_pred)
            flow = self.feature_attn(feature0, flow_pred.detach(),
                                            local_window_attn=prop_radius > 0,
                                            local_window_radius=prop_radius,
                                            )
            flow_preds.append(flow)

        return flow, flow_preds

