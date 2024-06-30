import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from .scale_encoder import ScaleEncoder
from .decoder import ScaleDecoder
from ..modules.attention import SelfAttnPropagation
from ..modules.geometry import flow_wrap

from .feature_net.feature_net import GmaAtten, FeatureNet, ScaleFormer

from .regrefine import BasicUpdateBlock
from ..modules.utils import upsample_scale_with_mask, upsample_flow_with_mask
from ..modules.matching import local_correlation_with_flow

class ScaleNet(nn.Module):
    def __init__(self,
                 num_scales=2,
                 feature_channels=128,
                 upsample_factor=4,
                 num_head=1,
                 num_transformer_layers=6,
                 scale_level=1,
                 reg_refine=False,  # optional local regression refinement
                 query_lvl = -1
                 ):
        super(ScaleNet, self).__init__()

        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.upsample_factor = upsample_factor
        self.reg_refine = reg_refine

        # the level of features, flows while generating query
        self.query_lvl = query_lvl

        # Transformer
        self.encoder = ScaleEncoder(num_layers=num_transformer_layers,
                                              d_model=feature_channels,
                                              nhead=num_head,
                                              num_feature_levels=num_scales,
                                              num_level=scale_level
                                              )
        self.scale_update = ScaleFormer(num_scales=1, feature_channels=feature_channels, 
                                        num_head=1, ffn_dim_expansion=2, 
                                        num_transformer_layers=1)
        self.flow_encoder = CorrEncoder(dim_in=2, dim_out=feature_channels)
        
        # propagation with self-attn
        self.decoder = ScaleDecoder(upsample_factor=upsample_factor)

        self.gma = GmaAtten(num_scales=1, feature_channels=feature_channels, 
                                  num_head=1, ffn_dim_expansion=2, 
                                  num_transformer_layers=1)
        self.num_reg_refine = 1
        self.refine = BasicUpdateBlock(corr_channels=(2 * 4 + 1) ** 2,
                                        downsample_factor=upsample_factor,
                                        flow_dim=2,
                                        bilinear_up=False,
                                        )


    def forward(self, feature0_listc, feature1_listc, 
                    ini_flow, ini_flow0
                ):

        cfeat0 = feature0_listc[-1]

        flow_list = []
        scale_list = []

        ini_scale = torch.ones_like(ini_flow[:,0:1,...])
        scale, flow = ini_scale, ini_flow
        #scale_list.append(F.interpolate(scale, scale_factor=4, mode='bilinear', align_corners=True))
        dis_f = (ini_flow-ini_flow0).detach()
        dis_s = ini_scale.detach()

        for i in range(self.num_reg_refine):
            scale = scale.detach()
            flow = flow.detach()

            for j in range(len(feature1_listc)):
                wrap_flow = F.interpolate(dis_f, scale_factor=0.5, mode='bilinear', align_corners=True) * 0.5 \
                            if j==0 else dis_f
                feature1_listc[j] = flow_wrap(feature1_listc[j], wrap_flow)
                scale_u = F.interpolate(dis_s, scale_factor=0.5, mode='bilinear', align_corners=True) \
                                            if j==0 else dis_s
                feature1_listc[j] = self.scale_update(scale_u, feature1_listc[j], 'swin', [1,4] if j==0 else [2,8])

            cfeat1 = feature1_listc[-1]

            flow_feature = self.flow_encoder(flow)

            t1 = time.time()
            scale_feature = self.encoder(feature0_listc, feature1_listc, \
                                        self.query_lvl, ini_query=cfeat0)     #(bs, c, h, w)
            t2 = time.time()
            agg_corr = self.gma(scale_feature, flow_feature, 'swin', [2,8])
            t3 = time.time()
            scale_out, flow_out, mask_s, mask_f, proj = self.decoder(scale_feature, \
                                                        flow_feature, cfeat0, agg_corr, scale.clone(), flow.clone())
            t4 = time.time()

            # print(t4 - t3, t3 - t2, t2 - t1, 'yyyyyyy')

            scale = scale+scale_out
            flow = flow+flow_out

            scale_up = upsample_scale_with_mask(scale, mask_s, upsample_factor=self.upsample_factor)
            flow_up = upsample_flow_with_mask(flow, mask_f, upsample_factor=self.upsample_factor)

            flow_list.append(flow_up)
            scale_list.append(scale_up)

            flow = flow.detach()
            correlation = local_correlation_with_flow(
                cfeat0,
                cfeat1,
                flow=flow_out.detach(),
                local_radius=4,
            )  # [B, (2R+1)^2, H, W]
            net, inp = torch.chunk(proj, chunks=2, dim=1)

            net = torch.tanh(net)
            inp = torch.relu(inp)

            net, up_mask, residual_flow = self.refine(net, inp, correlation, flow.clone(),
                                                        )
            flow = flow + residual_flow
            flow_up = upsample_flow_with_mask(flow, up_mask, upsample_factor=self.upsample_factor)
            flow_list.append(flow_up)
            
            dis_f = (flow_out + residual_flow).detach()
            dis_s = scale_out.detach()

        return scale_list, flow_list, flow_feature.detach(), scale_feature.detach()


class ScaleHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1):
        super(ScaleHead, self).__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, self.output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.output_dim==1:
            return F.relu(self.conv2(self.relu(self.conv1(x))))
        else:
            return self.conv2(self.relu(self.conv1(x)))


class UpdateFeatureBlock(nn.Module):
    def __init__(self, dim1, dim2):
        super(UpdateFeatureBlock, self).__init__()
        self.convc1 = nn.Conv2d(dim1, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(dim2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-dim2, 3, padding=1)
    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class CorrEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(CorrEncoder, self).__init__()
        self.convc1 = nn.Conv2d(dim_in, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, dim_out, 3, padding=1)

    def forward(self, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        return cor