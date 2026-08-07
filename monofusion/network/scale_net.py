import torch
import torch.nn as nn
import torch.nn.functional as F

from .scale_encoder import ScaleEncoder
from .decoder import ScaleDecoder
from ..layers.attention import SelfAttnPropagation
from ..layers.geometry import flow_wrap

from .feature_net.feature_net import GmaAtten, FeatureNet, ScaleFormer

from .update_block import BasicUpdateBlock
from ..layers.utils import upsample_scale_with_mask, upsample_flow_with_mask
from ..layers.matching import local_correlation_with_flow

class ScaleNet(nn.Module):
    def __init__(self,
                 num_scales=2,
                 feature_channels=128,
                 upsample_factor=4,
                 num_head=1,
                 num_transformer_layers=6,
                 scale_level=1,
                 reg_refine=False,  # optional local regression refinement
                 query_lvl = -1,
                 num_iterations=3,  # texture-scale fusion recurrent iterations:
                                     # 1 -> MonoFusion (single fusion-decode pass)
                                     # >1 -> MonoFusion-R (recurrent refinement, all
                                     #       iterations share the same weights)
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

        self.scale_to_depth = ScaleConverter(dim1=feature_channels, dim2=1, out_dim=1)
        
        # propagation with self-attn
        self.decoder = ScaleDecoder(upsample_factor=upsample_factor)

        self.gma = GmaAtten(num_scales=1, feature_channels=feature_channels, 
                                  num_head=1, ffn_dim_expansion=2, 
                                  num_transformer_layers=1)
        self.num_iterations = num_iterations
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
        disp_list = []

        ini_scale = torch.ones_like(ini_flow)
        ini_disp = ini_scale[:,0:1,...]
        ini_scale = ini_scale[:,1:,...]

        dis_f = (ini_flow-ini_flow0).detach()
        dis_s = ini_scale.detach()
        dis_d = ini_disp.detach()

        scale, flow = ini_scale, ini_flow
        normalized_disp = ini_disp
        disp = self.scale_to_depth(cfeat0, ini_disp)

        for i in range(self.num_iterations):
            scale = scale.detach()
            flow = flow.detach()

            for j in range(len(feature1_listc)):
                wrap_flow = F.interpolate(dis_f, scale_factor=0.5, mode='bilinear', align_corners=True) * 0.5 \
                            if j==0 else dis_f
                feature1_listc[j] = flow_wrap(feature1_listc[j], wrap_flow)
                scale_u = F.interpolate(dis_s, scale_factor=0.5, mode='bilinear', align_corners=True) \
                                            if j==0 else dis_s
                dc_u = F.interpolate(dis_d, scale_factor=0.5, mode='bilinear', align_corners=True) \
                                            if j==0 else dis_d
                
                warp_ds = torch.cat([dc_u, scale_u], dim=1)
                feature1_listc[j] = self.scale_update(warp_ds, feature1_listc[j], 'swin', [1,4] if j==0 else [2,8])

            cfeat1 = feature1_listc[-1]

            flow_feature = self.flow_encoder(flow)
            scale_feature = self.encoder(feature0_listc, feature1_listc, \
                                        self.query_lvl, ini_query=cfeat0)     #(bs, c, h, w)
            agg_corr = self.gma(scale_feature, flow_feature, 'swin', [2,8])
            disp_out, scale_out, flow_out, \
            mask_d, mask_s, mask_f, proj = self.decoder(scale_feature, \
                                                        flow_feature, cfeat0, agg_corr, \
                                                        normalized_disp.clone(), scale.clone(), flow.clone())

            dis_s = scale_out.detach()
            dis_d = disp_out.detach()

            normalized_disp = normalized_disp + disp_out
            disp_out = self.scale_to_depth(cfeat0, disp_out)
            disp = disp + disp_out
            scale = scale + scale_out
            flow = flow + flow_out


            scale_up = upsample_scale_with_mask(scale, mask_s, upsample_factor=self.upsample_factor)
            disp_up = upsample_scale_with_mask(disp, mask_d, upsample_factor=self.upsample_factor)
            flow_up = upsample_flow_with_mask(flow, mask_f, upsample_factor=self.upsample_factor)

            flow_list.append(flow_up)
            scale_list.append(scale_up)
            disp_list.append(disp_up)

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

        return scale_list, disp_list, flow_list


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


class ScaleConverter(nn.Module):
    def __init__(self, dim1, dim2, out_dim=128):
        super(ScaleConverter, self).__init__()
        self.convc1 = nn.Conv2d(dim1, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(dim2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, out_dim, 3, padding=1)
    def forward(self, img_feat, scale):
        cor = F.relu(self.convc1(img_feat))
        cor = F.relu(self.convc2(cor))
        disp = F.relu(self.convf1(scale))
        disp = F.relu(self.convf2(disp))

        cor_disp = torch.cat([cor, disp], dim=1)
        d = F.relu(self.conv(cor_disp))
        return d

class CorrEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(CorrEncoder, self).__init__()
        self.convc1 = nn.Conv2d(dim_in, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, dim_out, 3, padding=1)

    def forward(self, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        return cor