import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.utils import normalize_img
from .layers.matching import (global_correlation_softmax, local_correlation_softmax, \
                                local_correlation_with_flow, local_scale_correlation)

from .network.backbone import CNNEncoder
from .network.feature_net.feature_net import FeatureNet
from .network.flow_net import FlowNet
from .network.disp_net import DispNet
from .network.scale_net import ScaleNet

from .layers.geometry import flow_wrap

class MonoFusion(nn.Module):
    def __init__(self,
                 num_scales=2,
                 feature_channels=128,
                 upsample_factor=8,
                 num_head=1,
                 ffn_dim_expansion=4,
                 num_transformer_layers=6,
                 reg_refine=False,  # optional local regression refinement
                 train=False,
                 local_radius=3,
                 num_iterations=1,  # texture-scale fusion recurrent iterations
                 ):
        super(MonoFusion, self).__init__()

        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.upsample_factor = upsample_factor
        self.reg_refine = reg_refine
        self.is_trainning = train
        self.local_radius = local_radius
        self.num_iterations = num_iterations
        # CNN
        #norm_layer=nn.BatchNorm2d
        #self.backbone = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales, norm_layer=nn.BatchNorm2d)
        self.cnet = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales)

        self.featnet = FeatureNet(num_scales=num_scales, feature_channels=feature_channels, 
                                  num_head=num_head, ffn_dim_expansion=ffn_dim_expansion, 
                                  num_transformer_layers=num_transformer_layers)
        self.corrnet = FlowNet(num_scales=num_scales, feature_channels=feature_channels,
                               upsample_factor=upsample_factor, reg_refine=reg_refine)
        self.dispnet = DispNet(num_scales=num_scales, feature_channels=feature_channels,
                               upsample_factor=upsample_factor, reg_refine=reg_refine)
        # self.ini_scalenet = DispNet(num_scales=num_scales, feature_channels=feature_channels,
        #                        upsample_factor=upsample_factor, reg_refine=reg_refine)
        self.scalenet = ScaleNet(num_scales=num_scales, feature_channels=feature_channels,
                                 upsample_factor=upsample_factor, num_head=num_head,
                                 scale_level=num_scales, reg_refine=reg_refine,
                                 num_iterations=num_iterations)


    def forward(self, img0, img1,
                attn_type=None,
                attn_splits_list=None,
                corr_radius_list=None,
                prop_radius_list=None,
                num_reg_refine=6,
                pred_bidir_flow=False,
                testing=False,
                ):

        scale, corr, disp = None, None, None
        mlvl_feats0, mlvl_feats1, mlvl_flows, mlvl_flows_back = [], [], [], []

        start = time.time()

        flows = []     
        disps, disps2 = [], []
        scales = [] 

        img0, img1 = normalize_img(img0, img1)
        #with torch.no_grad():
        t0 = time.time()
        feature0_listc, feature1_listc = self.extract_feature(img0, img1)
        t1 = time.time()
        mlvl_feats0, mlvl_feats1 = [],[]

            
        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_listc[scale_idx], feature1_listc[scale_idx]
            if scale_idx < 1:
                feature0, feature1 = self.featnet(feature0, feature1, scale_idx, attn_type, attn_splits_list, corr)
                corr, final = self.corrnet(feature0, feature1, scale_idx, corr_radius_list,
                                    prop_radius_list, num_reg_refine, False, corr)

                mlvl_feats0.append(feature0)
                mlvl_feats1.append(flow_wrap(feature1,corr))

                disp = self.dispnet(feature0, scale_idx, prop_radius_list)
                disp = F.interpolate(disp, scale_factor=2, mode='bilinear', align_corners=True)
                disps.append(F.interpolate(disp, scale_factor=4, mode='bilinear', align_corners=True))
                disp = disp.detach()

                disp2 = self.dispnet(flow_wrap(feature1,corr), scale_idx, prop_radius_list)
                disp2 = F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=True)
                disps2.append(F.interpolate(disp2, scale_factor=4, mode='bilinear', align_corners=True))
                disp2 = disp2.detach()

                corr = F.interpolate(corr, scale_factor=2, mode='bilinear', align_corners=True) * 2
                ini_flow0 = corr
                for i in range(len(final)):
                    flows.append(F.interpolate(final[i], scale_factor=8, mode='bilinear', align_corners=True) * 8)

            else:
                feature0_f, feature1_f = self.featnet(feature0, feature1, scale_idx, attn_type, attn_splits_list, corr)
                mlvl_feats0.append(feature0_f)
                mlvl_feats1.append(feature1_f)
                corr, final = self.corrnet(feature0_f, feature1_f, scale_idx, corr_radius_list,
                                    prop_radius_list, num_reg_refine, False, corr)

                disp = self.dispnet(feature0_f, scale_idx, prop_radius_list, disp)
                disps.append(F.interpolate(disp, scale_factor=4, mode='bilinear', align_corners=True))
                disp = disp.detach()

                disp2 = self.dispnet(feature1_f, scale_idx, prop_radius_list, disp2)
                disps2.append(F.interpolate(disp2, scale_factor=4, mode='bilinear', align_corners=True))
                disp2 = disp2.detach()

                for i in range(len(final)):
                    flows.append(F.interpolate(final[i], scale_factor=4, mode='bilinear', align_corners=True) * 4)

        t2 = time.time()

        corr = corr.detach()
        ini_flow = corr
        scale, flow, featf, feats = self.scalenet(mlvl_feats0, mlvl_feats1, ini_flow, ini_flow0)

        t3 = time.time()


        for i in range(len(flow)):
            flows.append(flow[i])
        # for i in range(len(disp)):
        #     disps.append(disp[i])
        #     scales.append(scale[i])

        featf = F.interpolate(featf, scale_factor=4, mode='bilinear', align_corners=True)
        feats = F.interpolate(feats, scale_factor=4, mode='bilinear', align_corners=True)
        return scale, disps, disps2, flows, None, featf, feats
        # return scale, flows, None

    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]
        features = self.cnet(concat)  # list of [2B, C, H, W], resolution from high to low

        # reverse: resolution from low to high
        features = features[::-1]

        feature0, feature1 = [], []

        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 2, 0)  # tuple
            feature0.append(chunks[0])
            feature1.append(chunks[1])

        return feature0, feature1





class CorrEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(CorrEncoder, self).__init__()
        self.convc1 = nn.Conv2d(dim_in, 256, 3, padding=1)
        self.convc2 = nn.Conv2d(256, dim_out, 3, padding=1)

    def forward(self, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        return cor
