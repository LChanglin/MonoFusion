import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.utils import normalize_img
from .modules.geometry import flow_wrap

from .modules.backbone import CNNEncoder
from .modules.feature_net.feature_net import FeatureNet
from .modules.texture_net import TextureNet
from .modules.scale_net import ScaleNet



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
                 local_radius=3
                 ):
        super(MonoFusion, self).__init__()

        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.upsample_factor = upsample_factor
        self.reg_refine = reg_refine
        self.is_trainning = train
        self.local_radius = local_radius
        # CNN
        #norm_layer=nn.BatchNorm2d
        #self.backbone = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales, norm_layer=nn.BatchNorm2d)
        self.cnet = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales)

        self.featnet = FeatureNet(num_scales=num_scales, feature_channels=feature_channels, 
                                  num_head=num_head, ffn_dim_expansion=ffn_dim_expansion, 
                                  num_transformer_layers=num_transformer_layers)
        self.corrnet = TextureNet(num_scales=num_scales, feature_channels=feature_channels,
                               upsample_factor=upsample_factor, reg_refine=reg_refine)
        self.scalenet = ScaleNet(num_scales=num_scales, feature_channels=feature_channels,
                                 upsample_factor=upsample_factor, num_head=num_head,
                                 scale_level=num_scales, reg_refine=reg_refine)

    def forward(self, img0, img1,
                attn_type=None,
                attn_splits_list=None,
                corr_radius_list=None,
                prop_radius_list=None,
                num_reg_refine=6,
                testing=False,
                ):

        scale, corr = None, None
        mlvl_feats0, mlvl_feats1 = [], []

        start = time.time()

        flows = []       

        img0, img1 = normalize_img(img0, img1)
        with torch.no_grad():
            feature0_listc, feature1_listc = self.extract_feature(img0, img1)
        mlvl_feats0, mlvl_feats1 = [],[]

            
        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_listc[scale_idx], feature1_listc[scale_idx]
            if scale_idx < 1:
                feature0, feature1 = self.featnet(feature0, feature1, scale_idx, attn_type, attn_splits_list, corr)
                corr, final = self.corrnet(feature0, feature1, scale_idx, corr_radius_list,
                                    prop_radius_list, num_reg_refine, False, corr)
                mlvl_feats0.append(feature0)
                mlvl_feats1.append(flow_wrap(feature1,corr))
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
                for i in range(len(final)):
                    # print(final[i].shape)
                    flows.append(F.interpolate(final[i], scale_factor=4, mode='bilinear', align_corners=True) * 4)

        corr = corr.detach()
        ini_flow = corr
        scale, flow = self.scalenet(mlvl_feats0, mlvl_feats1, ini_flow, ini_flow0)
        for i in range(len(flow)):
            flows.append(flow[i])
        
        return scale, flows, None

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
        self.convc1 = nn.Conv2d(dim_in, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, dim_out, 3, padding=1)

    def forward(self, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        return cor
