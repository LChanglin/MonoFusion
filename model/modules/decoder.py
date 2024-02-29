import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.utils import upsample_scale_with_mask, upsample_flow_with_mask
from ..modules.attention import SelfAttnPropagation
from ..modules.matching import local_correlation_with_flow
from .regrefine import BasicUpdateBlock

class FeatureFusion(nn.Module):
    def __init__(self, dim):
        super(FeatureFusion, self).__init__()
        self.convc1 = nn.Conv2d(dim, 128, 3, padding=1)
        self.convf1 = nn.Conv2d(dim, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+128, dim, 3, padding=1)
    def forward(self, net, feat):
        cor = F.relu(self.convc1(feat))
        flo = F.relu(self.convf1(net))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return out

class ScaleHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1):
        super(ScaleHead, self).__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, self.output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))
        # return self.conv2(self.relu(self.conv1(x)))

class BasicMotionEncoder(nn.Module):
    def __init__(self, dim1, dim2):
        super(BasicMotionEncoder, self).__init__()
        self.convc1 = nn.Conv2d(dim1, 192, 1, padding=0)
        self.convc2 = nn.Conv2d(192, 128, 3, padding=1)
        self.convf1 = nn.Conv2d(dim2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+128, 128-dim2, 3, padding=1)
    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h



class BasicUpdateBlockX(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128, scale=True, mask=False):
        super(BasicUpdateBlockX, self).__init__()
        self.scale = scale
        if self.scale:
            self.encoder = BasicMotionEncoder(dim1=input_dim, dim2=1)
            self.scale_head = ScaleHead(hidden_dim, hidden_dim=256)
        else:
            self.encoder = BasicMotionEncoder(dim1=input_dim, dim2=2)
            self.scale_head = ScaleHead(hidden_dim, hidden_dim=256, output_dim=2)

        self.fusion = FeatureFusion(dim=input_dim)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.mask = mask
        if self.mask:
            self.mblock = nn.Sequential(nn.Conv2d(hidden_dim, 256, 3, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256, 4 ** 2 * 9, 1, padding=0))

    def forward(self, net, inp, feat, agg_corr, scale):
        motion_features = self.encoder(scale,agg_corr)
        motion_features = self.fusion(feat, motion_features)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        scale = self.scale_head(net)

        if self.mask:
            mask = self.mblock(net)
        else:
            mask = None

        return net, scale, mask


class ScaleDecoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=96,
                 out_dim=1, upsample_factor=4, num_blocks=2,
                 ):
        super(ScaleDecoder, self).__init__()

        self.upsample_factor = upsample_factor
        self.num_blocks = num_blocks

        self.refine_projx = nn.Conv2d(128*3, 256, 1)
        if self.num_blocks>1:
            self.net_proj = nn.Conv2d(128*2, 128, 1)

        self.scale_estimator = nn.ModuleList()
        self.flow_estimator = nn.ModuleList()
        for l in range(num_blocks):
            layer = BasicUpdateBlockX(input_dim, input_dim, mask=l>0)     
            self.scale_estimator.append(layer)
        for l in range(num_blocks):
            layer = BasicUpdateBlockX(input_dim, input_dim, scale=False, mask=l>0)     
            self.flow_estimator.append(layer)

        self.feature_flow_attn = SelfAttnPropagation(in_channels=input_dim)


    def forward(self, scale_feature, flow_feature, cfeat0, agg_corr, ini_scale, ini_flow, id=0):

        proj = self.refine_projx(torch.cat([scale_feature, flow_feature, cfeat0], dim=1))
        net, inp = torch.chunk(proj, chunks=2, dim=1)

        net = torch.tanh(net)
        inp = torch.relu(inp)

        scale = ini_scale
        flow = ini_flow

        for l in range(self.num_blocks):
            net_s, scale_out, mask_s = self.scale_estimator[l](net, inp, scale_feature, agg_corr, scale)
            net_t, flow_out, mask_f = self.flow_estimator[l](net, inp, flow_feature, agg_corr, flow)
            if l==0:
                scale = scale_out
                flow = flow_out
            else:
                scale = scale * scale_out  
                flow = flow + flow_out
                break
            net = self.net_proj(torch.cat([net_s, net_t], dim=1))

        result = self.feature_flow_attn(cfeat0, torch.cat([scale, flow], dim=1),
                                    local_window_attn=True,
                                    local_window_radius=1,
                                    )
        scale, flow = result[:,0:1,...], result[:,1:,...]

        return scale, flow, mask_s, mask_f, proj



    




# class ScaleDecoder(nn.Module):
#     def __init__(self, input_dim=128, hidden_dim=96,
#                  out_dim=1, upsample_factor=4, num_blocks=2,
#                  ):
#         super(ScaleDecoder, self).__init__()

#         self.upsample_factor = upsample_factor
        
#         # self.conv1 = nn.Conv2d(input_dim*2, hidden_dim*2, 3, padding=1)
#         self.conv1 = nn.Sequential(
#             conv(input_dim*2, input_dim*3, padding_mode="zeros"),
#             conv(input_dim*3, hidden_dim*3, padding_mode="zeros"),
#             conv(hidden_dim*3, hidden_dim*2, padding_mode="zeros")
#         )
#         # self.conv2 = nn.Conv2d(input_dim*2, input_dim, 3, padding=1)
#         self.scale_proj = conv(input_dim+1, input_dim)
#         self.pre_proj = conv(hidden_dim*2+1, hidden_dim*2)
#         # self.conv2 = nn.Conv2d(input_dim, input_dim, 3, padding=1)


#         self.scale_estimators = nn.ModuleList()
#         self.num_blocks = num_blocks
#         for l in range(num_blocks):
#             layer_sf = ScaleDecoderBlock_LSTM(input_dim)            
#             self.scale_estimators.append(layer_sf)

#         #self.feature_scale_attn = SelfAttnPropagation(in_channels=input_dim)

#         self.relu = nn.ReLU(inplace=True)
#         self.upsampler = nn.Sequential(nn.Conv2d(1 + input_dim, 256, 3, 1, 1),
#                                         nn.ReLU(inplace=True),
#                                         nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))
        
#     '''
#     x: (bs, c, H,W)
#     x_pre: (bs, c, H,W)
#     '''
#     def forward(self, x, feature0, feature1):
#         #print(x_pre.shape, feature0.shape)
#         proj_pre = self.conv1(torch.cat([feature0, feature1], dim=1)) #(bs, H,W, c*2)
#         h_pre, c_pre = torch.chunk(proj_pre, chunks=2, dim=1)

#         scale = None

#         for l in range(self.num_blocks):
#             scale_out, h_pre, c_pre = self.scale_estimators[l](x, h_pre, c_pre)
#             if l==0:
#                 scale = scale_out
#             else:
#                 scale *= scale_out   
#             if l==(self.num_blocks-1):
#                 break
#             x = self.scale_proj(torch.cat([x,scale],dim=1))
#             pre = self.pre_proj(torch.cat([h_pre, c_pre, scale],dim=1))
#             h_pre, c_pre = torch.chunk(pre, chunks=2, dim=1)
         
#         scale_f = self.upsample_scale(scale, feature0)

#         return scale_f
    

#     def upsample_scale(self, scale, feature, bilinear=False, upsample_factor=8,
#                       is_depth=False):
#         if bilinear:
#             multiplier = 1 if is_depth else upsample_factor
#             up_scale = F.interpolate(scale, scale_factor=upsample_factor,
#                                     mode='bilinear', align_corners=True) * multiplier
#         else:
#             concat = torch.cat((scale, feature), dim=1)
#             mask = self.upsampler(concat)
#             up_scale = upsample_scale_with_mask(scale, mask, upsample_factor=self.upsample_factor,
#                                               is_depth=is_depth)
#         return up_scale



# def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True, padding_mode="zeros"):
#     if isReLU:
#         return nn.Sequential(
#             nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
#                       padding=((kernel_size - 1) * dilation) // 2, bias=True, padding_mode=padding_mode),
#             nn.LeakyReLU(0.1, inplace=False)
#         )
#     else:
#         return nn.Sequential(
#             nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
#                       padding=((kernel_size - 1) * dilation) // 2, bias=True, padding_mode=padding_mode)
#         )


# class ScaleDecoderBlock_LSTM(nn.Module):
#     """
#     The split decoder model with LSTM
#     """
#     def __init__(self, ch_in, padding_mode="zeros"):
#         super(ScaleDecoderBlock_LSTM, self).__init__()

#         self.input_dim = 128
#         self.hidden_dim = 96

#         self.convs = nn.Sequential(
#             conv(ch_in, 128, padding_mode=padding_mode),
#             conv(128, 128, padding_mode=padding_mode),
#             conv(128, self.hidden_dim, padding_mode=padding_mode)
#         )
#         self.conv_scale = nn.Sequential(
#             conv(96*3, 64*2, padding_mode=padding_mode),
#             conv(64*2, 32, padding_mode=padding_mode),
#             conv(32, 1, isReLU=False, padding_mode=padding_mode)
#             )

#         ## LSTM Cell
#         self.conv_lstm = conv(self.hidden_dim*2, 4 * self.hidden_dim, isReLU=False, padding_mode=padding_mode)
#         self.cell_state = None

#     def forward_lstm(self, input_tensor, h_cur, c_cur):

#         combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

#         combined_conv = self.conv_lstm(combined)
#         cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
#         i = torch.sigmoid(cc_i)
#         f = torch.sigmoid(cc_f)
#         o = torch.sigmoid(cc_o)
#         g = nn.LeakyReLU(0.1, inplace=False)(cc_g)

#         c_next = f * c_cur + i * g
#         h_next = o * nn.LeakyReLU(0.1, inplace=False)(c_next)

#         return h_next, c_next


#     def forward(self, x, h_pre, c_pre):
        
#         x_curr = self.convs(x)

#         h_next, c_next = self.forward_lstm(x_curr, h_pre, c_pre)

#         scale = self.conv_scale(torch.cat([x_curr, h_next, c_next], dim=1))

#         return scale, h_next, c_next