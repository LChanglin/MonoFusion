import torch
import torch.nn as nn

from ...modules.attention import (single_head_full_attention, single_head_split_window_attention,
                        single_head_full_attention_1d, single_head_split_window_attention_1d)
from ...modules.utils import generate_shift_window_attn_mask, generate_shift_window_attn_mask_1d


class TransformerLayer(nn.Module):
    def __init__(self,
                 d_model=128,
                 nhead=1,
                 no_ffn=False,
                 ffn_dim_expansion=4,
                 ):
        super(TransformerLayer, self).__init__()

        self.dim = d_model
        self.nhead = nhead
        self.no_ffn = no_ffn

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.merge = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)

        # no ffn after self-attn, with ffn after cross-attn
        if not self.no_ffn:
            in_channels = d_model * 2
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                nn.GELU(),
                nn.Linear(in_channels * ffn_dim_expansion, d_model, bias=False),
            )

            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, source, target,
                height=None,
                width=None,
                shifted_window_attn_mask=None,
                shifted_window_attn_mask_1d=None,
                attn_type='swin',
                with_shift=False,
                attn_num_splits=None,
                ):
        # source, target: [B, L, C]
        query, key, value = source, target, source

        # for stereo: 2d attn in self-attn, 1d attn in cross-attn
        is_self_attn = (query - key).abs().max() < 1e-6

        # single-head attention
        query = self.q_proj(query)  # [B, L, C]
        key = self.k_proj(key)  # [B, L, C]
        value = self.v_proj(value)  # [B, L, C]

        if attn_type == 'swin' and attn_num_splits > 1:  # self, cross-attn: both swin 2d
            if self.nhead > 1:
                # we observe that multihead attention slows down the speed and increases the memory consumption
                # without bringing obvious performance gains and thus the implementation is removed
                raise NotImplementedError
            else:
                message = single_head_split_window_attention(query, key, value,
                                                             num_splits=attn_num_splits,
                                                             with_shift=with_shift,
                                                             h=height,
                                                             w=width,
                                                             attn_mask=shifted_window_attn_mask,
                                                             )

        elif attn_type == 'self_swin2d_cross_1d':  # self-attn: swin 2d, cross-attn: full 1d
            if self.nhead > 1:
                raise NotImplementedError
            else:
                if is_self_attn:
                    if attn_num_splits > 1:
                        message = single_head_split_window_attention(query, key, value,
                                                                     num_splits=attn_num_splits,
                                                                     with_shift=with_shift,
                                                                     h=height,
                                                                     w=width,
                                                                     attn_mask=shifted_window_attn_mask,
                                                                     )
                    else:
                        # full 2d attn
                        message = single_head_full_attention(query, key, value)  # [N, L, C]

                else:
                    # cross attn 1d
                    message = single_head_full_attention_1d(query, key, value,
                                                            h=height,
                                                            w=width,
                                                            )

        elif attn_type == 'self_swin2d_cross_swin1d':  # self-attn: swin 2d, cross-attn: swin 1d
            if self.nhead > 1:
                raise NotImplementedError
            else:
                if is_self_attn:
                    if attn_num_splits > 1:
                        # self attn shift window
                        message = single_head_split_window_attention(query, key, value,
                                                                     num_splits=attn_num_splits,
                                                                     with_shift=with_shift,
                                                                     h=height,
                                                                     w=width,
                                                                     attn_mask=shifted_window_attn_mask,
                                                                     )
                    else:
                        # full 2d attn
                        message = single_head_full_attention(query, key, value)  # [N, L, C]
                else:
                    if attn_num_splits > 1:
                        assert shifted_window_attn_mask_1d is not None
                        # cross attn 1d shift
                        message = single_head_split_window_attention_1d(query, key, value,
                                                                        num_splits=attn_num_splits,
                                                                        with_shift=with_shift,
                                                                        h=height,
                                                                        w=width,
                                                                        attn_mask=shifted_window_attn_mask_1d,
                                                                        )
                    else:
                        message = single_head_full_attention_1d(query, key, value,
                                                                h=height,
                                                                w=width,
                                                                )

        else:
            message = single_head_full_attention(query, key, value)  # [B, L, C]

        message = self.merge(message)  # [B, L, C]
        message = self.norm1(message)

        if not self.no_ffn:
            message = self.mlp(torch.cat([source, message], dim=-1))
            message = self.norm2(message)

        return source + message


class TransformerBlock(nn.Module):
    """self attention + cross attention + FFN"""

    def __init__(self,
                 d_model=128,
                 nhead=1,
                 ffn_dim_expansion=4,
                 ):
        super(TransformerBlock, self).__init__()

        self.self_attn = TransformerLayer(d_model=d_model,
                                          nhead=nhead,
                                          no_ffn=True,
                                          ffn_dim_expansion=ffn_dim_expansion,
                                          )

        self.cross_attn_ffn = TransformerLayer(d_model=d_model,
                                               nhead=nhead,
                                               ffn_dim_expansion=ffn_dim_expansion,
                                               )

    def forward(self, source, target,
                height=None,
                width=None,
                shifted_window_attn_mask=None,
                shifted_window_attn_mask_1d=None,
                attn_type='swin',
                with_shift=False,
                attn_num_splits=None,
                ):
        # source, target: [B, L, C]
        #print(source.shape, target.shape)
        # self attention

        # cross attention and ffn
        source = self.cross_attn_ffn(source, target,
                                     height=height,
                                     width=width,
                                     shifted_window_attn_mask=shifted_window_attn_mask,
                                     shifted_window_attn_mask_1d=shifted_window_attn_mask_1d,
                                     attn_type=attn_type,
                                     with_shift=with_shift,
                                     attn_num_splits=attn_num_splits,
                                     )

        source = self.self_attn(source, source,
                                height=height,
                                width=width,
                                shifted_window_attn_mask=shifted_window_attn_mask,
                                attn_type=attn_type,
                                with_shift=with_shift,
                                attn_num_splits=attn_num_splits,
                                )

        return source


class FeatureTransformerC(nn.Module):
    def __init__(self,
                 num_layers=3,
                 d_model=128,
                 nhead=1,
                 ffn_dim_expansion=2,
                 ):
        super(FeatureTransformerC, self).__init__()

        self.d_model = d_model
        self.nhead = nhead

        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model,
                             nhead=nhead,
                             ffn_dim_expansion=ffn_dim_expansion,
                             )
            for i in range(num_layers)])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature0, feature1,
                attn_type='swin',
                attn_num_splits=None,
                **kwargs,
                ):

        b, c, h, w = feature0.shape
        assert self.d_model == c

        feature0 = feature0.flatten(-2).permute(0, 2, 1).contiguous()  # [B, H*W, C]
        feature1 = feature1.flatten(-2).permute(0, 2, 1).contiguous()  # [B, H*W, C]

        # 2d attention
        if 'swin' in attn_type and attn_num_splits > 1:
            # global and refine use different number of splits
            window_size_h = h // attn_num_splits
            window_size_w = w // attn_num_splits

            # compute attn mask once
            shifted_window_attn_mask = generate_shift_window_attn_mask(
                input_resolution=(h, w),
                window_size_h=window_size_h,
                window_size_w=window_size_w,
                shift_size_h=window_size_h // 2,
                shift_size_w=window_size_w // 2,
                device=feature0.device,
            )  # [K*K, H/K*W/K, H/K*W/K]
        else:
            shifted_window_attn_mask = None

        # 1d attention
        if 'swin1d' in attn_type and attn_num_splits > 1:
            window_size_w = w // attn_num_splits

            # compute attn mask once
            shifted_window_attn_mask_1d = generate_shift_window_attn_mask_1d(
                input_w=w,
                window_size_w=window_size_w,
                shift_size_w=window_size_w // 2,
                device=feature0.device,
            )  # [K, W/K, W/K]
        else:
            shifted_window_attn_mask_1d = None

        agg_feat = self.layers[0](feature0, feature1,
                            height=h,
                            width=w,
                            attn_type=attn_type,
                            with_shift=False,
                            attn_num_splits=attn_num_splits,
                            shifted_window_attn_mask=shifted_window_attn_mask,
                            shifted_window_attn_mask_1d=shifted_window_attn_mask_1d,
                            )
        # agg_feat1 = self.layers[1](feature1, feature0,
        #                     height=h,
        #                     width=w,
        #                     attn_type=attn_type,
        #                     with_shift=False,
        #                     attn_num_splits=attn_num_splits,
        #                     shifted_window_attn_mask=shifted_window_attn_mask,
        #                     shifted_window_attn_mask_1d=shifted_window_attn_mask_1d,
        #                     )
        # agg_feat = self.layers[2](agg_feat0, agg_feat1,
        #                     height=h,
        #                     width=w,
        #                     attn_type=attn_type,
        #                     with_shift=False,
        #                     attn_num_splits=attn_num_splits,
        #                     shifted_window_attn_mask=shifted_window_attn_mask,
        #                     shifted_window_attn_mask_1d=shifted_window_attn_mask_1d,
        #                     )
            # update feature1
        #print(agg_feat.shape)
        agg_feat = agg_feat.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return agg_feat

