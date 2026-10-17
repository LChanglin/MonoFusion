"""Native PyTorch equivalents of the MMCV operations used by MonoFusion."""
import torch
import torch.nn.functional as F

def constant_init(module, val, bias=0.):
    torch.nn.init.constant_(module.weight, val)
    if module.bias is not None:
        torch.nn.init.constant_(module.bias, bias)

def xavier_init(module, gain=1., bias=0., distribution="normal"):
    init = torch.nn.init.xavier_uniform_ if distribution == "uniform" else torch.nn.init.xavier_normal_
    init(module.weight, gain=gain)
    if module.bias is not None:
        torch.nn.init.constant_(module.bias, bias)

def deformable_attention(value, spatial_shapes, locations, weights):
    # Equivalent to MMCV multi_scale_deformable_attn_pytorch: bilinear sampling,
    # zero padding, normalized [0,1] coordinates and align_corners=False.
    batch, _, heads, channels = value.shape
    _, queries, _, levels, points, _ = locations.shape
    shapes = spatial_shapes.tolist()
    values = value.split([h * w for h, w in shapes], dim=1)
    sampled = []
    for level, (height, width) in enumerate(shapes):
        image = values[level].flatten(2).transpose(1, 2).reshape(batch * heads, channels, height, width)
        grid = (2 * locations[:, :, :, level] - 1).transpose(1, 2).flatten(0, 1)
        sampled.append(F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False))
    weights = weights.transpose(1, 2).reshape(batch * heads, 1, queries, levels * points)
    output = (torch.stack(sampled, dim=-2).flatten(-2) * weights).sum(-1)
    return output.view(batch, heads * channels, queries).transpose(1, 2).contiguous()
