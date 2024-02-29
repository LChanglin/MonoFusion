from PIL import Image
import os
import time
import numpy as np
import torch
import torch.nn.functional as F


def get_loss(scale, flow, dc_gt, flow_gt, valid, epoch):

    alpha_s = 0.8
    alpha_f = 0.2
    
    gt_dchange = dc_gt[:,0:1,:,:]
    valids = dc_gt[:, 1, :, :].unsqueeze(1).bool()

    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    # mag0 = torch.sum(flow**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < 400)

    gt_dchange[gt_dchange<=0] = 10
    gt_dchange[gt_dchange>3] = 10

    maskdc = ((gt_dchange < 3) & (gt_dchange > 0.3) & valids & (scale>0))

    mask_nan = ~torch.isnan(scale)
    flow_nan = ~torch.isnan(flow) & ~torch.isnan(flow_gt)
    maskdc = maskdc & mask_nan
    mask_minus0 = (scale<=0)

    # flow_nan = flow_nan[:,0:1,...] & flow_nan[:,1:,...]
    valid_f = valid[:, None] & flow_nan

    # print(valid[:, None].sum(), valid_f.sum())
    # valid_f = valid[:, None]

    if mask_minus0.sum() == 0:    
        loss1 =  ((((scale.abs()).log()-(gt_dchange.abs()).log()).abs())*maskdc).sum() / (maskdc.sum()+1e-8)
        # loss1 =  (((scale.log()-gt_dchange.log()).abs())*maskdc).mean()
        sloss = loss1
    else:
        loss1 = (((scale-gt_dchange).abs())[mask_nan]).mean()
        loss2 = (((scale.abs())*mask_minus0)).sum() / (mask_minus0.sum()+1e-8)
        sloss = loss1 + loss2

    f_loss = (flow-flow_gt).abs()

    floss = (valid_f * f_loss).sum()/(valid_f.sum()+1e-8)
    # floss = (valid_f * f_loss).mean()

    if torch.isnan(f_loss).sum()>0:
        loss = sloss
    else:
        loss = alpha_s*sloss + alpha_f*floss
        
    epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)
    mag = mag.view(-1)
    val = valid.view(-1) >= 0.5
    out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
    f1 = torch.mean(out[val])

    return sloss, loss, gt_dchange, f1

def get_loss_multi(scale, flow_preds, dc_gt, flow_gt, valid, epoch, gamma=0.9):

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    alpha_s = 1.
    alpha_f = 0.2
    
    gt_dchange = dc_gt[:,0:1,:,:]
    valids = dc_gt[:, 1, :, :].unsqueeze(1).bool()
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < 400)

    gt_dchange[gt_dchange<=0] = 10
    gt_dchange[gt_dchange>3] = 10

    maskdc = ((gt_dchange < 3) & (gt_dchange > 0.3) & valids & (scale>0))
    mask_nan = ~torch.isnan(scale)
    maskdc = maskdc & mask_nan
    mask_minus0 = (scale<=0)

    if mask_minus0.sum() == 0:    
        loss1 =  ((((scale.abs()).log()-(gt_dchange.abs()).log()).abs())*maskdc).sum() / (maskdc.sum()+1e-8)
        # loss1 =  (((scale.log()-gt_dchange.log()).abs())*maskdc).mean()
        sloss = loss1
    else:
        loss1 = (((scale-gt_dchange).abs())[mask_nan]).mean()
        loss2 = (((scale.abs())*mask_minus0)).sum() / (mask_minus0.sum()+1e-8)
        sloss = loss1 + loss2

    for i in range(n_predictions):

        flow = flow_preds[i]
        flow_nan = ~torch.isnan(flow) & ~torch.isnan(flow_gt)
        valid_f = valid[:, None] & flow_nan
        i_weight = gamma ** (n_predictions - i - 1)

        f_loss = (flow-flow_gt).abs()
        floss = (valid_f * f_loss).sum()/(valid_f.sum()+1e-8)

        flow_loss += i_weight * floss

    if torch.isnan(f_loss).sum()>0:
        loss = sloss
    else:
        loss = alpha_s*sloss + alpha_f*flow_loss

    epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)
    mag = mag.view(-1)
    val = valid.view(-1) >= 0.5
    out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
    f1 = torch.mean(out[val])

    return sloss, loss, gt_dchange, f1

def get_loss_multi_multi(scale_preds, flow_preds, dc_gt, flow_gt, valid, epoch, gamma=0.9):

    n_predictions_s = len(scale_preds)
    n_predictions = len(flow_preds)
    # print(n_predictions_s, n_predictions)
    flow_loss = 0.0
    scale_loss = 0.0

    alpha_s = 1.
    alpha_f = 0.2
    
    gt_dchange = dc_gt[:,0:1,:,:]
    valids = dc_gt[:, 1, :, :].unsqueeze(1).bool()
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < 400)

    gt_dchange[gt_dchange<=0] = 10
    gt_dchange[gt_dchange>3] = 10

    for i in range(n_predictions_s):
        scale = scale_preds[i]
        i_weight = gamma ** (n_predictions_s - i - 1)
        
        maskdc = ((gt_dchange < 3) & (gt_dchange > 0.3) & valids & (scale>0))
        mask_nan = ~torch.isnan(scale)
        maskdc = maskdc & mask_nan
        mask_minus0 = (scale<=0)

        if mask_minus0.sum() == 0:    
            loss1 =  ((((scale.abs()).log()-(gt_dchange.abs()).log()).abs())*maskdc).sum() / (maskdc.sum()+1e-8)
            # loss1 =  (((scale.log()-gt_dchange.log()).abs())*maskdc).mean()
            sloss = loss1
        else:
            loss1 = (((scale-gt_dchange).abs())[mask_nan]).mean()
            loss2 = (((scale.abs())*mask_minus0)).sum() / (mask_minus0.sum()+1e-8)
            sloss = loss1 + loss2

        if i==n_predictions_s-1:
            sloss_f = sloss
        scale_loss += i_weight * sloss

    for i in range(n_predictions):

        flow = flow_preds[i]
        flow_nan = ~torch.isnan(flow) & ~torch.isnan(flow_gt)
        valid_f = valid[:, None] & flow_nan
        i_weight = gamma ** (n_predictions - i - 1)

        f_loss = (flow-flow_gt).abs()
        floss = (valid_f * f_loss).sum()/(valid_f.sum()+1e-8)

        flow_loss += i_weight * floss

    if torch.isnan(f_loss).sum()>0:
        loss = scale_loss
    else:
        loss = alpha_s*scale_loss + alpha_f*flow_loss

    epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)
    mag = mag.view(-1)
    val = valid.view(-1) >= 0.5
    out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
    f1 = torch.mean(out[val])

    return sloss_f, loss, gt_dchange, f1

def get_loss_of(scale, flow, dc_gt, flow_gt, valid, ts):

    alpha_s = 1.
    alpha_f = 0.15

    gt_dchange = dc_gt[:,0:1,:,:]
    valids = dc_gt[:, 1, :, :].unsqueeze(1).bool()

    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    # mag0 = torch.sum(flow**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < 400)

    gt_dchange[gt_dchange<=0] = 10
    gt_dchange[gt_dchange>3] = 10

    maskdc = ((gt_dchange < 3) & (gt_dchange > 0.3) & valids & (scale>0))

    mask_nan = ~torch.isnan(scale)
    flow_nan = ~torch.isnan(flow) & ~torch.isnan(flow_gt)
    maskdc = maskdc & mask_nan
    mask_minus0 = (scale<=0)

    # flow_nan = flow_nan[:,0:1,...] & flow_nan[:,1:,...]
    valid_f = valid[:, None] & flow_nan

    # print(valid[:, None].sum(), valid_f.sum())
    # valid_f = valid[:, None]

    mask_ts = torch.ones_like(maskdc)
    mask_ts = mask_ts * ts[:,None,None,None]

    maskdc = maskdc * mask_ts

    if mask_minus0.sum() == 0:    
        # loss1 =  ((((scale.abs()).log()-(gt_dchange.abs()).log()).abs())*maskdc).sum() / (maskdc.sum()+1e-8)
        loss1 =  ((((scale.abs())-(gt_dchange.abs())).abs())*maskdc).sum() / (maskdc.sum()+1e-8)
        sloss = loss1
    else:
        loss1 = (((scale-gt_dchange).abs())[mask_nan]).mean()
        loss2 = (((scale.abs())*mask_minus0)).sum() / (mask_minus0.sum()+1e-8)
        sloss = loss1 + loss2

    # print(sloss, ts)

    f_loss = (flow-flow_gt).abs()

    floss = (valid_f * f_loss).sum()/(valid_f.sum()+1e-8)

    if torch.isnan(f_loss).sum()>0:
        loss = sloss
    else:
        loss = alpha_s*sloss + alpha_f*floss
        
    epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)
    mag = mag.view(-1)
    val = valid.view(-1) >= 0.5
    out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
    f1 = torch.mean(out[val])

    return sloss, loss, gt_dchange, f1

def get_loss_test(scale, flow, dc_gt, flow_gt, valid):


    gt_dchange = dc_gt[:,0:1,:,:]
    valids = dc_gt[:, 1, :, :].unsqueeze(1).bool()

    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < 400)

    gt_dchange[gt_dchange<=0] = 10
    gt_dchange[gt_dchange>3] = 10

    maskdc = ((gt_dchange < 3) & (gt_dchange > 0.3) & valids)

    ls =  ((scale.log()-gt_dchange.log()).abs())[maskdc].mean()

    epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)
    mag = mag.view(-1)
    val = valid.view(-1) >= 0.5
    out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()

    epe = torch.mean(epe[val])
    f1 = torch.mean(out[val])

    return ls, epe, f1


def get_loss_norm(scale, flow, dc_gt, flow_gt, valid, epoch):

    alpha_s = 0.8
    alpha_f = 0.2

    b, _, h, w = flow_gt.shape

    gt_dchange = dc_gt[:,0:1,:,:]
    valids = dc_gt[:, 1, :, :].unsqueeze(1).bool()

    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    # mag0 = torch.sum(flow**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < 400)

    gt_dchange[gt_dchange<=0] = 10
    gt_dchange[gt_dchange>3] = 10

    maskdc = ((gt_dchange < 3) & (gt_dchange > 0.3) & valids & (scale>0))

    mask_nan = ~torch.isnan(scale)
    maskdc = maskdc & mask_nan
    mask_minus0 = (scale<=0)

    flow_nan = ~torch.isnan(flow) & ~torch.isnan(flow_gt)
    valid_f = valid[:, None] & flow_nan

    if mask_minus0.sum() == 0:    
        loss1 =  ((((scale.abs()).log()-(gt_dchange.abs()).log()).abs())*maskdc).sum() / (maskdc.sum())
        sloss = loss1
        if epoch < 5:
            loss2 = (((scale.abs())*mask_minus0)).sum() / (mask_minus0.sum()+1e-4)
            sloss += loss2
    else:
        loss1 = (((scale-gt_dchange).abs())[mask_nan]).mean()
        loss2 = (((scale.abs())*mask_minus0)).sum() / (mask_minus0.sum()+1e-4)
        sloss = loss1 + loss2

    f_loss = (flow-flow_gt).abs()
    floss = (valid_f * f_loss).sum()/(valid_f.sum()+1e-8)

    loss = alpha_s*sloss + alpha_f*floss
    # print(torch.max(mag0[valid]),torch.max(mag[valid]))
    # print(torch.mean(mag0[valid]),torch.mean(mag[valid]))
    # print('xxx')
    epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)
    mag = mag.view(-1)
    val = valid.view(-1) >= 0.5
    out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
    f1 = torch.mean(out[val])

    return sloss, loss, gt_dchange, f1


def get_loss2(scale, flow, flow_oris, dc_gt, flow_gt, valid, epoch):

    alpha_s = 0.8
    alpha_f = 0.2

    gt_dchange = dc_gt[:,0:1,:,:]
    valids = dc_gt[:, 1, :, :].unsqueeze(1).bool()

    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    # mag0 = torch.sum(flow**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < 400)

    gt_dchange[gt_dchange<=0] = 10
    gt_dchange[gt_dchange>3] = 10

    maskdc = ((gt_dchange < 3) & (gt_dchange > 0.3) & valids & (scale>0))

    mask_nan = ~torch.isnan(scale)
    maskdc = maskdc & mask_nan
    mask_minus0 = (scale<=0)

    if mask_minus0.sum() == 0:    
        loss1 =  ((((scale.abs()).log()-(gt_dchange.abs()).log()).abs())*maskdc).sum() / (maskdc.sum())
        sloss = loss1
        if epoch < 5:
            loss2 = (((scale.abs())*mask_minus0)).sum() / (mask_minus0.sum()+1e-4)
            sloss += loss2
    else:
        loss1 = (((scale-gt_dchange).abs())[mask_nan]).mean()
        loss2 = (((scale.abs())*mask_minus0)).sum() / (mask_minus0.sum()+1e-4)
        sloss = loss1 + loss2

    gamma=0.5
    floss2 = 0
    n_predictions = len(flow_oris)
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        flow0 = flow_oris[i]
        f_loss2 = (flow0-flow_gt).abs()
        floss2 += (valid[:, None] * f_loss2).sum()/valid.sum() * i_weight

    f_loss = (flow-flow_gt).abs()
    f_loss2 = (flow0-flow_gt).abs()
    floss1 = (valid[:, None] * f_loss).sum()/valid.sum()
    floss = floss1 + 0.1*floss2

    loss = alpha_s*sloss + alpha_f*floss
    # print(torch.max(mag0[valid]),torch.max(mag[valid]))
    # print(torch.mean(mag0[valid]),torch.mean(mag[valid]))
    # print('xxx')
    epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)
    mag = mag.view(-1)
    val = valid.view(-1) >= 0.5
    out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
    f1 = torch.mean(out[val])

    return sloss, loss, gt_dchange, f1



def ttc_smooth_loss(img, disp, mask):
    """
        Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
    """
    # normalize
    mean_disp = disp.mean(2, True).mean(3, True)
    norm_disp = disp / (mean_disp + 1e-7)
    disp = norm_disp

    grad_disp_x = torch.abs(disp - torch.roll(disp, 1, dims=3))
    grad_disp_y = torch.abs(disp - torch.roll(disp, 1, dims=2))
    grad_disp_x[:,:,:,0] = 0
    grad_disp_y[:,:,0,:] = 0

    # grad_disp_xx = torch.abs(torch.roll(grad_disp_x, -1, dims=3) - grad_disp_x)
    # grad_disp_yy = torch.abs(torch.roll(grad_disp_y, -1, dims=3) - grad_disp_y)
    # grad_disp_xx[:,:,:,0] = 0
    # grad_disp_yy[:,:,0,:] = 0
    # grad_disp_xx[:,:,:,-1] = 0
    # grad_disp_yy[:,:,-1,:] = 0

    grad_img_x = torch.mean(torch.abs(img - torch.roll(img, 1, dims=3)), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img - torch.roll(img, 1, dims=2)), 1, keepdim=True)
    grad_img_x[:,:,:,0] = 0
    grad_img_y[:,:,0,:] = 0

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return (grad_disp_x*mask).sum()/mask.sum() + (grad_disp_y*mask).sum()/mask.sum()





def get_loss_selfsup(scale, valid, scale_gt):
    return (scale.log()-scale_gt.log()).abs()[valid.bool()].mean()


def self_supervised_gt_affine(flow):

    b,_,lh,lw=flow.shape
    bs, w,h = b, lw, lh
    grid_H = torch.linspace(0, w-1, w).view(1, 1, 1, w).expand(bs, 1, h, w).to(device=flow.device, dtype=flow.dtype)
    grid_V = torch.linspace(0, h-1, h).view(1, 1, h, 1).expand(bs, 1, h, w).to(device=flow.device, dtype=flow.dtype)
    pref = torch.cat([grid_H, grid_V], dim=1)
    ptar = pref + flow
    pw = 1
    pref = F.unfold(pref, (pw*2+1,pw*2+1), padding=(pw)).view(b,2,(pw*2+1)**2,lh,lw)-pref[:,:,np.newaxis]
    ptar = F.unfold(ptar, (pw*2+1,pw*2+1), padding=(pw)).view(b,2,(pw*2+1)**2,lh,lw)-ptar[:,:,np.newaxis] # b, 2,9,h,w
    pref = pref.permute(0,3,4,1,2).reshape(b*lh*lw,2,(pw*2+1)**2)
    ptar = ptar.permute(0,3,4,1,2).reshape(b*lh*lw,2,(pw*2+1)**2)

    prefprefT = pref.matmul(pref.permute(0,2,1))
    ppdet = prefprefT[:,0,0]*prefprefT[:,1,1]-prefprefT[:,1,0]*prefprefT[:,0,1]
    ppinv = torch.cat((prefprefT[:,1,1:],-prefprefT[:,0,1:], -prefprefT[:,1:,0], prefprefT[:,0:1,0]),1).view(-1,2,2)/ppdet.clamp(1e-10,np.inf)[:,np.newaxis,np.newaxis]

    Affine = ptar.matmul(pref.permute(0,2,1)).matmul(ppinv)
    Error = (Affine.matmul(pref)-ptar).norm(2,1).mean(1).view(b,1,lh,lw)

    Avol = (Affine[:,0,0]*Affine[:,1,1]-Affine[:,1,0]*Affine[:,0,1]).view(b,1,lh,lw).abs().clamp(1e-10,np.inf)
    exp = Avol.sqrt()
    mask = (exp>0.5) & (exp<2)
    mask = mask[:,0]

    exp = exp.clamp(0.5,2)
    # exp[Error>0.1]=1
    return torch.reciprocal(exp)


def self_supervised_gt(flow_f):

    b,_,h,w = flow_f.size()
    grid_H = torch.linspace(0, w-1, w).view(1, 1, 1, w).expand(b, 1, h, w).to(device=flow_f.device, dtype=flow_f.dtype)
    grid_V = torch.linspace(0, h-1, h).view(1, 1, h, 1).expand(b, 1, h, w).to(device=flow_f.device, dtype=flow_f.dtype)
    grids1_ = torch.cat([grid_H, grid_V], dim=1)

    gw = 2
    pad_dim = (gw,gw,gw,gw)
    grids_pad = F.pad(grids1_, pad_dim, "replicate")
    flow_f_pad = F.pad(flow_f, pad_dim, "replicate")
    grids_w_f = grids_pad + flow_f_pad

    # tm - m
    len_ori = torch.abs(grids_pad[...,0:-2*gw,gw:-gw] - grids_pad[...,gw:-gw,gw:-gw]) 
    len_scale_f = torch.abs(grids_w_f[...,0:-2*gw,gw:-gw] - grids_w_f[...,gw:-gw,gw:-gw]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_y_len1 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)
    # bm - m
    len_ori = torch.abs(grids_pad[...,2*gw:,gw:-gw] - grids_pad[...,gw:-gw,gw:-gw]) 
    len_scale_f = torch.abs(grids_w_f[...,2*gw:,gw:-gw] - grids_w_f[...,gw:-gw,gw:-gw]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_y_len2 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)

    # mr - m
    len_ori = torch.abs(grids_pad[...,gw:-gw,2*gw:] - grids_pad[...,gw:-gw,gw:-gw]) 
    len_scale_f = torch.abs(grids_w_f[...,gw:-gw,2*gw:] - grids_w_f[...,gw:-gw,gw:-gw]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_x_len1 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)
    # ml - m
    len_ori = torch.abs(grids_pad[...,gw:-gw,0:-2*gw] - grids_pad[...,gw:-gw,gw:-gw]) 
    len_scale_f = torch.abs(grids_w_f[...,gw:-gw,0:-2*gw] - grids_w_f[...,gw:-gw,gw:-gw]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_x_len2 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)

    # tm - mr
    len_ori = torch.abs(grids_pad[...,0:-2*gw,gw:-gw] - grids_pad[...,gw:-gw,2*gw:]) 
    len_scale_f = torch.abs(grids_w_f[...,0:-2*gw,gw:-gw] - grids_w_f[...,gw:-gw,2*gw:]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_l_len1 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)
    # ml - bm
    len_ori = torch.abs(grids_pad[...,gw:-gw,0:-2*gw] - grids_pad[...,2*gw:,gw:-gw]) 
    len_scale_f = torch.abs(grids_w_f[...,gw:-gw,0:-2*gw] - grids_w_f[...,2*gw:,gw:-gw]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_l_len2 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)

    # tm - ml
    len_ori = torch.abs(grids_pad[...,0:-2*gw,gw:-gw] - grids_pad[...,gw:-gw,0:-2*gw]) 
    len_scale_f = torch.abs(grids_w_f[...,0:-2*gw,gw:-gw] - grids_w_f[...,gw:-gw,0:-2*gw]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_r_len1 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)
    # mr - bm
    len_ori = torch.abs(grids_pad[...,gw:-gw,2*gw:] - grids_pad[...,2*gw:,gw:-gw]) 
    len_scale_f = torch.abs(grids_w_f[...,gw:-gw,2*gw:] - grids_w_f[...,2*gw:,gw:-gw]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_r_len2 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)


    exp_x_len = torch.min(torch.stack([exp_x_len1,exp_x_len2],dim=1),dim=1)[0]
    exp_y_len = torch.min(torch.stack([exp_y_len1,exp_y_len2],dim=1),dim=1)[0]
    exp_l_len = torch.min(torch.stack([exp_l_len1,exp_l_len2],dim=1),dim=1)[0]
    exp_r_len = torch.min(torch.stack([exp_r_len1,exp_r_len2],dim=1),dim=1)[0]

    exp_corner_len = torch.max(torch.stack([exp_l_len,exp_r_len],dim=1),dim=1)[0]
    exp_f_xy_len = torch.max(exp_x_len,exp_y_len)

    threshold = 0.15
    ttc_range = 0.95
    mask_exp1 = torch.logical_and(exp_corner_len<1. ,(exp_corner_len/exp_f_xy_len)>1)
    mask_exp2 = torch.logical_and((exp_x_len/exp_y_len)>(1-threshold),(exp_x_len/exp_y_len)<(1+threshold))
    mask_exp2 = torch.logical_and(mask_exp2,exp_f_xy_len<ttc_range*torch.mean(exp_f_xy_len))
    mask_exp = torch.logical_and(mask_exp1, mask_exp2)

    scale_gt = mask_exp*exp_corner_len+ ~mask_exp*exp_f_xy_len
    scale_gt[...,0:1,:] = 1
    scale_gt[...,-1:,:] = 1
    scale_gt[...,:,0:1] = 1
    scale_gt[...,:,-1:] = 1

    return scale_gt

