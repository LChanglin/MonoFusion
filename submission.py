from PIL import Image
import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import datetime
from glob import glob
from selfttc.self_ttc_agg4gxr3 import SelfTTC
from utils.draw import disp2rgb, flow_uv_to_colors, flow_to_image

parser = argparse.ArgumentParser()

# dataset
parser.add_argument('--checkpoint_dir', default='tmp', type=str,
                    help='where to save the training log and models')
parser.add_argument('--stage', default='chairs', type=str,
                    help='training stage on different datasets')
parser.add_argument('--val_dataset', default=['chairs'], type=str, nargs='+',
                    help='validation datasets')
parser.add_argument('--max_flow', default=400, type=int,
                    help='exclude very large motions during training')
parser.add_argument('--image_size', default=[384, 512], type=int, nargs='+',
                    help='image size for training')
parser.add_argument('--padding_factor', default=16, type=int,
                    help='the input should be divisible by padding_factor, otherwise do padding or resizing')

# evaluation
parser.add_argument('--eval', action='store_true',
                    help='evaluation after training done')
parser.add_argument('--save_eval_to_file', action='store_true')
parser.add_argument('--evaluate_matched_unmatched', action='store_true')
parser.add_argument('--val_things_clean_only', action='store_true')
parser.add_argument('--with_speed_metric', action='store_true',
                    help='with speed methic when evaluation')

# training
parser.add_argument('--load_flow_param', action='store_true')
parser.add_argument('--epoch', default=400, type=int)
parser.add_argument('--lr', default=4e-4, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--grad_clip', default=1.0, type=float)
parser.add_argument('--num_steps', default=100000, type=int)
parser.add_argument('--seed', default=326, type=int)
parser.add_argument('--summary_freq', default=100, type=int)
parser.add_argument('--val_freq', default=10000, type=int)
parser.add_argument('--save_ckpt_freq', default=10000, type=int)
parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)

# resume pretrained model or resume training
parser.add_argument('--resume', default=None, type=str,
                    help='resume from pretrained model or resume from unexpectedly terminated training')
parser.add_argument('--strict_resume', action='store_true',
                    help='strict resume while loading pretrained weights')
parser.add_argument('--no_resume_optimizer', action='store_true')

# model: learnable parameters
parser.add_argument('--num_scales', default=1, type=int,
                    help='feature scales: 1/8 or 1/8 + 1/4')
parser.add_argument('--feature_channels', default=128, type=int)
parser.add_argument('--upsample_factor', default=8, type=int)
parser.add_argument('--num_head', default=1, type=int)
parser.add_argument('--ffn_dim_expansion', default=4, type=int)
parser.add_argument('--num_transformer_layers', default=6, type=int)
parser.add_argument('--reg_refine', action='store_true',
                    help='optional task-specific local regression refinement')
parser.add_argument('--parallel', action='store_true',
                    help='optional task-specific local regression refinement')
parser.add_argument('--load_opt', action='store_true',
                    help='optional task-specific local regression refinement')

# model: parameter-free
parser.add_argument('--attn_type', default='swin', type=str,
                    help='attention function')
parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                    help='number of splits in attention')
parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                    help='correlation radius for matching, -1 indicates global matching')
parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                    help='self-attention radius for propagation, -1 indicates global attention')
parser.add_argument('--num_reg_refine', default=1, type=int,
                    help='number of additional local regression refinement')

# loss
parser.add_argument('--gamma', default=0.9, type=float,
                    help='exponential weighting')

# predict on sintel and kitti test set for submission
parser.add_argument('--kittidataset', default='/mnt/pool2/lcl/data/data_scene_flow/training/', type=str)
parser.add_argument('--drivingdataset', default='/mnt/pool2/lcl/data/Driving/', type=str)
parser.add_argument('--submission', action='store_true',
                    help='submission to sintel or kitti test sets')
parser.add_argument('--output_path', default='output', type=str,
                    help='where to save the prediction results')
parser.add_argument('--save_vis_flow', action='store_true',
                    help='visualize flow prediction as .png image')
parser.add_argument('--no_save_flo', action='store_true',
                    help='not save flow as .flo if only visualization is needed')

# inference on images or videos
parser.add_argument('--inference_dir', default=None, type=str)
parser.add_argument('--inference_video', default=None, type=str)
parser.add_argument('--inference_size', default=None, type=int, nargs='+',
                    help='can specify the inference size for the input to the network')
parser.add_argument('--save_flo_flow', action='store_true')
parser.add_argument('--pred_bidir_flow', action='store_true',
                    help='predict bidirectional flow')
parser.add_argument('--pred_bwd_flow', action='store_true',
                    help='predict backward flow only')
parser.add_argument('--fwd_bwd_check', action='store_true',
                    help='forward backward consistency check with bidirection flow')
parser.add_argument('--save_video', action='store_true')
parser.add_argument('--concat_flow_img', action='store_true')

# distributed training
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--launcher', default='none', type=str, choices=['none', 'pytorch'])
parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

# misc
parser.add_argument('--count_time', action='store_true',
                    help='measure the inference time')

parser.add_argument('--debug', action='store_true')

args = parser.parse_args()
torch.cuda.set_device(0)
device = torch.device("cuda")


def readPFM(file):
    import re
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(b'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def disparity_loader(path):
    if '.png' in path:
        data = Image.open(path)
        data = np.ascontiguousarray(data,dtype=np.float32)/256
        return data
    else:
        return readPFM(path)[0]

def writeFlowKITTI(filename, uv):
    uv = 64.0 * uv + 2**15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])

def main():
    model_loaded = SelfTTC(num_scales=args.num_scales,
                            feature_channels=args.feature_channels,
                            upsample_factor=args.upsample_factor,
                            num_head=args.num_head,
                            ffn_dim_expansion=args.ffn_dim_expansion,
                            num_transformer_layers=args.num_transformer_layers,
                            reg_refine=args.reg_refine,
                            train=False).to(device)
    num_params = sum(p.numel() for p in model_loaded.parameters())
    print('Number of params:', num_params)


    if args.resume is not None:
        loc = 'cuda:{}'.format(args.local_rank) if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(args.resume, map_location=loc)
        if 'net' in checkpoint:
            # model.load_state_dict(data['net'])
            #print(data['net'].items())
            model_loaded.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})
        elif 'state_dict' in checkpoint:
            model_loaded.load_state_dict(checkpoint['state_dict'])
        else:
            model_loaded.load_state_dict(checkpoint)


    time_stamp = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S")

    model_loaded.eval()
    out_dir = "./output/%s_selfcon_ttc"%(time_stamp)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
        os.mkdir(out_dir+'/submit/')
        os.mkdir(out_dir+'/submit/flow/')
        os.mkdir(out_dir+'/submit/disp_0/')
        os.mkdir(out_dir+'/submit/disp_1/')

    # path1, path2 = 'test_img/2341.jpg', 'test_img/2344.jpg'
    inference_dir = args.inference_dir
    filenames = sorted(glob(inference_dir + '/*.png') + glob(inference_dir + '/*.jpg'))
    # print(filenames)
    print('%d images found' % len(filenames))

    w, h = 1216,352
    # w, h = 960, 288
    # w, h = 480,270
    total = 0
    with torch.no_grad():
        for test_id in range(0, 200):
            
            file_1 = inference_dir+'image_2/'+str(test_id).zfill(6)+'_10.png'
            file_2 = inference_dir+'image_2/'+str(test_id).zfill(6)+'_11.png'

            image1 = Image.open(file_1).convert('RGB')
            image2 = Image.open(file_2).convert('RGB')
            
            image1 = np.array(image1).astype(np.uint8)
            image2 = np.array(image2).astype(np.uint8)
            ori_size = image1.shape[0:2] 
            image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).to(device)
            image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0).to(device)

            padding_factor = 32
            inference_size = [h,w]

            
            if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                        align_corners=True)
                image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                        align_corners=True)

            start = time.time()
            scale, flow, depth,_ = model_loaded(image1, image2,
                        attn_type=args.attn_type,
                        attn_splits_list=args.attn_splits_list,
                        corr_radius_list=args.corr_radius_list,
                        prop_radius_list=args.prop_radius_list,
                        num_reg_refine=args.num_reg_refine,
                        testing=False)
            print(time.time()-start)

            if type(flow) == list:
                flow = flow[-1]
            if type(scale) == list:
                scale = scale[-1]

            # resize back
            if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                scale = F.interpolate(scale, size=ori_size, mode='bilinear',
                                        align_corners=True)
                flow = F.interpolate(flow, size=ori_size, mode='bilinear',
                                        align_corners=True)
                #print(flow.shape)
                flow[:,0:1,...] = flow[:,0:1,...] * ori_size[-1] / inference_size[-1]
                flow[:,1:,...] = flow[:,1:,...] * ori_size[0] / inference_size[0]
            

            flow = (flow.permute(0,2,3,1).detach().cpu().numpy())[0]
            scale = scale[0].transpose(0,1).transpose(1,2).detach().cpu().numpy()


            output_filename = os.path.join(out_dir+'/submit/flow/', str(test_id).zfill(6)+'_10.png')
            writeFlowKITTI(output_filename, flow)

            file_disp = inference_dir+'disp_ganet_testing/'+str(test_id).zfill(6)+'_10.png'
            disp = disparity_loader(file_disp)
            disp2 = disp / scale[:,:,0]
            disp1 =  (disp * 256).astype('uint16')
            disp2 =  (disp2 * 256).astype('uint16')
            cv2.imwrite(out_dir+'/submit/disp_0/'+str(test_id).zfill(6)+'_10.png', disp1)
            cv2.imwrite(out_dir+'/submit/disp_1/'+str(test_id).zfill(6)+'_10.png', disp2)

            out_of_viz = flow_to_image(flow)
            Image.fromarray(out_of_viz).save(os.path.join(out_dir,  'flow'+str(test_id)+'.jpg'))
            ttc_warp_image2 = (scale - 0.5) / (1.0) # [H, W, 2]
            ttc_warp_image2 = disp2rgb(np.clip(ttc_warp_image2, 0.0, 1.0))
            ttc_warp_image2 = ttc_warp_image2*255.0
            cv2.imwrite(os.path.join(out_dir, 'scale'+str(test_id)+'.png'), ttc_warp_image2)
            # cv2.imwrite(os.path.join(out_dir,  'flow'+str(test_id)+'.jpg'), out_of_viz)
            # txt_result = scale[0,0].detach().cpu().numpy()

        print('t: ',total/196)
    # w, h = 1152, 320
    # # w, h = 480, 270
    # startx=4300
    # endx=startx+500
    # total_t = 0
    # interval = 4
    # with torch.no_grad():
    #     for test_id in range(startx,endx):
    #         print(filenames[test_id+3][-8:-4])

    #         image1 = Image.open(filenames[test_id])
    #         image2 = Image.open(filenames[test_id + interval])
    #         # print(filenames[test_id+3], test_id+3)
    #         image1 = np.array(image1).astype(np.uint8)
    #         image2 = np.array(image2).astype(np.uint8)
    #         ori_shape = image1.shape
    #         ori_size = image1.shape[0:2] 
    #         image1 = cv2.resize(image1, (w,h))
    #         image2 = cv2.resize(image2, (w,h))
    #         image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).to(device)
    #         image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0).to(device)

    #         padding_factor = 32
    #         inference_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
    #                         int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]

    #         # ori_size = image1.shape[-2:]
    #         # if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
    #         #     image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
    #         #                             align_corners=True)
    #         #     image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
    #         #                             align_corners=True)

    #         start = time.time()
    #         scale, flow, _ = model_loaded(image1, image2,
    #                     attn_type=args.attn_type,
    #                     attn_splits_list=args.attn_splits_list,
    #                     corr_radius_list=args.corr_radius_list,
    #                     prop_radius_list=args.prop_radius_list,
    #                     num_reg_refine=args.num_reg_refine,
    #                     testing=True)
    #         end = time.time()-start
    #         # print(end)

    #         if test_id>startx+10:
    #             #print(total_t, end)
    #             total_t += end
                
    #         flow_pr = scale

    #         # resize back
    #         if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
    #             flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
    #                                     align_corners=True)
    #             flow = F.interpolate(flow, size=ori_size, mode='bilinear',
    #                                     align_corners=True)
    #             # print(ori_size[-1], inference_size[-1], torch.max(flow[:,0:1,...]), torch.max(flow[:,1:,...]))
    #             flow[:,0:1,...] = flow[:,0:1,...] * ori_size[-1] / inference_size[-1]
    #             flow[:,1:,...] = flow[:,1:,...] * ori_size[0] / inference_size[0]
    #         # print(flow_pr.shape, flow.shape, ori_size[-1] / inference_size[-1])

    #         out_of_viz = flow_to_image((flow.permute(0,2,3,1).detach().cpu().numpy())[0])
    #         cv2.imwrite(os.path.join(out_dir,  'flow'+str(test_id)+'.jpg'), out_of_viz)


    #         ttc_warp_image2 = (flow_pr[0].transpose(0,1).transpose(1,2) - 0.5) / (1.0) # [H, W, 2]
    #         ttc_warp_image2 = disp2rgb(np.clip(ttc_warp_image2.detach().cpu().numpy(), 0.0, 1.0))
    #         ttc_warp_image2 = ttc_warp_image2*255.0
    #         cv2.imwrite(os.path.join(out_dir, 'scale'+str(test_id)+'.png'), ttc_warp_image2)

    #     # print("????: ", total_t/100)

if __name__ == "__main__":
    main()