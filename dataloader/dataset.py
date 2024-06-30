import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
import os
import pickle
import math
import random
from glob import glob
import os.path as osp
from .utils.rectangle_noise import retangle
from .utils import frame_utils
import  cv2
from .utils.augmentor import FlowAugmentor, SparseFlowAugmentorm


def depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth

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
def get_grid_np(B,H,W):
    meshgrid_base = np.meshgrid(range(0, W), range(0, H))[::-1]
    basey = np.reshape(meshgrid_base[0], [1, 1, 1, H, W])
    basex = np.reshape(meshgrid_base[1], [1, 1, 1, H, W])
    grid = torch.tensor(np.concatenate((basex.reshape((-1, H, W, 1)), basey.reshape((-1, H, W, 1))), -1)).float()
    return grid.view( H, W, 2)

class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, vkitti2=False, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentorm(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)
        self.driving = False
        self.is_test = False
        self.init_seed = False
        self.test_scene = False
        self.stereo = False
        self.flow_list = []
        self.dispnet =[]
        self.depth_list = []
        self.image_list = []
        self.extra_info = []
        self.mask_list = []
        self.occ_list = []
        self.rect = retangle()
        self.kit = 0
        self.k = 1
        self.kr = 0
        self.get_depth = 0
        self.kitti_test = 0
        self.sintel_test = 0

        self.train_scale = 1

        self.vkitti2 = vkitti2

        self.last_image = np.random.randn(320,960,3)
    def __getitem__(self, index):
        self.kit = self.kit +1
        #print(self.image_list[index][0], self.image_list[index][1], self.flow_list[index])
        if self.test_scene:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            dispnet = np.abs(disparity_loader(self.dispnet[index]))
            return img1, img2, self.extra_info[index],dispnet
        if self.is_test and not self.kitti_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]
        if self.get_depth:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            d1, d2, mask = self.get_dc(index)
            dc_change = d2 / d1
            mask[dc_change > 1.5] = 0
            mask[dc_change < 0.5] = 0
            d1[mask == 0] = 0
            d2[mask == 0] = 0
            dc_change[mask == 0] = 0
            dc_change = np.concatenate((dc_change[:, :, np.newaxis], mask[:, :, np.newaxis]), axis=2)
            #读取光流结果
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
            flow = np.array(flow).astype(np.float32)
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)
            for i in range(int(self.kr)):
                imgb1, imgb2, ansb, flag = self.rect.get_mask(img1)
                if flag > 1:
                    img1[imgb1 > 0] = imgb1[imgb1 > 0]
                    img2[imgb2 > 0] = imgb2[imgb2 > 0]
                    flow[imgb1[:, :, 0] > 0, :] = ansb[imgb1[:, :, 0] > 0, :2]
                    dc_change[imgb1[:, :, 0] > 0, 0:1] = ansb[imgb1[:, :, 0] > 0, 2:]
                    d1[imgb1[:, :, 0] > 0] = 10
                    d2[imgb1[:, :, 0] > 0] = dc_change[imgb1[:, :, 0] > 0,0]*10
                    li = ansb[:, :, 2] > 0
                    dc_change[li, 1] = 1
                    mask[imgb1[:, :, 0] > 0]=2

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
            disp1 = self.depth_to_disp(d1)
            disp2 = self.depth_to_disp(d2)
            disp1[mask == 0] = 0
            disp2[mask == 0] = 0

            return img1,img2,flow,dc_change,d1,d2,disp1,disp2,mask,self.extra_info[index]#这个mask是是否有噪音块的掩膜

        if self.kitti_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            #mask = frame_utils.read_gen(self.mask_list[index])
            d1, d2, mask = self.get_dc(index)
            dc_change = d2 / d1
            d1[mask == 0] = 0
            d2[mask == 0] = 0
            dc_change[mask == 0] = 0
            dc_change = np.concatenate((dc_change[:, :, np.newaxis], mask[:, :, np.newaxis]), axis=2)
            # 读取光流结果
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
            flow = np.array(flow).astype(np.float32)
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)
            dc_change = np.array(dc_change).astype(np.float32)

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
            dc_change = torch.from_numpy(dc_change).permute(2, 0, 1).float()
            valid = torch.from_numpy(valid)
            return img1, img2, flow, dc_change, valid.float()  # 这个mask是是否有噪音块的掩膜

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None

        d1, d2, mask, disp1 = self.get_dc(index)
        dc_change = d2/d1
        mask[dc_change>1.5] = 0
        mask[dc_change <0.5] = 0

        if self.sparse:
            if self.driving:
                flow, valid = frame_utils.readFlowdriving(self.flow_list[index])
            elif self.stereo:
                flowx = disparity_loader(self.depth_list[index][0])
                flow = np.concatenate((flowx[:, :, np.newaxis], flowx[:, :, np.newaxis]), axis=2)
                valid = flowx>0
                flow[:,:,1]=0
            elif self.vkitti2:
                flow, valid = frame_utils.read_vkitti2_flow(self.flow_list[index])
                mask = np.logical_and(mask, valid)
            else:
                flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])

        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        dc_change[mask==0]=0
        disp1[mask==0]=0
        if self.occlusion:
            dcc = dc_change
            dcc = abs(cv2.filter2D(dcc,-1,kernel=self.kernel2))
            maskd = torch.from_numpy(dcc>1).bool()
            dc_change[maskd!=0] = 0
            masku = dc_change>0
            #再加一个遮挡
            dc_change = np.concatenate((dc_change[:,:,np.newaxis],masku[:,:,np.newaxis]),axis =2 )
        else:
            dc_change = np.concatenate((dc_change[:, :, np.newaxis], mask[:, :, np.newaxis]), axis=2)
            #disp1 = np.concatenate((disp1[:, :, np.newaxis], mask[:, :, np.newaxis]), axis=2)

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        dc_change = np.array(dc_change).astype(np.float32)
        disp1 = np.array(disp1).astype(np.float32)
        

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]


        if self.augmentor is not None:
            if self.sparse:
                disp1 = disp1[:, :, np.newaxis]
                img1, img2, flow, dc_change, disp1, valid = self.augmentor(img1, img2, flow,dc_change, disp1, valid)
            else:
                img1, img2, flow, dc_change, disp1 = self.augmentor(img1, img2, flow, dc_change, disp1)
                disp1 = disp1[:, :, np.newaxis]

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        dc_change = torch.from_numpy(dc_change).permute(2, 0, 1).float()
        disp1 = torch.from_numpy(disp1).permute(2, 0, 1).float()
        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, dc_change, disp1, valid.float(), self.train_scale

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        self.depth_list = v * self.depth_list
        self.occ_list = v * self.occ_list
        return self

    def __len__(self):
        return len(self.image_list)

class MpiSinteltest(FlowDataset):#/home/lh/RAFT-master/dataset/Sintel
    def __init__(self, aug_params=None, split='training', root='/mnt/pool/Datasets/OpticalFlow/Sintel', dstype='clean'):
        super(MpiSinteltest, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)
        depth_root = osp.join(root, split, 'depth')
        occ_root = osp.join(root, split, 'occlusions')
        self.occlusion = True
        self.sintel_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            depth_list = sorted(glob(osp.join(depth_root, scene, '*.dpt')))
            occ_list = sorted(glob(osp.join(occ_root, scene, '*.png')))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                if split != 'test':
                    self.depth_list += [[depth_list[i], depth_list[i + 1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))

                if self.occlusion:
                    for i in range(len(image_list) - 1):
                        self.occ_list += [occ_list[i]]
    def get_dc(self,index):
        if self.occ_list is not None:
            occ = frame_utils.read_gen(self.occ_list[index])
            occ = np.array(occ).astype(np.uint8)
            occ = torch.from_numpy(occ // 255).bool()

        flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)
        h, w, c = flow.shape

        depth1 = torch.tensor(depth_read(self.depth_list[index][0]))
        depth2 = torch.tensor(depth_read(self.depth_list[index][1])).view(1, 1, h, w)
        flowg = torch.tensor(flow)
        frep = get_grid_np(c, h, w)
        frepb = (frep + flowg).view(1, h, w, 2)
        frepb[:, :, :, 0] = frepb[:, :, :, 0] / (w / 2.) - 1
        frepb[:, :, :, 1] = frepb[:, :, :, 1] / (h / 2.) - 1
        depth2 = (torch.nn.functional.grid_sample(depth2, frepb,mode='nearest').view(h, w))
        depth2 = depth2.view(h, w)
        return depth1.numpy(),depth2.numpy(),1-occ.numpy()
    def depth_to_disp(self,Z, bl=1, fl=1000):
        disp = bl * fl / Z
        return disp

class MpiSintel(FlowDataset):#/home/lh/RAFT-master/dataset/Sintel
    def __init__(self, aug_params=None, split='training', root='/mnt/pool/Datasets/OpticalFlow/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)
        depth_root = osp.join(root, split, 'depth')
        occ_root = osp.join(root, split, 'occlusions')
        self.train_scale = 0
        self.occlusion = True
        if split == 'test':
            self.is_test = True
        self.kernel = np.ones([5, 5], np.float32)
        self.kernel2 = np.ones([3, 3], np.float32)*-1
        self.kernel2[1,1] = 8
        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            depth_list = sorted(glob(osp.join(depth_root, scene, '*.dpt')))
            occ_list = sorted(glob(osp.join(occ_root, scene, '*.png')))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                if split != 'test':
                    self.depth_list += [[depth_list[i], depth_list[i + 1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))

                if self.occlusion:
                    for i in range(len(image_list) - 1):
                        self.occ_list += [occ_list[i]]
    def get_dc(self,index):
        if self.occ_list is not None:
            occ = frame_utils.read_gen(self.occ_list[index])
            occ = np.array(occ).astype(np.uint8)
            occ = torch.from_numpy(occ // 255).bool()
            #膨胀occ
            '''
            acc = occ.numpy().astype(np.uint8)
            occ = cv2.filter2D(acc,-1,kernel=self.kernel)
            occ = torch.from_numpy(occ>0).bool()
            '''
        flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)
        h, w, c = flow.shape

        depth1 = torch.tensor(depth_read(self.depth_list[index][0]))
        depth2 = torch.tensor(depth_read(self.depth_list[index][1])).view(1, 1, h, w)
        flowg = torch.tensor(flow)
        frep = get_grid_np(c, h, w)
        frepb = (frep + flowg).view(1, h, w, 2)
        frepb[:, :, :, 0] = frepb[:, :, :, 0] / (w / 2.) - 1
        frepb[:, :, :, 1] = frepb[:, :, :, 1] / (h / 2.) - 1
        depth2 = (torch.nn.functional.grid_sample(depth2, frepb,mode='nearest').view(h, w))
        depth2 = depth2.view(h, w)
        return depth1.numpy(),depth2.numpy(),1-occ.numpy()
        
class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='/mnt/pool/Datasets/OpticalFlow/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images) // 2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == 'training' and xid == 1) or (split == 'validation' and xid == 2):
                self.flow_list += [flows[i]]
                self.image_list += [[images[2 * i], images[2 * i + 1]]]

class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='/mnt/pool/Datasets/OpticalFlow/FlyingThings3D/', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)
        exclude = np.loadtxt('/mnt/pool/Datasets/OpticalFlow/exclude.txt', delimiter=' ', dtype=np.unicode_)
        exclude = set(exclude)
        self.occlusion = False
        self.driving = True
        self.train_scale = 0

        self.dstype = 'frames_cleanpass'
        self.mode = 'TRAIN'

        for cam in ['left', 'right']:
            image_dirs = sorted(glob(osp.join(root, self.dstype, self.mode, '*/*')))
            image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

            flow_dirs = sorted(glob(osp.join(root, 'optical_flow', self.mode, '*/*')))
            flow_dirs_forw = sorted([osp.join(f, 'into_future', cam) for f in flow_dirs])
            flow_dirs_back = sorted([osp.join(f, 'into_past', cam) for f in flow_dirs])

            depth_dirs = sorted(glob(osp.join(root, 'disparity', self.mode, '*/*')))
            depth_dirs = sorted([osp.join(f, cam) for f in depth_dirs])

            delta_dirs = sorted(glob(osp.join(root, 'disparity_change', self.mode, '*/*')))
            delta_dirs_forw = sorted([osp.join(f, 'into_future', cam) for f in delta_dirs])
            delta_dirs_back = sorted([osp.join(f, 'into_past', cam) for f in delta_dirs])



            for idir, fdir_forw, fdir_back, ddir_forw, ddir_back, zdir in \
                    zip(image_dirs, flow_dirs_forw, flow_dirs_back, delta_dirs_forw, delta_dirs_back, depth_dirs):
                
                images = sorted(glob(osp.join(idir, '*.png')))
                flows_forw = sorted(glob(osp.join(fdir_forw, '*.pfm')))
                flows_back = sorted(glob(osp.join(fdir_back, '*.pfm')))

                delta_forw = sorted(glob(osp.join(ddir_forw, '*.pfm')))
                delta_back = sorted(glob(osp.join(ddir_back, '*.pfm')))
                
                depths = sorted(glob(osp.join(zdir, '*.pfm')))

                for i in range(1, len(images)-1):
                    tag = '/'.join(images[i].split('/')[-5:])
                    if tag in exclude:
                        # print("Excluding %s" % tag)
                        continue

                    self.image_list += [[images[i], images[i+1]]]
                    self.flow_list += [flows_forw[i]]
                    self.depth_list += [[depths[i], delta_forw[i]]]
                    frame_id = images[i].split('/')[-1]
                    self.extra_info += [[frame_id]]

                    self.image_list += [[images[i], images[i-1]]]
                    self.flow_list += [flows_back[i]]
                    self.depth_list += [[depths[i], delta_back[i]]]
                    frame_id = images[i].split('/')[-1]
                    self.extra_info += [[frame_id]]

    def triangulation(self, disp, bl=1):#kitti flow 2015

        fl = 1050
        depth = bl * fl / disp  # 450px->15mm focal length
        Z = depth
        return Z

    def get_dc(self,index):
        # print(index, '????')
        disp1 = np.abs(disparity_loader(self.depth_list[index][0]))
        disp2 = np.abs(disparity_loader(self.depth_list[index][1])+disp1)
        depth1 = self.triangulation(disp1)
        depth2 = self.triangulation(disp2)
        flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)
        mask = np.logical_and(np.logical_and(np.logical_and(flow[:, :, 0] != 0, flow[:, :, 1] != 0), disp1 != 0), disp1 != 0).astype(float)
        # print(depth1.shape, depth2.shape, mask.shape, disp1.shape)
        return depth1, depth2, mask, disp1

class VKITTI2(FlowDataset):
    def __init__(self, aug_params=None,
                 vkitti2=True,
                 root='/mnt/pool/Datasets/OpticalFlow/vkitti',
                 ):
        super(VKITTI2, self).__init__(aug_params, sparse=True, vkitti2=True,
                                      )

        data_dir = root
        self.occlusion = False
        self.sflow_list = []
        scenes = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']

        for scene in scenes:
            scene_dir = os.path.join(data_dir, scene)

            types = os.listdir(scene_dir)

            for scene_type in types:
                type_dir = os.path.join(scene_dir, scene_type)

                imgs = sorted(glob(os.path.join(type_dir, 'frames', 'rgb', 'Camera_0', '*.jpg')))
                depths = sorted(glob(os.path.join(type_dir, 'frames', 'depth', 'Camera_0', '*.png')))

                flows_fwd = sorted(glob(os.path.join(type_dir, 'frames', 'forwardFlow', 'Camera_0', '*.png')))
                flows_bwd = sorted(glob(os.path.join(type_dir, 'frames', 'backwardFlow', 'Camera_0', '*.png')))

                sflows_fwd = sorted(glob(os.path.join(type_dir, 'frames', 'forwardSceneFlow', 'Camera_0', '*.png')))
                sflows_bwd = sorted(glob(os.path.join(type_dir, 'frames', 'backwardSceneFlow', 'Camera_0', '*.png')))

                assert len(imgs) == len(flows_fwd) + 1 and len(imgs) == len(flows_bwd) + 1

                for i in range(len(imgs) - 1):
                    # forward
                    self.image_list += [[imgs[i], imgs[i + 1]]]
                    self.flow_list += [flows_fwd[i]]
                    self.depth_list += [depths[i]]
                    self.sflow_list += [sflows_fwd[i]]

                    # backward
                    self.image_list += [[imgs[i+1], imgs[i]]]
                    self.flow_list += [flows_bwd[i]]
                    self.depth_list += [depths[i+1]]
                    self.sflow_list += [sflows_bwd[i]]

    def depth_to_disp(self,Z, bl=1, fl=1000):
        disp = bl * fl / Z
        return disp

    #获取有效区域的掩膜，以及两个深度
    def get_dc(self,index):

        d1 = cv2.imread(self.depth_list[index], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 100.
        #print(np.min(d1), np.max(d1))
        dc = frame_utils.read_vkitti2_dc(self.sflow_list[index])
        # print(np.min(dc), np.max(dc))
        d2 = d1 + dc
        # print(np.min(dc/d1+1), np.max(dc/d1+1))

        disp1 = self.depth_to_disp(d1)

        return d1, d2 , np.ones_like(d1).astype(float), disp1
    



class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='../../../../mnt/pool/Datasets/OpticalFlow/kitti/data_scene_flow',get_depth=0):
        super(KITTI, self).__init__(aug_params, sparse=True)
        self.get_depth=get_depth
        if split == 'testing':
            self.is_test = True
        if split == 'submit':
            self.is_test = True
        if split == 'submitother':
            self.is_test = True
        if split =='test':
            self.test_scene = True
        self.occlusion = False
        images1 =[]
        images2 =[]
        disp1 = []
        disp2 = []
        flow =[]

        k = 0
        if split == 'training':
            root = osp.join(root, split)

            images1o = sorted(glob(osp.join(root, 'image_2/*_10.png')))
            images2o = sorted(glob(osp.join(root, 'image_2/*_11.png')))
            disp1o = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
            disp2o = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))

            for j in range(images2o.__len__()):
                #if j%10!=k and j%10!=(k+5):
                #if j%10!=k:
                images1.append(images1o[j])
                images2.append(images2o[j])
                disp1.append(disp1o[j])
                disp2.append(disp2o[j])
        elif split=='testing':
            root = osp.join(root, 'training')
            images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
            images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))
            disp1 = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
            disp2 = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))
        elif split=='submit':
            images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
            images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))
            disp1 = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
            disp2 = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))
        elif split=='test':
            images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
            images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))
            disp1 = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
            disp2 = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))
            disp1LEA = sorted(glob(osp.join(root, 'disp_ganet_testing/*_10.png')))
            self.dispnet = disp1LEA
        else:
            images1 = sorted(glob(osp.join(root, '*.jpg')))
            images2 = images1[1:]
            images1.pop()
            disp1 = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
            disp2 = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))


        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]
        for disps1, disps2 in zip(disp1, disp2):
            self.depth_list += [[disps1, disps2]]
        if split == 'training':
            flowo = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
            for j in range(flowo.__len__()):
                #if j%10!=k and j%10!=(k+5):
                    #if j%10!=k:
                flow.append(flowo[j])
        elif split == 'testing':
            flow = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
        self.flow_list = flow


    def triangulation(self, disp, bl=0.5327254279298227, fl=721.5377):#kitti flow 2015
        disp[disp==0]= 1
        depth = bl * fl / disp  # 450px->15mm focal length
        Z = depth
        return Z
    def depth_to_disp(self,Z, bl=0.5327254279298227, fl=721.5377):
        disp = bl * fl / Z
        return disp

    #获取有效区域的掩膜，以及两个深度
    def get_dc(self,index):

        d1 = disparity_loader(self.depth_list[index][0])
        d2 = disparity_loader(self.depth_list[index][1])
        flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)
        mask = np.logical_and(np.logical_and(np.logical_and(flow[:, :, 0] != 0, flow[:, :, 1] != 0), d1 != 0), d2 != 0).astype(float)

        return self.triangulation(d1), self.triangulation(d2), mask, d1


class KITTI_160(FlowDataset):
    def __init__(self, aug_params=None, split='kitti_test', root='/mnt/pool/Datasets/OpticalFlow/kitti/data_scene_flow',get_depth=0):
        super(KITTI_160, self).__init__(aug_params, sparse=True)
        self.get_depth=get_depth
        if split == 'kitti_test':
           self.kitti_test = 1
        self.occlusion = False
        images1 =[]
        images2 =[]
        disp1 = []
        disp2 = []
        flow =[]
        if split == 'kitti_test':
            root = osp.join(root, 'training')

            images1o = sorted(glob(osp.join(root, 'image_2/*_10.png')))
            images2o = sorted(glob(osp.join(root, 'image_2/*_11.png')))
            disp1o = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
            disp2o = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))

            for j in range(images2o.__len__()):
                # if j%5==0:
                images1.append(images1o[j])
                images2.append(images2o[j])
                disp1.append(disp1o[j])
                disp2.append(disp2o[j])


        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]
        for disps1, disps2 in zip(disp1, disp2):
            self.depth_list += [[disps1, disps2]]
        if split == 'kitti_test':
            flowo = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
            for j in range(flowo.__len__()):
                # if j%5==0:
                flow.append(flowo[j])

        self.flow_list = flow


    def triangulation(self, disp, bl=0.5327254279298227, fl=721.5377):#kitti flow 2015
        disp[disp==0]= 1
        depth = bl * fl / disp  # 450px->15mm focal length
        Z = depth
        return Z
    def depth_to_disp(self,Z, bl=0.5327254279298227, fl=721.5377):
        disp = bl * fl / Z
        return disp

    #获取有效区域的掩膜，以及两个深度
    def get_dc(self,index):

        d1 = disparity_loader(self.depth_list[index][0])
        d2 = disparity_loader(self.depth_list[index][1])
        flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)
        mask = np.logical_and(np.logical_and(np.logical_and(flow[:, :, 0] != 0, flow[:, :, 1] != 0), d1 != 0), d2 != 0).astype(float)

        return self.triangulation(d1),self.triangulation(d2),mask

class KITTI_test200(FlowDataset):#/home/lh/RAFT3D-DEPTH/data_train_test /home/xuxian/RAFT3D/data_train_test
    def  __init__(self, aug_params=None, split='kitti_test', root='/mnt/pool/Datasets/OpticalFlow/kitti/data_scene_flow/training',get_depth=0):
        super(KITTI_test200, self).__init__(aug_params, sparse=True)
        self.get_depth=get_depth
        self.occlusion = False
        if split == 'kitti_test':
           self.kitti_test = 1
        images1 =[]
        images2 =[]
        disp1 = []
        disp2 = []
        flow =[]
        mask = []
        images1o = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2o = sorted(glob(osp.join(root, 'image_2/*_11.png')))
        disp1o = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
        disp2o = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))
        masko = sorted(glob(osp.join(root, 'mask_img/*_10.png')))
        flowo = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
        for j in range(images2o.__len__()):
            if j%5==0:
                images1.append(images1o[j])
                images2.append(images2o[j])
                disp1.append(disp1o[j])
                disp2.append(disp2o[j])
                mask.append(masko[j])
                flow.append(flowo[j])


        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]
        for disps1, disps2 in zip(disp1, disp2):
            self.depth_list += [[disps1, disps2]]

        
        # for j in range(flowo.__len__()):
                

        self.flow_list = flow
        self.mask_list = mask

    def triangulation(self, disp, bl=0.5327254279298227, fl=721.5377):#kitti flow 2015
        disp[disp==0]= 1
        depth = bl * fl / disp  # 450px->15mm focal length
        Z = depth
        return Z
    def depth_to_disp(self,Z, bl=0.5327254279298227, fl=721.5377):
        disp = bl * fl / Z
        return disp

    #获取有效区域的掩膜，以及两个深度
    def get_dc(self,index):

        d1 = disparity_loader(self.depth_list[index][0])
        d2 = disparity_loader(self.depth_list[index][1])
        flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)
        mask = np.logical_and(np.logical_and(np.logical_and(flow[:, :, 0] != 0, flow[:, :, 1] != 0), d1 != 0), d2 != 0).astype(float)

        return self.triangulation(d1),self.triangulation(d2),mask

class KITTI_test(FlowDataset):#/home/lh/RAFT3D-DEPTH/data_train_test /home/xuxian/RAFT3D/data_train_test#/new_data/kitti_data/datasets/training
    def  __init__(self, aug_params=None, split='kitti_test', root='/new_data/kitti_data/datasets/training',get_depth=0):
        super(KITTI_test, self).__init__(aug_params, sparse=True)
        self.get_depth=get_depth
        self.occlusion = False
        if split == 'kitti_test':
           self.kitti_test = 1
        images1 =[]
        images2 =[]
        disp1 = []
        disp2 = []
        flow =[]
        mask = []
        images1o = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2o = sorted(glob(osp.join(root, 'image_2/*_11.png')))
        disp1o = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
        disp2o = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))
        masko = sorted(glob(osp.join(root, 'mask_img/*_10.png')))
        for j in range(images2o.__len__()):
            images1.append(images1o[j])
            images2.append(images2o[j])
            disp1.append(disp1o[j])
            disp2.append(disp2o[j])
            mask.append(masko[j])


        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]
        for disps1, disps2 in zip(disp1, disp2):
            self.depth_list += [[disps1, disps2]]

        #flowo = sorted(glob(osp.join(root, 'flow/*_10.png')))
        flowo = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
        for j in range(flowo.__len__()):
                flow.append(flowo[j])

        self.flow_list = flow
        self.mask_list = mask

    def triangulation(self, disp, bl=0.5327254279298227, fl=721.5377):#kitti flow 2015
        disp[disp==0]= 1
        depth = bl * fl / disp  # 450px->15mm focal length
        Z = depth
        return Z
    def depth_to_disp(self,Z, bl=0.5327254279298227, fl=721.5377):
        disp = bl * fl / Z
        return disp

    #获取有效区域的掩膜，以及两个深度
    def get_dc(self,index):

        d1 = disparity_loader(self.depth_list[index][0])
        d2 = disparity_loader(self.depth_list[index][1])
        flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)
        mask = np.logical_and(np.logical_and(np.logical_and(flow[:, :, 0] != 0, flow[:, :, 1] != 0), d1 != 0), d2 != 0).astype(float)

        return self.triangulation(d1),self.triangulation(d2),mask

class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='/home/lh/RAFT-master/dataset/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows) - 1):
                self.flow_list += [flows[i]]
                self.image_list += [[images[i], images[i + 1]]]

            seq_ix += 1


class Driving(FlowDataset):
    def __init__(self, aug_params=None,  split='training',root='/mnt/pool/Datasets/OpticalFlow/Driving'):
        super(Driving, self).__init__(aug_params, sparse=False)
        self.calib = []
        self.occlusion = False
        self.driving = True
        level_stars = '/*' * 6
        candidate_pool = glob('%s/optical_flow%s' % (root, level_stars))
        for flow_path in sorted(candidate_pool):
            idd = flow_path.split('/')[-1].split('_')[-2]
            if 'into_future' in flow_path:
                idd_p1 = '%04d' % (int(idd) + 1)
            else:
                idd_p1 = '%04d' % (int(idd) - 1)
            if os.path.exists(flow_path.replace(idd, idd_p1)):
                d0_path = flow_path.replace('/into_future/', '/').replace('/into_past/', '/').replace('optical_flow','disparity')
                d0_path = '%s/%s.pfm' % (d0_path.rsplit('/', 1)[0], idd)
                dc_path = flow_path.replace('optical_flow', 'disparity_change')
                dc_path = '%s/%s.pfm' % (dc_path.rsplit('/', 1)[0], idd)
                im_path = flow_path.replace('/into_future/', '/').replace('/into_past/', '/').replace('optical_flow','frames_cleanpass')
                im0_path = '%s/%s.png' % (im_path.rsplit('/', 1)[0], idd)
                im1_path = '%s/%s.png' % (im_path.rsplit('/', 1)[0], idd_p1)
                frame_id = im1_path.split('/')[-1]
                self.extra_info += [[frame_id]]
                #calib.append('%s/camera_data.txt' % (im0_path.replace('frames_cleanpass', 'camera_data').rsplit('/', 2)[0]))
                self.flow_list += [flow_path]
                self.image_list += [[im0_path,im1_path]]
                self.depth_list += [[d0_path,dc_path]]
                self.calib +=['%s/camera_data.txt' % (im0_path.replace('frames_cleanpass', 'camera_data').rsplit('/', 2)[0])]
    def triangulation(self, disp,index, bl=1):#kitti flow 2015

        #print(len(self.calib), index)
        if '15mm_' in self.calib[index]:
            fl = 450  # 450
        else:
            fl = 1050
        depth = bl * fl / disp  # 450px->15mm focal length
        Z = depth
        return Z

    def get_dc(self,index):
        d1 = np.abs(disparity_loader(self.depth_list[index][0]))
        d2 = np.abs(disparity_loader(self.depth_list[index][1])+d1)
        flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)
        mask = np.logical_and(np.logical_and(np.logical_and(flow[:, :, 0] != 0, flow[:, :, 1] != 0), d1 != 0), d2 != 0).astype(float)

        return self.triangulation(d1,index),self.triangulation(d2,index),mask,d1

def fetch_dataloader(args, TRAIN_DS='C+T+K/S'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')

    elif args.stage == 'driving':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        driving = Driving(aug_params, split='training')
        train_dataset = driving

    elif args.stage == 'dk':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': True}
        driving = Driving(aug_params, split='training')
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': True}
        kitti = KITTI(aug_params, split='training')
        train_dataset = 100*kitti + driving


    elif args.stage == 'vkitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True}

        vkitti2 = VKITTI2(aug_params)  # 42420

        train_dataset = vkitti2

    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        driving = Driving(aug_params, split='training')
        train_dataset = clean_dataset+driving
        # train_dataset = clean_dataset

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        # things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        # sintel_final = MpiSintel(aug_params, split='training', dstype='final')
        train_dataset = 10 * sintel_clean

    elif args.stage == 'tdk':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        driving = Driving(aug_params, split='training')
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': True}
        kitti = KITTI(aug_params, split='training')
        # train_dataset = clean_dataset + 100*kitti + driving
        train_dataset =torch.utils.data.ConcatDataset([clean_dataset]*1 + [driving]*3 + [kitti]*300)

    elif args.stage == 'sdk':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        driving = Driving(aug_params, split='training')
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': True}
        kitti = KITTI(aug_params, split='training')

        train_dataset = 10*sintel_clean + 100*kitti + driving

    elif args.stage == 'td':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        driving = Driving(aug_params, split='training')
        train_dataset =torch.utils.data.ConcatDataset([clean_dataset]*1 + [driving]*1)

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': True}
        #aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': True}

        kitti = KITTI(aug_params, split='training')
        train_dataset = 100*kitti

    # train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
    #                                pin_memory=False, shuffle=True, num_workers=6, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_dataset