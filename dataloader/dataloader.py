import os
import json
from networkx import center
import numpy as np
from scipy.stats import multivariate_normal
from PIL import Image
from scipy.sparse import coo_matrix
from torchvision.datasets import VisionDataset
import torch
from torchvision.transforms import ToTensor
from utils import basic
from operator import itemgetter

class GetDataset(VisionDataset):
    def __init__(self, base, reID=False, train=True, transform=ToTensor(), target_transform=ToTensor(),grid_reduce=4, img_reduce=4, train_ratio=0.9, sample_require=0):
        # parameters in super can be accessed as (self.param) eg:- self.transform, self.target_transform
        super().__init__(base.root, transform=transform, target_transform=target_transform)
        self.base = base
        self.reID =  reID
        self.root, self.num_cam, self.num_frames = base.root, base.num_cam, base.num_frames
        self.dataset_name = base.dataset_name
        self.img_shape, self.world_grid_shape = base.img_shape, base.world_grid_shape # H,W; N_row,N_col
        if sample_require:
           self.skip_frame_ratio = int(self.num_frames//sample_require)
        else:
           self.skip_frame_ratio = 1
        
        # Reduce grid [480,1440] by factor of 4.
        self.grid_reduce = grid_reduce
        #[480,1440]/4 --> [120,360] i.e [1200cm,3600cm] and 10cm is pixel size, therefore [1200/10,3600/10]
        self.reducedgrid_shape = list(map(lambda x: int(x / self.grid_reduce), self.world_grid_shape)) 

        self.img_reduce = img_reduce
        
        # Map kernel size = 41*41
        map_sigma, map_kernel_size = 20 / grid_reduce, 20
        x, y = np.meshgrid(np.arange(-map_kernel_size, map_kernel_size + 1),
                           np.arange(-map_kernel_size, map_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        map_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * map_sigma)
        # normalizing the kernel with max value.
        map_kernel = map_kernel / map_kernel.max()
        kernel_size = map_kernel.shape[0]
        self.map_kernel = torch.zeros([1, 1, kernel_size, kernel_size], requires_grad=False)
        self.map_kernel[0, 0] = torch.from_numpy(map_kernel)
        
        # Split train/test data
        print(self.skip_frame_ratio)
        print(train_ratio)
        if train:
            frame_range = range(0,int(train_ratio*self.num_frames), self.skip_frame_ratio)
        else:
            frame_range = range(int(train_ratio*self.num_frames), self.num_frames, self.skip_frame_ratio)
        
        ############ Cam set Selection ############
        #frame_range = range(0,self.num_frames) 
        ###########################################
        self.img_fpath = self.base.get_image_paths(frame_range)
  
        self.world_gt = {}
        self.imgs_gt = {}
        self.pid_dict = {}
        self.download(frame_range)


    def download(self, frame_range):
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)

                world_pts, world_pids = [], []
                img_bboxs, img_pids = [[] for _ in range(self.num_cam)], [[] for _ in range(self.num_cam)]
                for pedestrian in all_pedestrians:
                    if pedestrian is None:
                        continue
                    
                    x, y = self.base.get_worldgrid_from_pos(pedestrian['positionID'])
                    
                    if pedestrian['personID'] not in self.pid_dict:
                        self.pid_dict[pedestrian['personID']] = len(self.pid_dict)
                    
                    if self.base.indexing == 'xy':
                        world_pts.append((x / self.grid_reduce, y / self.grid_reduce))
                    else:
                        world_pts.append((y / self.grid_reduce, x / self.grid_reduce))
                    
                    world_pids.append(pedestrian['personID'])
                    for cam in range(self.num_cam):
                        if itemgetter('xmin', 'ymin', 'xmax', 'ymax')(pedestrian['views'][cam]) != (-1, -1, -1, -1):
                            img_bboxs[cam].append(itemgetter('xmin', 'ymin', 'xmax', 'ymax')(pedestrian['views'][cam]))
                            img_pids[cam].append(pedestrian['personID'])

                self.world_gt[frame] = (np.array(world_pts), np.array(world_pids))
                self.imgs_gt[frame] = {}
                for cam in range(self.num_cam):
                    self.imgs_gt[frame][cam] = (np.array(img_bboxs[cam]), np.array(img_pids[cam]))


    def get_bev_gt(self, mem_pts, pids):
        Y, X = self.reducedgrid_shape
        center = torch.zeros((1, Y, X), dtype=torch.float32)
        valid_mask = torch.zeros((1, Y, X), dtype=torch.bool)
        offset = torch.zeros((2, Y, X), dtype=torch.float32)
        person_ids = torch.zeros((1, Y, X), dtype=torch.long)
        for pt, pid in zip(mem_pts, pids):
            ct = torch.tensor(pt[:2], dtype=torch.float32)
            ct_int = ct.int()
            if ct_int[1] >= Y or ct_int[0] >= X or ct_int[1] < 0 or ct_int[0] < 0:
                continue
            basic.draw_umich_gaussian(center[0], ct_int, sigma=1.5)
            valid_mask[:, ct_int[1], ct_int[0]] = 1
            offset[:, ct_int[1], ct_int[0]] = ct - ct_int
            person_ids[:, ct_int[1], ct_int[0]] = pid
        return center, valid_mask, person_ids, offset

    def get_img_gt(self, img_pts, img_pids, H, W):

        orig_H, orig_W = self.img_shape  # 예: (720, 1280)
        scale_x = W / orig_W
        scale_y = H / orig_H

        # center = torch.zeros((2, H, W), dtype=torch.float32)
        # offset = torch.zeros((2, H, W), dtype=torch.float32)
        valid_mask = torch.zeros((1, H, W), dtype=torch.bool)
        person_ids = torch.zeros((1, H, W), dtype=torch.long)

        for pt, pid in zip(img_pts, img_pids):
            # bbox 좌표 스케일링
            x1, y1, x2, y2 = pt
            x1 = x1 * scale_x
            x2 = x2 * scale_x
            y1 = y1 * scale_y
            y2 = y2 * scale_y
            
            # ct = torch.tensor([(pt[0]+pt[2])/2, (pt[1]+pt[3])/2], dtype=torch.float32)
            ct = torch.tensor([(x1 + x2) / 2, (y1 + y2) / 2], dtype=torch.float32)
            ct_int = ct.int()
            # print(f"bbox: {pt}, center: {ct_int}, H: {H}, W: {W}")
            if ct_int[0] < 0 or ct_int[0] >= W or ct_int[1] < 0 or ct_int[1] >= H:
                continue
            # basic.draw_umich_gaussian(center[0], ct_int, sigma=1.5)
            valid_mask[:, ct_int[1], ct_int[0]] = 1
            # offset[:, ct_int[1], ct_int[0]] = ct - ct_int
            person_ids[:, ct_int[1], ct_int[0]] = pid
        # return center, offset, person_ids, valid_mask
        return person_ids, valid_mask

    def __getitem__(self, index):
        frame = list(self.world_gt.keys())[index]
        imgs = []
        for cam in range(self.num_cam):

            img = Image.open(self.img_fpath[cam][frame]).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)

        # BEV GT
        worldgrid_pts, world_pids = self.world_gt[frame]
        center_bev, valid_bev, pid_bev, offset_bev = self.get_bev_gt(worldgrid_pts, world_pids)

        # IMG GT (각 카메라별)
        _, _, H, W = imgs.shape
        H, W = H // self.img_reduce, W // self.img_reduce
        center_imgs, offset_imgs, pid_imgs, valid_imgs = [], [], [], []
        for cam in range(self.num_cam):
            img_pts, img_pids = self.imgs_gt[frame][cam]
            if len(img_pts) == 0:
                # 빈 경우도 shape 맞춰서 반환
                # center_img = torch.zeros((2, H, W), dtype=torch.float32)
                # offset_img = torch.zeros((2, H, W), dtype=torch.float32)
                pid_img = torch.zeros((1, H, W), dtype=torch.long)
                valid_img = torch.zeros((1, H, W), dtype=torch.bool)
            else:
                # center_img, offset_img, pid_img, valid_img = self.get_img_gt(img_pts, img_pids, H, W)
                pid_img, valid_img = self.get_img_gt(img_pts, img_pids, H, W)
            # center_imgs.append(center_img)
            # offset_imgs.append(offset_img)
            pid_imgs.append(pid_img)
            valid_imgs.append(valid_img)
        # center_imgs = torch.stack(center_imgs)
        # offset_imgs = torch.stack(offset_imgs)
        pid_imgs = torch.stack(pid_imgs)
        valid_imgs = torch.stack(valid_imgs)

        # grid_gt: (max_objects, 3) [x, y, pid]
        max_objects = 60
        worldgrid_pts_torch = torch.tensor(worldgrid_pts, dtype=torch.long)
        world_pids_torch = torch.tensor(world_pids, dtype=torch.long)
        grid_gt = torch.zeros((max_objects, 3), dtype=torch.long)
        n_obj = min(len(worldgrid_pts_torch), max_objects)
        if n_obj > 0:
            grid_gt[:n_obj, :2] = worldgrid_pts_torch[:n_obj]
            grid_gt[:n_obj, 2] = world_pids_torch[:n_obj]

        return {
            'img': imgs,
            'frame': frame,
            'grid_gt': grid_gt,
            'root': self.root,
        }, {
            'center_bev': center_bev,
            'valid_bev': valid_bev,
            'pid_bev': pid_bev,
            'offset_bev': offset_bev,
            
            'pid_img': pid_imgs,
            'valid_img': valid_imgs,
        }


    def __len__(self):
        # length of dataset
        return len(self.world_gt.keys())
