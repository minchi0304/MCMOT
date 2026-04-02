import os
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform.imgwarp import warp_perspective

import matplotlib.pyplot as plt
import warnings
from decoder import Decoder

warnings.filterwarnings("ignore")

class MultiView_Detection(nn.Module):
    def __init__(self, backbone_model, logdir, loss, avgpool, cam_set, len_cam_set,latent_dim=512,n_ids=10000, device=torch.device('cuda')):
        super().__init__()
        self.logdir = logdir
        self.avgpool = avgpool
        self.cam_set = cam_set
        self.MAX_CAM = 8
        self.device = device
        self.reid_feat = 64
        self.feat2d = 128
        self.base_arch = nn.Sequential(*list(backbone_model.children())[:-2]).to(self.device)
        self.n_ids = n_ids

        if avgpool:
            self.out_channel = 512
        else:
            if self.cam_set:
                self.out_channel = 512 * len_cam_set
            else:
                self.out_channel = 512 * self.MAX_CAM

        shared_out_channels = self.out_channel
        
        # bev
        self.instance_offset_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
        ).to(self.device)
        self.instance_center_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 1, kernel_size=1, padding=0),
        ).to(self.device)
        self.instance_center_head[-1].bias.data.fill_(-2.19)

        # re_id
        self.id_feat_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, self.reid_feat, kernel_size=1, padding=0),
        ).to(self.device)
        self.img_id_feat_head = nn.Sequential(
            nn.Conv2d(shared_out_channels,shared_out_channels , kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, self.reid_feat, kernel_size=1, padding=0),
        ).to(self.device)
        self.emb_scale = math.sqrt(2) * math.log(self.n_ids - 1)


        self.decoder = Decoder(in_channels=latent_dim,).to(self.device)
 
        # Ground Plane Convolution
        if loss == 'klcc':
            #### for KLDiv+CC ####
            self.map_classifier = nn.Sequential(nn.Conv2d(self.out_channel, 512, 3, padding=1), nn.ReLU(),
                                                nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
                                                nn.Conv2d(512, 1, 3, padding=4, dilation=4), nn.Sigmoid()).to(self.device)
            #############################################
        elif loss == 'mse':
            self.map_classifier = nn.Sequential(nn.Conv2d(self.out_channel, 512, 3, padding=1), nn.ReLU(),
                                                nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
                                                nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False)).to(self.device)

    # Uncertainty Weights
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.reid_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)


    def forward(self, dataset, dataset_name, imgs, ignore_cam, duplicate_cam, random, cam_selected):

        B, N, C, H, W = imgs.shape
        config_dict = dataset.dicts[dataset_name[0]]
        num_cam = config_dict['num_cam']
        upsample_shape = config_dict['upsample_shape']
        reducedgrid_shape = config_dict['reducedgrid_shape']
        img_reduce = config_dict['img_reduce']
        proj_mats = config_dict['proj_mats']
        coord_map = config_dict['coord_map']

        #vector code
        imgs_vec = imgs[0]  # (N, 3, H, W)
        img_features = self.base_arch(imgs_vec.to(self.device))  # (N, latent_dim, h, w)
        # img_features = self.encoder(imgs_vec.to(self.device))  # (N, latent_dim, h, w)

        img_features = F.interpolate(img_features, upsample_shape, mode='bilinear')  # (N, 512, H/8, W/8)


        # proj_mats: (N, 3, 3) 벡터 처리
        proj_mats = proj_mats.float().to(self.device)
        world_features = warp_perspective(img_features, proj_mats, reducedgrid_shape) #(N, 512, Hg/4, Wg/4)
        if self.avgpool:
            world_features = torch.mean(world_features, dim=0, keepdim=True)  # (1, 512, Hg/4, Wg/4)
            world_features = torch.cat([world_features, coord_map.repeat([B, 1, 1, 1]).to(self.device)], dim=1) #coord map 추가하면 채널 +2 해야함
        else:
            if num_cam < self.MAX_CAM:
                replicate_num = self.MAX_CAM - num_cam
                zero_vector = torch.zeros((1,512*replicate_num,reducedgrid_shape[0], reducedgrid_shape[1])).to(self.device)
                world_features = torch.cat([world_features.reshape(1, -1, *world_features.shape[2:]), zero_vector, coord_map.repeat([B, 1, 1, 1]).to(self.device)], dim=1)
            else:
                world_features = torch.cat([world_features.reshape(1, -1, *world_features.shape[2:]), coord_map.repeat([B, 1, 1, 1]).to(self.device)], dim=1)
            

        fig = plt.figure()
        subplt0 = fig.add_subplot(211, title="output")
        subplt0.imshow(torch.norm(world_features[0].detach(), dim=0).cpu().numpy())
        plt.savefig(os.path.join(self.logdir,'world_features.jpg'))
        plt.close(fig)
        
        
        bev_features = self.decoder(world_features)

        # map_result = self.map_classifier(bev_features.to(self.device)) #bev occupancy map
        # map_result = F.interpolate(map_result, reducedgrid_shape, mode='bilinear') 
        
        bev_occupancy = self.instance_center_head(bev_features)
        bev_offset = self.instance_offset_head(bev_features)

        # re_id features
        bev_reid = self.emb_scale * F.normalize(self.id_feat_head(bev_features), dim=1)
        img_reid = self.emb_scale * F.normalize(self.img_id_feat_head(img_features), dim=1)  # B*N,C,H/8,W/8


        return  bev_occupancy, bev_offset, bev_reid, img_reid
            
    def get_imgcoord2worldgrid_matrices(self, intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
        projection_matrices = {}
        for cam in range(self.num_cam):
            # removing third column(z=0) from extrinsic matrix of size 3x4 
            worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)
            worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
            imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)

            # transforming img axis to grid map axis
            # x(col),y(row) img coord -->  y(col), x(row) grid map coord
            permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
        return projection_matrices
    
    def get_coord_map(self, grid_shape):
        H, W, C = grid_shape
        # img[y,x] = img[h,w]
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        # making x and y in range [-1.0 to 1.0]
        grid_x = torch.from_numpy((grid_x / (W - 1) * 2) - 1).float()
        grid_y = torch.from_numpy((grid_y / (H - 1) * 2) - 1).float()
        ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        return ret
        
