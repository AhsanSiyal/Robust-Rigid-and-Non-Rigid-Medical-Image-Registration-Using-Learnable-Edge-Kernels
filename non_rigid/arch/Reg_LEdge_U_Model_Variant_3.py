# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 12:53:16 2022

@author: Ahsan
"""

import torch.nn as nn
import torch
import nibabel as nib
import numpy as np
import torch.nn.functional as nnf
from arch.util_g1 import (
    ResidualConv,
    ASPP,
    AttentionBlock,
    Upsample_,
    Squeeze_Excite_Block,
)


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


def apply_affine_to_mri(mri_volume, affine_matrix):
    """
    Apply a 3D affine transformation to an MRI volume using PyTorch.
    :param mri_volume: A 3D PyTorch tensor representing the MRI data.
    :param affine_matrix: A 4x4 affine transformation matrix.
    :return: Transformed MRI volume.
    """
    B, C, D, H, W = mri_volume.size()

    # Create 3D grid
    grid = nnf.affine_grid(affine_matrix, [mri_volume.shape[0], 3, mri_volume.shape[2], mri_volume.shape[3], mri_volume.shape[4]], align_corners=False)

    # Apply the affine transformation
    transformed_volume = nnf.grid_sample(mri_volume, grid, align_corners=False, mode='bilinear')

    return transformed_volume


class AffineTransform(nn.Module):
    """
    3-D Affine Transformer
    """

    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def apply_affine(self, src, mat):
        grid = nnf.affine_grid(mat, [src.shape[0], 3, src.shape[2], src.shape[3], src.shape[4]], align_corners=False)
        return nnf.grid_sample(src, grid, align_corners=False, mode=self.mode)

    def forward(self, src, affine, scale, translate, shear):

        theta_x = affine[:, 0]
        theta_y = affine[:, 1]
        theta_z = affine[:, 2]
        scale_x = scale[:, 0]
        scale_y = scale[:, 1]
        scale_z = scale[:, 2]
        trans_x = translate[:, 0]
        trans_y = translate[:, 1]
        trans_z = translate[:, 2]
        shear_xy = shear[:, 0]
        shear_xz = shear[:, 1]
        shear_yx = shear[:, 2]
        shear_yz = shear[:, 3]
        shear_zx = shear[:, 4]
        shear_zy = shear[:, 5]

        rot_mat_x = torch.stack([torch.stack([torch.ones_like(theta_x), torch.zeros_like(theta_x), torch.zeros_like(theta_x)], dim=1), torch.stack([torch.zeros_like(theta_x), torch.cos(theta_x), -torch.sin(theta_x)], dim=1), torch.stack([torch.zeros_like(theta_x), torch.sin(theta_x), torch.cos(theta_x)], dim=1)], dim=2).cuda()
        rot_mat_y = torch.stack([torch.stack([torch.cos(theta_y), torch.zeros_like(theta_y), torch.sin(theta_y)], dim=1), torch.stack([torch.zeros_like(theta_y), torch.ones_like(theta_x), torch.zeros_like(theta_x)], dim=1), torch.stack([-torch.sin(theta_y), torch.zeros_like(theta_y), torch.cos(theta_y)], dim=1)], dim=2).cuda()
        rot_mat_z = torch.stack([torch.stack([torch.cos(theta_z), -torch.sin(theta_z), torch.zeros_like(theta_y)], dim=1), torch.stack([torch.sin(theta_z), torch.cos(theta_z), torch.zeros_like(theta_y)], dim=1), torch.stack([torch.zeros_like(theta_y), torch.zeros_like(theta_y), torch.ones_like(theta_x)], dim=1)], dim=2).cuda()
        scale_mat = torch.stack(
            [torch.stack([scale_x, torch.zeros_like(theta_z), torch.zeros_like(theta_y)], dim=1),
             torch.stack([torch.zeros_like(theta_z), scale_y, torch.zeros_like(theta_y)], dim=1),
             torch.stack([torch.zeros_like(theta_y), torch.zeros_like(theta_y), scale_z], dim=1)], dim=2).cuda()
        shear_mat = torch.stack(
            [torch.stack([torch.ones_like(theta_x), torch.tan(shear_xy), torch.tan(shear_xz)], dim=1),
             torch.stack([torch.tan(shear_yx), torch.ones_like(theta_x), torch.tan(shear_yz)], dim=1),
             torch.stack([torch.tan(shear_zx), torch.tan(shear_zy), torch.ones_like(theta_x)], dim=1)], dim=2).cuda()
        trans = torch.stack([trans_x, trans_y, trans_z], dim=1).unsqueeze(dim=2)
        mat = torch.bmm(shear_mat, torch.bmm(scale_mat, torch.bmm(rot_mat_z, torch.matmul(rot_mat_y, rot_mat_x))))
        inv_mat = torch.inverse(mat)
        mat = torch.cat([mat, trans], dim=-1)
        inv_trans = torch.bmm(-inv_mat, trans)
        inv_mat = torch.cat([inv_mat, inv_trans], dim=-1)
        grid = nnf.affine_grid(mat, [src.shape[0], 3, src.shape[2], src.shape[3], src.shape[4]], align_corners=False)
        return nnf.grid_sample(src, grid, align_corners=False, mode=self.mode), mat, inv_mat
    
# class Laplacian3DConv(nn.Module):
#     def __init__(self, in_channels):
#         super(Laplacian3DConv, self).__init__()
#         device=torch.device('cuda:0')
#         self.in_channels = in_channels
#         # Define the 3D Laplacian kernel
#         laplacian_kernel = torch.tensor([[[[0, 0, 0], [0, -1, 0], [0, 0, 0]],
#                                           [[0, -1, 0], [-1, 6, -1], [0, -1, 0]],
#                                           [[0, 0, 0], [0, -1, 0], [0, 0, 0]]]], 
#                                          dtype=torch.float32, device=device)
#         self.laplacian_kernel = laplacian_kernel.repeat(in_channels, 1, 1, 1, 1)

#     def forward(self, x):
#         # Applying convolution with padding to maintain the dimensions
#         padding = 1
#         x = nnf.conv3d(x, self.laplacian_kernel, padding=padding, groups=self.in_channels)
#         return x

# class EdgeDetectionModule3D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(EdgeDetectionModule3D, self).__init__()
#         self.edge_conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
#         # Initialize with a simple edge detection kernel; for more complex initialization,
#         # you might want to look into 3D edge detection kernels or design your own.
#         # Here we use a basic form where we focus on the central slice.
#         edge_kernel = torch.tensor([[[[0, 0, 0], [0, -1, 0], [0, 0, 0]],
#                                      [[0, -1, 0], [-1, 6, -1], [0, -1, 0]],
#                                      [[0, 0, 0], [0, -1, 0], [0, 0, 0]]]], dtype=torch.float32)
#         self.edge_conv.weight = nn.Parameter(edge_kernel.repeat(out_channels, in_channels, 1, 1, 1))

#     def forward(self, x):
#         edge_features = self.edge_conv(x)
#         return edge_features
class EdgeDetectionModule3D(nn.Module):
    def __init__(self, in_channels, out_channels=16):  # Increased to 32 for initial diversity
        super(EdgeDetectionModule3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        nn.init.kaiming_uniform_(self.edge_conv.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
        self.scale = nn.Parameter(torch.ones(out_channels, 1, 1, 1, 1))

    def forward(self, x):
        edge_features = self.edge_conv(x)
        edge_features = edge_features * self.scale
        # Here we would add code to select 16 most diverse channels
        edge_features = edge_features[0,:]
        return edge_features[None,...]

    def initialize_custom_kernel(self):
        edge_kernel = torch.tensor([[[[0, 0, 0], [0, -1, 0], [0, 0, 0]],
                                     [[0, -1, 0], [-1, 6, -1], [0, -1, 0]],
                                     [[0, 0, 0], [0, -1, 0], [0, 0, 0]]]], dtype=torch.float32)
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                self.edge_conv.weight.data[i, j] = edge_kernel * (1 + 0.1 * torch.randn_like(edge_kernel))



class DenseFeatureFusionBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DenseFeatureFusionBlock, self).__init__()
        # Define convolution layers with appropriate channel sizes
        self.conv1 = nn.Conv3d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(input_channels + 16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(input_channels + 16 + 16, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(input_channels + 16 + 16 + 16, output_channels, kernel_size=3, padding=1, stride=2)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # First convolution and concatenation
        conv1_out = self.relu(self.conv1(x))
        concat1 = torch.cat([x, conv1_out], dim=1)  # Concatenate along channel dimension

        # Second convolution and concatenation
        conv2_out = self.relu(self.conv2(concat1))
        concat2 = torch.cat([x, conv1_out, conv2_out], dim=1)

        # Third convolution and concatenation
        conv3_out = self.relu(self.conv3(concat2))
        concat3 = torch.cat([x, conv1_out, conv2_out, conv3_out], dim=1)

        # Final convolution
        output = self.conv4(concat3)
        return output

class ResUnet(nn.Module):
    def __init__(self, shape, channel, filters=[16, 16, 32, 32, 64]):
        super(ResUnet, self).__init__()
        
        

        self.input_layer = nn.Sequential(
            nn.Conv3d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(filters[0]),
            nn.ReLU(),
            nn.Conv3d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv3d(channel, filters[0], kernel_size=3, padding=1)
        )


        self.edge_conv1 = EdgeDetectionModule3D(filters[0], filters[0])
        self.edge_conv1.initialize_custom_kernel()
        self.dense_block1 = DenseFeatureFusionBlock(16,32)





        self.dense_block2 = DenseFeatureFusionBlock(32,32)
        
        self.dense_block3 = DenseFeatureFusionBlock(32,32)
        
        

        

        self.aspp_bridge = ASPP(filters[3], filters[3])


        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[3], filters[3], 1, 1)

        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[3], filters[2], 1, 1)

        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

        self.aspp_out = ASPP(filters[1], filters[0], rate=[6, 12, 18])

        self.output_layer = nn.Sequential(nn.Conv3d(filters[0], 3, 1))
        
        self.transformer = SpatialTransformer(shape)

    def forward(self, x):
        # x_00 = x[:,0] ; x_00 = x_00[None,...]
        # x_01 = x[:,1] ; x_01 = x_01[None,...]
        
        # x_lap_0 = self.edge_conv(x_00)
        # x_lap_1 = self.edge_conv(x_01)      
        # x = torch.cat((x_lap_0, x_lap_1),dim=1)
        
        x1 = self.input_layer(x) + self.input_skip(x)
        
        x2_e = self.edge_conv1(x1) 
        x2 = self.dense_block1(x2_e)
        x2_sm = nnf.avg_pool3d(x2,kernel_size=3, stride=1, padding=1)


        x3_1 = self.dense_block2(x2_sm)
        x3_sm = nnf.avg_pool3d(x3_1,kernel_size=3, stride=1, padding=1)

        
        x4_1 = self.dense_block3(x3_sm)
        x4_sm = nnf.avg_pool3d(x4_1,kernel_size=3, stride=1, padding=1)

        


        x5 = self.aspp_bridge(x4_sm) # 64,20,24,28


        x6 = self.upsample1(x5)
        x6 = torch.cat([x6, x3_sm], dim=1)
        x6 = self.up_residual_conv1(x6)


        x7 = self.upsample2(x6)
        x7 = torch.cat([x7, x2_sm], dim=1)
        x7 = self.up_residual_conv2(x7)


        x8 = self.upsample3(x7)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        flow = self.output_layer(x9)
        
        moving = x[:, 0:1, :, :]
        y = self.transformer(moving, flow)
                
        
        
        

        # return 
        return y, flow
        
        
  

#%%
# import pickle
# from torchvision import transforms
# import matplotlib.pyplot as plt

# def pkload(fname):
#     with open(fname, 'rb') as f:
#         return pickle.load(f)

# device=torch.device('cuda:0')

# pth = 'C:/Drive/Workspace/data_sets/Ruth dataset/temp/subject1.pkl'


# def read_pkl(pth):
#     p, h = pkload(pth)

#     p, h = p[None, ...], h[None, ...]
#     p = np.ascontiguousarray(p)# [Bsize,channelsHeight,,Width,Depth]
#     h = np.ascontiguousarray(h)
#     p, h = p[None, ...], h[None, ...]

#     p, h = torch.from_numpy(p), torch.from_numpy(h)
    
#     return p, h

# h, p = read_pkl(pth)
# x_in = torch.cat((h.float(),p.float()),dim=1)
# # cat_tensor = torch.tensor(cat, device=device, dtype=torch.float)
# vol_size=(160,192,224)
# net = ResUnet(vol_size,2)
# net.cuda(device)

'''
for layers
'''
# con = net(x_in.cuda(device))
# print(con.shape)
# # con1 = con.detach().cpu().numpy()[0,7]
# # plt.imshow(con1[10,:,:])

# ab = con[0,:,:,:,:]
# ab = ab.detach().cpu().numpy()

# fig, axes = plt.subplots(4, 4, figsize=(10,8))
# axes = axes.ravel() 
# for idx in range (ab.shape[0]):
#     ax = axes[idx]
#     ax.imshow(ab[idx,40,:,:], cmap='bone')
#     ax.axis("off")
# fig.suptitle("block (16 channels) (16,160,192,224)")

# plt.savefig('plots/block1_features.jpg', dpi=300)

# '''
# for full network
# '''
# aff, scale, transl, shear = net(x_in.cuda(device))
# affine_trans = AffineTransform()
# trans_vol, mat, inv_mat = affine_trans(h.float().cuda(device), aff, scale, transl, shear)

# trans_vol1 = trans_vol.detach().cpu().numpy()[0,0]
# p1 = p.detach().cpu().numpy()[0,0]

# plt.imshow(p1[50,:,:])
# plt.imshow(trans_vol1[50,:,:], 'jet', interpolation='none', alpha=0.5)    

    #%%
    
# path= "C:/Drive/Workspace/data_sets/normalized_selected/atlas/20.nii"
# device=torch.device('cuda:0')
# vol_size=(160,192,224)
# # enc_f1 = (16, 32, 32, 32)
# # dec_f1 = (32, 32, 16, 16)
# atlas_vol = nib.load(path).get_fdata()
# atlas_vol = np.expand_dims(atlas_vol, 0)
# atlas_vol = np.expand_dims(atlas_vol, 0)
# net = ResUnet(vol_size,2)
# net.cuda(device)
# cat = np.concatenate([atlas_vol, atlas_vol], axis=1)
# # #print(net)
# atlas_tensor = torch.tensor(cat, device=device, dtype=torch.float)
# atlas_volh = torch.tensor(atlas_vol, device=device, dtype=torch.float)
# aff, scale, transl, shear = net(atlas_tensor)

# affine_trans = AffineTransform()
# trans_vol, mat, inv_mat = affine_trans(atlas_volh, aff, scale, transl, shear)

# trans_vol = trans_vol.detach().cpu().numpy()[0]
# plt.imshow(trans_vol[0,:,:,90])

# print(con[0].shape)

# import matplotlib.pyplot as plt
# con_vol = con[0].detach().cpu().numpy()[0]
# con_mat = con[1].detach().cpu().numpy()[0]
# con_mat_inv = con[2].detach().cpu().numpy()[0]


#%%
