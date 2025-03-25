# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 09:12:39 2022

@author: Ahsan
"""

'''
writing script for test bench of 5 methods and their comparision. 
following matrices will be compaired:
Dice, MI, number of Positive jacobians, histogram of displacment, trainable paramters, loss function landscape.
'''

#%%
'''
loading libraries
'''

import torch
# from models import VxmDense_1
#import os, losses, utils, utils1
from torch.utils.data import DataLoader
import datasets, trans
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
from torchvision import transforms
# import glob
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from arch.case4 import ResUnet as ResNet
# import numpy as np
# from natsort import natsorted
# from models.TransMorph import CONFIGS as CONFIGS_TM
# import models.TransMorph as TransMorph
from arch.Reg_LEdge_U_Model_Variant_4_monte import ResUnet as resnet

# from models1 import VxmDense_1, VxmDense_2, VxmDense_huge
print('\n\n Libraries are loaded')

#%%
'''
GPU configuration
'''
GPU_iden = 0
GPU_num = torch.cuda.device_count()
print('Number of GPU: ' + str(GPU_num))
for GPU_idx in range(GPU_num):
    GPU_name = torch.cuda.get_device_name(GPU_idx)
    print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
torch.cuda.set_device(GPU_iden)
GPU_avai = torch.cuda.is_available()
print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
print('If the GPU is available? ' + str(GPU_avai))

# def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
#     grid_img = np.zeros(grid_sz)
#     for j in range(0, grid_img.shape[1], grid_step):
#         grid_img[:, j+line_thickness-1, :] = 1
#     for i in range(0, grid_img.shape[2], grid_step):
#         grid_img[:, :, i+line_thickness-1] = 1
#     grid_img = grid_img[None, None, ...]
#     grid_img = torch.from_numpy(grid_img).cuda()
#     return grid_img

print('\n\n GPU configured')

#%%
'''
loading the model
'''

'''
MODEL 1: 
    Discription: 
'''
img_size = (160, 192, 224)
try:
    # model_1 =  VxmDense_1(img_size)
    model_1 =  resnet(img_size,2)
    model_1.load_state_dict(torch.load('/scratch/ahsan/1h_31p_registration/Non_rigid/src/experiments/NR_incep_monte_ep500.pth.tar', map_location='cuda:0'), strict=False)
    model_1.cuda()
    print('\n\n Model 1 has loaded')
except Exception as err:
    print(f"\n\n Oops! Model 1 could not be loaded: {err}, {type(err)}")

# '''
# MODEL 2: 
#     discription: 
# '''
# config = CONFIGS_TM['TransMorph']
# model_1 = TransMorph.TransMorph(config)
# model_1.load_state_dict(torch.load('experiments/transmorph_elastic_ep500.pth.tar', map_location='cuda:0'), strict=False)
# model_1.cuda()




print('\n\n Model 1 has loaded')    
  

'''
    Initialize spatial transformation function
'''
# reg_model = utils.register_model(img_size, 'nearest')
# reg_model.cuda()
# reg_model_bilin = utils.register_model(img_size, 'bilinear')
# reg_model_bilin.cuda()



#%%
'''
loading the test dataset
'''

'''
Model 1
'''
batch_size = 1
test_dir = '/scratch/ahsan/1h_31p_registration/temp/'


# test_composed = transforms.Compose([trans.RandomFlip(0),
#                                       trans.NumpyType((np.float32, np.float32)),
#                                       ])

test_set = datasets.ClinicalDatasettest(glob.glob(test_dir + '*.pkl'))


test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, pin_memory=True)
# simf = losses.SSIM3D()
# lmi_f = losses.localMutualInformation()
# eval_dsc = utils1.AverageMeter()

vols = []
num_sample = 10
for i in range(num_sample): 
    for data in test_loader:
        model_1.train()
        data = [t.cuda() for t in data]
        x = data[0]
        y = data[1]
        x_seg = data[2]
        y_seg = data[3]
        x_in = torch.cat((x, y), dim=1)
        output = model_1(x_in)
    
        vols.append(output[0].detach().cpu().numpy()[0, 0, :, :, :])

        
        


outputs = np.array(vols)
np.save('vols.npy', outputs)
