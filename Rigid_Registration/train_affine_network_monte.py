from tensorboardX import SummaryWriter
import os, utils, glob, losses, random, math
import sys
from torch.utils.data import DataLoader
import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from natsort import natsorted
from arch.New_affine_net_id2_1_Monte import ResUnet as resnet
import arch.New_affine_net_id2_1_Monte as affine_net
import torch.nn as nn
import losses




class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main():
    batch_size = 1
    train_dir = '/scratch/ahsan/1h_31p_registration/Train_data_05022024/'
    save_dir = 'NEW_MyAffine_MI/'
    if not os.path.exists('experiments/'+save_dir):
        os.makedirs('experiments/'+save_dir)
    if not os.path.exists('logs/'+save_dir):
        os.makedirs('logs/'+save_dir)
    sys.stdout = Logger('logs/'+save_dir)
    lr = 0.1 # learning rate
    epoch_start = 0
    max_epoch = 500 #max traning epoch
    cont_training = False #if continue training

    '''
    Initialize model
    '''
    vol_size = (160, 192, 224)


    model = resnet(vol_size, 2)
    model.cuda()
    
    affine_trans = affine_net.AffineTransform()


    


    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 201
        model_dir = 'experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-1]))
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([trans.RandomFlip(0),
                                          trans.NumpyType((np.float32, np.float32)),
                                          ])

    train_set = datasets.ClinicalDataset(glob.glob(train_dir + '*.pkl'), transforms=train_composed)

    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    criterion = losses.MutualInformation()
    kernal_array= []
    edge_image_array = []
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            data = [t.cuda() for t in data]
            h = data[0]
            p = data[1]
            # x_ = affine_aug(pat_img, seed=idx)
            x_in = torch.cat((h,p),dim=1)
            
            aff, scale, transl, shear, kernal, edge_image  = model(x_in)
            if idx==1:
                kernal_array.append(kernal.detach().cpu().numpy()) ; edge_image_array.append(edge_image.detach().cpu().numpy())
            trans_vol, mat, inv_mat  = affine_trans(h, aff, scale, transl, shear)
            loss = criterion(trans_vol, p)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}'.format(idx, len(train_loader), loss.item()))
        print('Epoch {}'.format(epoch))
    torch.save(model.state_dict(), 'experiments/ID2_1_Monte_ep500.pth.tar')
    kernal_array = np.array(kernal_array)
    edge_image_array = np.array(edge_image_array)
    np.save('experiments/kernal.npy', kernal_array)
    np.save('experiments/edge_image.npy', edge_image_array)



def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

if __name__ == '__main__':
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
    main()