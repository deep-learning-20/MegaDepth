import torch
import sys
from torch.autograd import Variable
import numpy as np
from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
from data.data_loader import CreateDataLoader
from models.models import create_model
from skimage import io
from skimage.transform import resize
from time import perf_counter
import os
import glob
import time

img_dir = './'
img_name = 'tokyo'
img_path = img_dir+'/'+img_name+'.jpeg'

model = create_model(opt)

input_height = 384
input_width  = 512

dataset_dir = './datasets/hpatches-sequences-release'


def calc_hpatches_depth(model):
    model.switch_to_eval()
        
    sequence_folders = os.listdir(dataset_dir)
    for sequence_folder in sequence_folders:
        print('Processing Sequence: '+sequence_folder)
        sequence_path = os.path.join(dataset_dir, sequence_folder)
        
        for img_path in glob.glob(sequence_path+'/*.ppm'):

            name = os.path.basename(img_path)
            clean_name = name.split('.')[0]
            new_img_path = os.path.join(sequence_path, clean_name+'d.ppm')
            
            img = np.float32(io.imread(img_path))/255.0
            original_shape = img.shape 
            img = resize(img, (input_height, input_width), order = 1)
            
            input_img =  torch.from_numpy( np.transpose(img, (2,0,1)) ).contiguous().float()
            input_img = input_img.unsqueeze(0)

            input_images = Variable(input_img.cuda() )
            
            pred_log_depth = model.netG.forward(input_images) 
            pred_log_depth = torch.squeeze(pred_log_depth)

            pred_depth = torch.exp(pred_log_depth)

            # visualize prediction using inverse depth, so that we don't need sky segmentation (if you want to use RGB map for visualization, \
            # you have to run semantic segmentation to mask the sky first since the depth of sky is random from CNN)
            pred_inv_depth = 1/pred_depth
            pred_inv_depth = pred_inv_depth.data.cpu().numpy()
            # you might also use percentile for better visualization
            pred_inv_depth = pred_inv_depth/np.amax(pred_inv_depth)
            
            pred_inv_depth_resized = resize(pred_inv_depth, original_shape[0:2], order = 1)
            io.imsave(new_img_path, pred_inv_depth_resized)

            sys.exit()


calc_hpatches_depth(model)
print("We are done")
