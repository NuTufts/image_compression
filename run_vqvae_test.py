import os, sys
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from compressor import encode, decode, images_as_tensor_blocks, images_as_tensor_blocks_files_given, encode_image_files
from stitcher import *
import copy
import random

import pathlib
sys.path.append('LArTPC-VQVAE/')

from setup_model import build_vqvae
from argparser import train_parser

from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

config = {'model': 'res', 'checkpoint': '',
          'MNIST': False, 'save_root': '/train_save',
          'dataset': 256, 'sample_size': 8, 'gpu': 0, 
          'multi_gpu': False, 'shuffle': True, 
          'drop_last': False, 'num_workers': 8, 
          'k': 512, 'd': 64, 'beta': 1.0, 
          'vqvae_batch_size': 17, 'vqvae_epochs': 1, 
          'vqvae_lr': 0.0003, 'vqvae_layers': [32, 64], 
          'pcnn_batch_size': 256, 'pcnn_epochs': 15, 
          'pcnn_lr': 0.001, 'pcnn_blocks': 3, 
          'pcnn_features': 512}

os.environ["CUDA_VISIBLE_DEVICES"]=str(config['gpu'])


# DATA PREPARATION 
base_directory = ''
directory = base_directory + '/image_compression/results/originals/originals_png/'
train_split = 0.8
total_images = []
for i, file in enumerate(os.listdir(directory)): # read back in those saved images
    if file.endswith('.png'):
        total_images.append((directory + file, i))
    if i == 1: break

random.shuffle(total_images)
train_files = total_images[:int(len(total_images)*train_split)]
test_files = total_images[int(len(total_images)*train_split):]
ds_train = images_as_tensor_blocks_files_given([file for file, i in train_files], verbose=True)
ds_test = images_as_tensor_blocks_files_given([file for file, i in test_files], verbose=True)


# model instantiation and training
vqvae, vqvae_sampler, encoder, decoder, codes_sampler, get_vqvae_codebook = build_vqvae(config)
vqvae.summary()

history = vqvae.fit(x=ds_train, y=ds_train, epochs=config['vqvae_epochs'], 
                    batch_size=config['vqvae_batch_size'],  verbose=1) 
                    # validation_data=(ds_test, ds_test), verbose=2)
            

# import os, signal; os.kill(os.getpid(), signal.SIGTRAP)
# vqvae_codebook = get_vqvae_codebook()
# encode_image_files(encoder, [file for file, i in test_files], base_directory+'/image_compression/results/codes/codes_npz.npz', verbose=True)
# decoded_imgs = decode(decoder, vqvae_codebook, base_directory+'/image_compression/results/codes/codes_npz.npz', code_sampler=codes_sampler,
#                                     verbose=True, save=base_directory+'/image_compression/results/comparison/vqvae_compression/')

# # vqvae.save(base_directory+'models/'+datetime.now().strftime('%m-%d-%Y-%H:%M')+'.vqvae')

# def mse_images(original, reproduced):
#     return np.array(list(map(lambda x: ((x[0] - x[1])**2).mean(), zip(original, reproduced)))).mean()

# print(mse_images(stitch_nblocks_1d(np.reshape(ds_test, [-1, 256, 256]), 1008, 3456), decoded_imgs))