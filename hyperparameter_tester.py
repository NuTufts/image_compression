import os, sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
import tensorflow_datasets as tfds
import datasets.larcv
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img

import pathlib
sys.path.append('LArTPC-VQVAE/')

from setup_model import build_vqvae
from argparser import train_parser

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# sys.path.append('/image_compression/')
from compressor import encode, decode, images_as_tensor_blocks, images_as_tensor_blocks_files_given, encode_image_files
from stitcher import *
import copy
import random


config = {'model': 'res', 'checkpoint': '',
          'MNIST': False, 'save_root': '/train_save',
          'dataset': 256, 'sample_size': 8, 'gpu': 2, 
          'multi_gpu': False, 'shuffle': True, 
          'drop_last': False, 'num_workers': 8, 
          'k': 512, 'd': 64, 'beta': 1.0, 
          'vqvae_batch_size': 15, 'vqvae_epochs': 50, 
          'vqvae_lr': 0.0003, 'vqvae_layers': [32, 64], 
          'pcnn_batch_size': 256, 'pcnn_epochs': 15, 
          'pcnn_lr': 0.001, 'pcnn_blocks': 3, 
          'pcnn_features': 512}


os.environ["CUDA_VISIBLE_DEVICES"]=str(config['gpu'])
configProt = ConfigProto()
configProt.gpu_options.allow_growth = True
session = InteractiveSession(config=configProt)

directory = '/image_compression/results/originals/originals_png/'
train_split = 0.8
total_images = []
for i, file in enumerate(os.listdir(directory)): # read back in those saved images
    if file.endswith('.png'):
        total_images.append((directory + file, i))
random.shuffle(total_images)

total_images = total_images[:int(len(total_images)*0.5)]

train_files = total_images[:int(len(total_images)*train_split)]
test_files = total_images[int(len(total_images)*train_split):]
ds_train = images_as_tensor_blocks_files_given([file for file, i in train_files], verbose=True)
ds_test = images_as_tensor_blocks_files_given([file for file, i in test_files], verbose=True)


curr_best = {"config": config, 'mse': float('inf'), 'loss': None}
def mse_images(original, reproduced):
    # print(original.shape)
    # print(reproduced.shape)
    # mse = 0
    # for i in range(len(original)):
    #     mse += ((reproduced[i] - original[i])**2).mean()
    #     print((reproduced[i]-original[i]).mean())
    # mse = mse / len(original)
    # print(mse)
    return np.array(list(map(lambda x: ((x[0] - x[1])**2).mean(), zip(original, reproduced)))).mean()

def train_and_decode(config, save=None, vqvae_compare=True, return_images=False):
    vqvae, vqvae_sampler, encoder, decoder, codes_sampler, get_vqvae_codebook = build_vqvae(config)
    vqvae.summary()
    
    history = vqvae.fit(x=ds_train, y=ds_train, epochs=config['vqvae_epochs'], 
                        batch_size=config['vqvae_batch_size'],#  verbose=2) 
                        validation_data=(ds_test, ds_test), verbose=2)

    vqvae_codebook = get_vqvae_codebook()
    encode_image_files(encoder, [file for file, i in test_files], '/image_compression/results/codes/codes_npz.npz', verbose=True)
    if save is None:
        decoded_imgs = decode(decoder, vqvae_codebook, '/image_compression/results/codes/codes_npz.npz', code_sampler=codes_sampler, verbose=True)
    else:
        decoded_imgs = decode(decoder, vqvae_codebook, '/image_compression/results/codes/codes_npz.npz', code_sampler=codes_sampler, verbose=True,
                              save=save)

    mseloss = mse_images(stitch_nblocks_1d(np.reshape(ds_test, [-1, 256, 256]), 1008, 3456), decoded_imgs)

    if vqvae_compare:
        global curr_best
        if mseloss < curr_best['mse']:
            curr_best = {'config': copy.deepcopy(config), 'mse': mseloss, 'loss': history.history['loss'], 'val_loss': history.history['val_loss']}

    if return_images:
        return mseloss, decoded_imgs

    return mseloss


mselosses = []
def test_parameter(name, options, config):
    curr_best = (options[0], float('inf'))
    orig = config[name]
    global mselosses
    for option in options:
        print("Option: {} {}".format(name, option))
        config[name] = option
        mse = train_and_decode(config)
        mselosses.append(mse)
        if mse < curr_best[1]:
            curr_best = (option, mse)
    config[name] = orig
    return curr_best[0]

def difference_images(original, reproduced, save=''):
    differences = np.array(list(map(lambda x: np.abs(x[0] - x[1]), zip(original, reproduced))))
    if save: 
        for i, img in enumerate(differences):
            save_img(f"{save}diff_image_{i+1}.jpeg", img, data_format='channels_last')
    return differences

# # # test parameters
# best_parameters = {}
# # test latent code dimension vectors
# dimensions = [16, 32, 64]
# best_parameters['d'] = test_parameter('d', dimensions, config)

# # test batch size
# batch_sizes = [25, 50]
# best_parameters['batch_size'] = test_parameter('vqvae_batch_size', batch_sizes, config)

# # number of embedding vectors
# embedding_vectors = [256, 512]
# best_parameters['k'] = test_parameter('k', embedding_vectors, config)

# # vqvae layers
# layers = [[16, 32], [64, 128]]
# best_parameters['vqvae_layers'] = test_parameter('vqvae_layers', layers, config)

# print(best_parameters)
# print(mselosses)
# print(curr_best)



# compare vqvae vs trad
from PIL import Image
output_directory = '/image_compression/results/comparison/standard_compression/'
originals_directory = '/image_compression/results/comparison/originals/'

mse_trad_orig = 0
compressed_images = None
originals = None
for i, file in enumerate([file for file, i in test_files]):
    img = Image.open(file)
    img.save(f"{output_directory}jpeg_compression_{i+1}.jpeg", optimize=True, quality=50)
    img.save(f"{originals_directory}original_{i+1}.png")

    orig = img_to_array(load_img(file, color_mode='grayscale'))
    compressed = img_to_array(load_img(f"{output_directory}jpeg_compression_{i+1}.jpeg", color_mode='grayscale'))
    if compressed_images is None:
        compressed_images = np.array([compressed])
        originals = np.array([orig])
    else:
        compressed_images = np.append(compressed_images, np.reshape(compressed, (1, compressed.shape[0], compressed.shape[1], 1)), axis=0)
        originals = np.append(originals, np.reshape(orig, (1, orig.shape[0], orig.shape[1], 1)), axis=0)
    mse_trad_orig += ((compressed - orig)**2).mean()
mse_trad_orig = mse_trad_orig / len(test_files)

mse_vqvae_orig, decoded_imgs = train_and_decode(config, save='/image_compression/results/comparison/vqvae_compression/', vqvae_compare=True, return_images=True)
print(curr_best)
print(f"VQVAE MSE: {mse_vqvae_orig}, Trad compression MSE: {mse_trad_orig}")

standard_compression_diff = difference_images(originals, compressed_images, save='/image_compression/results/comparison/standard_differences/') # save diff images
vqvae_compression_diff = difference_images(originals, np.reshape(decoded_imgs, originals.shape), save='/image_compression/results/comparison/vqvae_differences/') # save vqvae diff images
difference_images(compressed_images, np.reshape(decoded_imgs, originals.shape), save='/image_compression/results/comparison/standard_vqvae_differences/') # save difference between vqvae and standard compression images



# overlay ~ with color ~
'''
def images_to_rgb(images):
    rgb = np.reshape(images, (images.shape[0], images.shape[1], images.shape[2], 1))
    return np.repeat(rgb.astype(np.uint8), 3, axis=3)

def color_filter(color, num_images, rows=1008, cols=3456):
    filt = [1, 1, 1]
    if color == 'red': 
        filt = [10, 1, 1]
    if color == 'blue': 
        filt = [1, 1, 10]
    if color == 'green': 
        filt = [1, 10, 1]
    filter_ray = np.repeat(np.array([filt]), rows*cols,axis=0).reshape((rows, cols, 3))
    filter_ray = np.array([filter_ray])
    filter_ray = np.repeat(filter_ray, num_images, axis=0)
    return filter_ray

import copy
def overlay(originals, reproduced, color, save=''):
    reproduced_rgb = copy.deepcopy(reproduced)
    originals_rgb = images_to_rgb(originals)
    reproduced_rgb = images_to_rgb(reproduced)
    filter_layer = color_filter(color, originals_rgb.shape[0])
    red_layer = color_filter('red', originals_rgb.shape[0])
    for i in range(len(reproduced_rgb)):
        reproduced_rgb[i] = (reproduced_rgb[i] / reproduced_rgb[i].max()) * 255 if (reproduced_rgb[i].max() != 0) else reproduced_rgb[i]
        originals_rgb[i] = (originals_rgb[i] / originals_rgb[i].max()) * 255 if (originals_rgb[i].max() != 0) else originals_rgb[i]

    reproduced_rgb = reproduced_rgb * filter_layer
    originals_rgb = originals_rgb * red_layer

    if save:
        for i in range(len(originals_rgb)):
            background = originals_rgb[i]
            foreground = reproduced_rgb[i]
            # background = (originals_rgb[i] / originals_rgb[i].max()) * 255 if (originals_rgb[i].max() != 0) else originals_rgb[i]
            # foreground = (reproduced_rgb[i] / reproduced_rgb[i].max()) * 255 if (reproduced_rgb[i].max() != 0) else reproduced_rgb[i]
            img = background + foreground

            # background = Image.fromarray(background.astype('uint8'))
            # foreground = Image.fromarray(foreground.astype('uint8'))

            # foreground.paste(background)
            # img = np.asarray(foreground)
            save_img(f"{save}diff_overlay_{i+1}.jpeg", img, data_format='channels_last')


overlay(originals, standard_compression_diff, 'green', save='/image_compression/results/comparison/standard_differences/overlay/')
overlay(originals, vqvae_compression_diff, 'green', save='/image_compression/results/comparison/vqvae_differences/overlay/')
'''
