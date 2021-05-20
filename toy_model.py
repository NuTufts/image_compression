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

os.environ["CUDA_VISIBLE_DEVICES"]=str(1)
# configProt = ConfigProto()
# configProt.gpu_options.allow_growth = True
# session = InteractiveSession(config=configProt)

base_directory = ''
directory = base_directory + '/image_compression/results/originals/originals_png/'
train_split = 0.8
total_images = []
for i, file in enumerate(os.listdir(directory)): # read back in those saved images
    if file.endswith('.png'):
        total_images.append((directory + file, i))
    if i == 50: break

random.shuffle(total_images)

print('about to process')
train_files = total_images[:int(len(total_images)*train_split)]
test_files = total_images[int(len(total_images)*train_split):]
ds_train = images_as_tensor_blocks_files_given([file for file, i in train_files], verbose=True)
ds_test = images_as_tensor_blocks_files_given([file for file, i in test_files], verbose=True)
print('processed')

latent_dim = 500 
batch_size = 10
class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(65536, activation='sigmoid'),
      layers.Reshape((256, 256))
    ])

  def call(self, x):
    print('call start')
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    print('call end')
    return decoded

autoencoder = Autoencoder(latent_dim)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder.fit(ds_train, ds_train,
                epochs=10,
                shuffle=True,
                batch_size=batch_size,
                validation_data=(ds_test, ds_test),
                verbose=1)

# encoded_imgs = autoencoder.encoder(ds_test).numpy()
# decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
# print(encoded_imgs.shape)
# print(decoded_imgs.shape)