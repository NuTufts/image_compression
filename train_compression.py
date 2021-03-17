import os, sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
import tensorflow_datasets as tfds
import pathlib

sys.path.append('../LArTPC-VQVAE/')
from setup_model import build_vqvae
from argparser import train_parser
import datasets.larcv

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from compressor import encode, decode, images_as_tensor_blocks, images_as_tensor_blocks_files_given


# import dataset
# import config, set mnist option
def train(config): 

    batch_size = config['vqvae_batch_size']
    directory = '/image_compression/data/model_data/jpeg/'
    data_dir = pathlib.Path(directory)
    image_count = len(list(data_dir.glob('*/*.jpeg')))
    print(image_count)
    exit()

    # ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    #     data_dir,
    #     subset='training',
    #     validation_split=0.2,
    #     seed=123,
    #     image_size=(256, 256),
    #     batch_size=image_count,
    #     color_mode='grayscale',
    #     label_mode=None
    # )
    # ds_test = tf.keras.preprocessing.image_dataset_from_directory(
    #     data_dir,
    #     validation_split=0.2,
    #     subset="validation",
    #     seed=123,
    #     image_size=(256, 256),
    #     batch_size=image_count,
    #     color_mode='grayscale',
    #     label_mode=None
    # )

    # ds_train = np.array([i.numpy() for i in ds_train])
    # ds_train = ds_train[0]  # takes out the single batch from produced ds_train
    # ds_train = np.array([i / i.max() * 10 if i.max() != 0 else i for i in ds_train ])
    # ds_train = np.reshape(ds_train, [-1, 256, 256, 1]).astype(float)
    
    # ds_test = np.array([i.numpy() for i in ds_test])
    # ds_test = ds_test[0]    # takes out the single batch from produced ds_test
    # ds_test = np.array([i / i.max() * 10 if i.max() != 0 else i for i in ds_test ])
    # ds_test = np.reshape(ds_test, [-1, 256, 256, 1]).astype(float)

    # print(ds_test.shape)
    # print(ds_test[0].shape)

    import random
    directory = '/image_compression/results/originals/originals_png/'
    train_split = 0.8
    total_images = []
    for i, file in enumerate(os.listdir(directory)): # read back in those saved images
        if file.endswith('.png'):
            total_images.append((directory + file, i))
    random.shuffle(total_images)

    total_images = total_images[:int(len(total_images)*0.3)]

    train_files = total_images[:int(len(total_images)*train_split)]
    test_files = total_images[int(len(total_images)*train_split):]
    ds_train = images_as_tensor_blocks_files_given([file for file, i in train_files], verbose=True)
    ds_test = images_as_tensor_blocks_files_given([file for file, i in test_files], verbose=True)

    print(ds_test.shape)
    print(ds_test[0].shape)
    
    vqvae, vqvae_sampler, encoder, decoder, codes_sampler, get_vqvae_codebook = build_vqvae(config)
    vqvae.summary()
    vqvae_codebook = get_vqvae_codebook()

    history = vqvae.fit(x=ds_train, y=ds_train, epochs=config['vqvae_epochs'], 
                        batch_size=config['vqvae_batch_size'],
                        validation_data=(ds_test, ds_test), verbose=1)
    print("Train loss")
    print(history.history['loss'])
    print("Val loss")
    print(history.history['val_loss'])

    # vqvae_codebook = get_vqvae_codebook()
    # to_compress = images_as_tensor_blocks("/image_compression/results/originals/originals_png/")
    # encode(encoder, to_compress, '/image_compression/results/codes/codes_npz.npz',k=config['k'], verbose=True)
    # # encode(encoder, to_compress, '/image_compression/results/codes/codes_npz.jpeg')
    # decode(decoder, vqvae_codebook, '/image_compression/results/codes/codes_npz.npz', code_sampler=codes_sampler, save="/image_compression/results/reproduced/reproduced/", verbose=True)

    print('Fin')


def main():
    parser = train_parser()
    config = vars(parser.parse_args())

    os.environ["CUDA_VISIBLE_DEVICES"]=str(config['gpu'])
    configProt = ConfigProto()
    configProt.gpu_options.allow_growth = True
    session = InteractiveSession(config=configProt)

    train(config)

if __name__ == '__main__':
    main()

