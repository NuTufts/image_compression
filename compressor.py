import os
import numpy as np
import pickle
from stitcher import *
import argparse

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.keras import backend as K
import tensorflow_datasets as tfds
import pathlib


# encodes blocks
def encode(encoder, input_data, output_file, k=512, verbose=False):
    to_save = None
    num_blocks = len(input_data)
    for i, block in enumerate(input_data):
        if verbose: print(f"Encoding: {i+1}/{num_blocks}", end='\r')
        if to_save is None:
            to_save = encoder.predict(np.reshape(block, [-1, 256, 256, 1]).astype(float))
        else:
            to_save = np.append(to_save, encoder.predict(np.reshape(block, [-1, 256, 256, 1]).astype(float)), axis=0)


    # savez_compressed
    with open(output_file, 'wb') as out_file:
        np.savez_compressed(out_file, to_save) 

    if k <= 256:
        to_save = to_save.astype(np.ushort)
        with open('/image_compression/results/codes/codes_npz_short.npz', 'wb') as out_file:
            np.savez_compressed(out_file, to_save) 

    # # save as jpeg (flattened)
    # og_shape = to_save.shape
    # save_img(output_file, np.reshape(to_save, (og_shape[0], og_shape[1]*og_shape[2])))

    print('Encoded.')

def encode_image_files(encoder, input_files, output_file, verbose=False):
    blocks = images_as_tensor_blocks_files_given(input_files, rows=1008, cols=3456, verbose=False)
    encode(encoder, blocks, output_file, verbose=verbose)

# decodes codes
def decode(decoder, codebook, input_file, code_sampler=None, origRows=1008, origCols=3456, save='', verbose=False):
    codes = np.load(input_file)['arr_0']
    num_codes, blocks = len(codes), None
    code_dimensions = codebook[0].shape[0]

    for i, code in enumerate(codes):
        if verbose: print(f"Decoding: {i+1}/{num_codes}", end='\r')
        code = tf.convert_to_tensor(code)
        if code_sampler is not None:
            code = code_sampler(code)
        else:
            code = np.reshape(codebook[np.reshape(code, (code.shape[0]*code.shape[1]))], (1, code.shape[0], code.shape[1], code_dimensions)) #extract indices from codebook
        block = decoder.predict(code)
        block = np.array(block)
        if blocks is None:
            blocks = block
        else:
            blocks = np.append(blocks, block, axis=0)
    print(blocks.shape)

    images = stitch_nblocks_1d(blocks, origRows, origCols)
    if save:
        img_num = 1
        for image in images:
            if verbose: print(f"Saving decoded image {img_num}", end='\r')
            img_name = save + 'decoded_{}.jpeg'.format(img_num)
            image = np.reshape(image, (image.shape[0], image.shape[1], 1))
            save_img(img_name, image, data_format='channels_last')
            img_num += 1

    print('\nDecoded.')
    return images

def images_as_tensor_blocks_files_given(input_files, rows=1008, cols=3456, verbose=False):
    num_images = len(input_files)
    print(f"Num images to compress: {num_images}")

    img_num = 1
    blocks = None
    for file in input_files:
        img = load_img(file,color_mode='grayscale')
        img = np.reshape(img_to_array(img), [rows,cols])
        
        img = (img / img.max()) * 10 if img.max() != 0 else img # normalize image
        img_blocks = partition_image_1d(img, 256)
        if blocks is None:
            blocks = img_blocks
        else:
            blocks = np.append(blocks, img_blocks, axis=0)
        if verbose: print(f'Processed {img_num}/{num_images}', end='\r')
        img_num += 1

    print("Images read in.")
    return blocks

def images_as_tensor_blocks(input_directory, rows=1008, cols=3456, verbose=False):
    input_files = []
    for file in os.listdir(input_directory):
        if file.endswith(".png"):
            input_files.append(os.path.join(input_directory, file))

    return images_as_tensor_blocks_files_given(input_files, rows=rows, cols=cols, verbose=verbose)


def images_as_tensors(input_directory, output_file,  rows=1008, cols=3456, verbose=False):
    input_files = []
    for file in os.listdir(input_directory):
        if file.endswith(".jpeg"):
            input_files.append(os.path.join(input_directory, file))

    num_images = len(input_files)
    print(f"Num images to save: {num_images}")

    img_num = 1
    images = None
    for file in input_files:
        img = load_img(file,color_mode='grayscale')
        img = np.reshape(img_to_array(img), [1, rows,cols])
        if images is None:
            images = img
        else:
            images = np.append(images, img, axis=0)
        if verbose: print(f'Processed {img_num}/{num_images}')
        img_num += 1

    print(images.shape)
    
    with open(output_file, 'wb') as out_file:
        # np.save(out_file, images)
        np.savetxt(out_file, images.flatten())
        # np.savez_compressed(out_file, images)

    print("Images read in.")
    return images



def main():
    parser = argparse.ArgumentParser(description='Compress and decompress a folder of root images using VQVAE')
    parser.add_argument('-i', '--input-directory', type=str, help="Directory to ROOT images.")
    parser.add_argument('-e', '--encoder', type=str, help="Path to VQVAE encoder")
    # parser.add_argument('-o', '--output-directory', type=str, help="Directory to save JPG images.")
    parser.add_argument('-b', '--block-size', type=int, default=256, help="Block size to split images into")
    args = parser.parse_args()

    images_as_tensors('/image_compression/results/originals/', '/image_compression/results/originals_npsave.txt', verbose=True)

    # files = []
    # for file in os.listdir(args.input_directory):
    #     if file.endswith(".root"):
    #         files.append(os.path.join(args.input_directory, file))

    # blocks = convert(files, block_size=args.block_size)
    # encoder = '' # temp
    # encode(encoder, blocks, 'compressed_test_data.npz')


if __name__ == "__main__":
    main()
