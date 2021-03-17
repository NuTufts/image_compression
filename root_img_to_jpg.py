import os
import numpy as np
import argparse

import ROOT
from ROOT import TChain
from larcv import larcv
from larlite import larlite
from larflow import larflow

import torch
import torch.utils.data
import torchvision
from torchvision import datasets, transforms 
from torchvision.utils import save_image
from stitcher import *

def convert(input_directory, block_size=256):
    input_files = []
    for file in os.listdir(input_directory):
        if file.endswith(".root"):
            input_files.append(os.path.join(input_directory, file))

    blocks = []
    for filenum, file in enumerate(input_files):
        ROOT.TFile.Open(file).ls()
        chain_image2d = ROOT.TChain('image2d_wire_tree')
        chain_image2d.AddFile(file)
        num_events = chain_image2d.GetEntries()
        print(file)
        print(num_events, 'entries found')

        img_num = 0
        for idx in range(num_events):
            print("Event {}".format(idx))
            chain_image2d.GetEntry(idx)
            cpp_object = chain_image2d.image2d_wire_branch
            image2d_array = cpp_object.as_vector()
            for plane in image2d_array:
                ## Get the image data out as a 1D vector, then reshape to 2D.
                ## A little jank, reaching in to the ROOT classes directly but 
                ## for some reason `torch.from_numpy(larcv.as_ndarray(plane))`
                ## was segfaulting and I had no mind to fix it
                rows, cols = plane.meta().rows(), plane.meta().cols()
                img = np.asarray(plane.as_vector())
                img = np.reshape(img, (cols, rows))
                img = img.transpose()
                img_blocks = partition_image_1d(img, 256)
                blocks += img_blocks
    return blocks

def extract_root_to_jpg(input_files, output_directory):
    print(len(input_files))
    planes = 0
    img_num = 0
    for filenum, file in enumerate(input_files):
        ROOT.TFile.Open(file).ls()
        chain_image2d = ROOT.TChain('image2d_wire_tree')
        chain_image2d.AddFile(file)
        num_events = chain_image2d.GetEntries()
        print(file)
        print(num_events, 'entries found')
        
        for idx in range(num_events):
            print("Event {}".format(idx))
            chain_image2d.GetEntry(idx)
            cpp_object = chain_image2d.image2d_wire_branch
            image2d_array = cpp_object.as_vector()
            for plane in image2d_array:
                planes += 1
                ## Get the image data out as a 1D vector, then reshape to 2D.
                ## A little jank, reaching in to the ROOT classes directly but 
                ## for some reason `torch.from_numpy(larcv.as_ndarray(plane))`
                ## was segfaulting and I had no mind to fix it
                rows, cols = plane.meta().rows(), plane.meta().cols()
                img = np.asarray(plane.as_vector())
                img = np.reshape(img, (cols, rows))
                img = img.transpose()
                img_name = output_directory + '/original_{}.jpeg'.format(img_num)
                torchvision.utils.save_image(torch.from_numpy(img), img_name, normalize=True)
                img_num += 1
    print(img_num, planes)

def convert_and_save(input_files, output_directory, block_size=256):

    print(file)
    print(num_events, 'entries found')  
    for filenum, file in enumerate(input_files):
        ROOT.TFile.Open(file).ls()
        chain_image2d = ROOT.TChain('image2d_wire_tree')
        chain_image2d.AddFile(file)
        num_events = chain_image2d.GetEntries()
        
        img_num = 0
        for idx in range(num_events):
            print("Event {}".format(idx))
            chain_image2d.GetEntry(idx)
            cpp_object = chain_image2d.image2d_wire_branch
            image2d_array = cpp_object.as_vector()
            for plane in image2d_array:
                ## Get the image data out as a 1D vector, then reshape to 2D.
                ## A little jank, reaching in to the ROOT classes directly but 
                ## for some reason `torch.from_numpy(larcv.as_ndarray(plane))`
                ## was segfaulting and I had no mind to fix it
                rows, cols = plane.meta().rows(), plane.meta().cols()

                img = np.asarray(plane.as_vector())
                img = np.reshape(img, (cols, rows))
                img = img.transpose()
                img_blocks = partition_image_1d(img, 256)
                for block in img_blocks:
                    img_name = output_directory + '/larcv_256_{}.jpeg'.format(img_num)
                    torchvision.utils.save_image(torch.from_numpy(block), img_name, normalize=True)
                    img_num += 1

def main():
    parser = argparse.ArgumentParser(description='Convert a folder of root images to jpgs')
    parser.add_argument('-i', '--input-directory', type=str, help="Directory to ROOT images.")
    parser.add_argument('-o', '--output-directory', type=str, help="Directory to save JPG images.")
    parser.add_argument('-b', '--block-size', type=int, default=256, help="Block size to split images into")
    args = parser.parse_args()

    files = []
    for file in os.listdir(args.input_directory):
        if file.endswith(".root"):
            files.append(os.path.join(args.input_directory, file))

    # convert_and_save(files, args.output_directory, block_size=args.block_size)
    extract_root_to_jpg(files, args.output_directory)

if __name__ == "__main__":
    main()