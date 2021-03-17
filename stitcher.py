# Stitcher.py
# Functions to split images into square blocks and recombine them
# Author: Jared Hwang 2021
import numpy as np
import math

# partition_image
# Splits a 2D array and paritions it into square blocks
# Parameters: 2D Array->int, square block size->int, pad value->float
# Returns: a numpy array of shape (# num blocks in row direction, # blocks in col direction, blocksize, blocksize)
def partition_image(ray, block_size, default=0.0):
    if len(ray) == 0 or len(ray[0]) == 0: return np.array([])

    # Pad array with 0's
    origRows, origCols = ray.shape
    if origRows % block_size != 0 or origCols % block_size != 0:
        newRows = origRows + (block_size - origRows % block_size if origRows % block_size else 0)
        newCols = origCols + (block_size - origCols % block_size if origCols % block_size else 0)
        split = np.full((newRows, newCols), default)
        split[:origRows,:origCols] = ray

    rows, cols = split.shape
    split = np.vsplit(split, rows//block_size)
    split = np.array([np.hsplit(subblock, cols//block_size) for subblock in split])
    return split

# partition_image_1d
# Splits a 2D array and paritions it into square blocks, returns blocks flattened into 1d
# Parameters: 2D Array->int, square block size->int, pad value->float
# Returns: a numpy array of shape (# num blocks in row direction * # blocks in col direction, blocksize, blocksize)
def partition_image_1d(ray, block_size, default=0.0):
    blocks = partition_image(ray, block_size, default=default)
    blocks = blocks.reshape(blocks.shape[0]*blocks.shape[1], block_size, block_size)
    return blocks

# stitch_blocks
# Takes array of shape (rows, cols, blocksize, blocksize) and stitches the blocks together to make 2d image
#      rows and cols referring to number of blocks in row direction and col direction
# Parameters: the block array->float array, row size of the original image->int, col size of orig image->int
# Returns: a numpy array of shape (orig rowsize, orig colsize)
def stitch_blocks(blocks, origRows=None, origCols=None):    
    if len(blocks) == 0 or len(blocks[0]) == 0: return np.array([])

    ret = np.array([np.hstack(row) for row in blocks])
    ret = np.vstack(ret)

    if origRows is None: origRows = ret.shape[0]
    if origCols is None: origCols = ret.shape[1]
    if origRows == ret.shape[0] and origCols == ret.shape[1]:
        return ret
    else:
        return ret[:origRows, :origCols]

# stitch_blocks_1d
# Takes array of shape (rows*cols, blocksize, blocksize) and stitches blocks together to make 2d image
#       rows and cols referring to number of blocks in row direction and col direction
#       (therefore 1d as in it is a list of rows*cols blocks)
# Parameters: the 1d array of blocks->float array, row size of orig image->int, col size of orig image->int
# Returns: a numpy array of shape (orig rowsize, origw colsize)
def stitch_blocks_1d(blocks, origRows, origCols):
    if len(blocks) == 0: return np.array([])
    blockSize = blocks[0][0].shape[0]
    blockRows = math.ceil((origRows) / blockSize)
    blockCols = math.ceil((origCols) / blockSize)
    return stitch_blocks(blocks.reshape(blockRows, blockCols, blockSize, blockSize), origRows, origCols)

# stitch_nblocks_1d
# Takes array of shape (n, blocksize, blocksize) and stitches blocks together to make 2d images
# Parameters: the 1d array of blocks->float array, row size of orig image->int, col size of orig image->int
# Returns: a numpy array of shape (x, orig rowsize, orig colsize), x being the number of images
#          produced from the 1d array. 
def stitch_nblocks_1d(blocks, origRows, origCols):
    if len(blocks) == 0: return np.array([])
    blockSize = blocks[0][0].shape[0]
    blockRows = math.ceil((origRows) / blockSize)
    blockCols = math.ceil((origCols) / blockSize)
    numBlocksinImage = blockRows*blockCols
    ret = []
    begin, end = 0, numBlocksinImage
    while end <= len(blocks):
        ret.append(stitch_blocks(blocks[begin:end].reshape(blockRows, blockCols, blockSize, blockSize), origRows, origCols))
        begin += numBlocksinImage
        end += numBlocksinImage
    return np.array(ret)



# # testing
# a = np.arange(30).reshape([5,6])
# print(a)
# result = partition_image(a, 2)
# print(result)
# result = result.reshape((result.shape[0]*result.shape[1], 2, 2))
# print(result)
# result = stitch_blocks_1d(result, 5, 6)
# print(result)
# print(np.array_equal(a, result))

# a = np.arange(1008*3456).reshape((1008, 3456))
# result = stitch_blocks(partition_image(a, 64), origRows=1008, origCols=3456)
# print(np.array_equal(a, result))

# t1 = np.arange(25).reshape([5,5])
# t2 = np.arange(25, 50).reshape([5,5])
# print(t1)
# print(t2)
# result1 = partition_image_1d(t1, 2)
# result2 = partition_image_1d(t2, 2)
# # print(result1)
# # print(result2)
# tot = np.concatenate((result1, result2))
# restitched = stitch_nblocks_1d(tot, 5, 5)
# print(restitched[0])
# print(restitched[1])