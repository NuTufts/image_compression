#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ../train_compression.py \
--gpu 1 \
--k 256 \
--d 32 \
--beta 1 \
--vqvae_batch_size 15 \
--vqvae_epochs 45 \
--vqvae_lr 3e-4 \
--vqvae_layers 32 64 \
--dataset 256


# --dataset to set image size

# original

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ../train.py \
# --gpu 3 \
# --k 256 \
# --d 16 \
# --beta 1 \
# --vqvae_batch_size 50 \
# --vqvae_epochs 100 \
# --vqvae_lr 3e-4 \
# --vqvae_layers 16 32 \
# --dataset 256

