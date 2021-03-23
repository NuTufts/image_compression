#!/bin/bash

# Slurm submission script for training Spatial Embed Instance Segmentation Network

#SBATCH --job-name=vqvae_train
#SBATCH --output=logs/vqvae_train.%A.log
#SBATCH --error=logs/vqvae_train_error.%A.log
#SBATCH --mem-per-cpu=4000
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:p100:1

container=/cluster/tufts/wongjiradlab/jhwang11/image_compression/containers/vqvae_singularity.simg
SCRIPT_DIR=/cluster/tufts/wongjiradlab/jhwang11/image_compression/

COMMAND="python3 hyperparameter_tester.py"

module load singularity
module load cuda/10.1
srun singularity exec --nv ${container} bash -c "cd ${SCRIPT_DIR} && ${COMMAND}"


