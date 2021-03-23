#!/bin/bash

# Slurm submission script for training Spatial Embed Instance Segmentation Network

#SBATCH --job-name=vqvae_train
#SBATCH --output=logs/rootimgpng.%A.log
#SBATCH --error=logs/rootimgpng_error.%A.log
#SBATCH --mem-per-cpu=4000
#SBATCH --time=1-00:00:00

container=/cluster/tufts/wongjiradlab/jhwang11/image_compression/containers/vqvae_singularity.simg
SCRIPT_DIR=/cluster/tufts/wongjiradlab/jhwang11/image_compression/
INPUT_IMG_DIR=/cluster/tufts/wongjiradlab/larbys/data/mcc9/mcc9_v29e_dl_run3_G1_extbnb_dlana/data/mcc9_v29e_dl_run3_G1_extbnb_dlana/
OUTPUT_IMG_DIR=/cluster/tufts/wongjiradlab/jhwang11/image_compression/results/originals/originals_png/
BLOCKSIZE=256
COMMAND="python3 root_img_to_jpg.py"

module load singularity
module load cuda/10.1
srun singularity exec --nv ${container} bash -c "cd ${SCRIPT_DIR} && ${COMMAND} -i ${INPUT_IMG_DIR} -o ${OUTPUT_IMG_DIR} -b ${BLOCKSIZE}"


