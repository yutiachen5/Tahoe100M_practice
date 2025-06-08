#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A naderilab
#SBATCH -p scavenger-gpu,gpu-common
#SBATCH -J tahoe_train
#SBATCH --gres=gpu:1
#SBATCH -o /hpc/home/yc583/Tahoe100M_practice/train.out
#SBATCH -e /hpc/home/yc583/Tahoe100M_practice/train.err
#SBATCH --mem=100GB
#

hostname
nvidia-smi
python /hpc/home/yc583/Tahoe100M_practice/main.py