#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A naderilab
#SBATCH -J train-array
#SBATCH -o /hpc/home/yc583/Tahoe100M_practice/slurms/%a.out
#SBATCH -e /hpc/home/yc583/Tahoe100M_practice/slurms/%a.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --array=0-2%3
#SBATCH -p gpu-common,scavenger-gpu,biostat-gpu
#SBATCH --nice=100
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=1
#

hostname
nvidia-smi
python /hpc/home/yc583/Tahoe100M_practice/main.py 