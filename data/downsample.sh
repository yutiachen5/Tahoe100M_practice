#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A naderilab
#SBATCH -p scavenger
#SBATCH -o /hpc/home/yc583/Tahoe100M_practice/data/downsample.out
#SBATCH -e /hpc/home/yc583/Tahoe100M_practice/data/downsample.err
#SBATCH --mem=10GB
#

python /hpc/home/yc583/Tahoe100M_practice/downsample.py