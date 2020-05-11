#!/bin/bash
# Example slurm running script.
#SBATCH -A GR79-29
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1

export NEPTUNE_API_TOKEN=''

module load common/compilers/gcc/8.3.1
module load gpu/cuda/10.2
cd ~/nasze-ca/models
python3 cnn_regressor.py
