#!/bin/bash
# Example slurm running script.
#SBATCH -A GR79-29
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=90000

module load common/compilers/gcc/8.3.1
module load gpu/cuda/10.2
cd ~/nasze-ca/models
python3 cnn_regressor.py --cnn_filters="125, 182, 59, 26, 33" --cnn_pools="7, 17, 15, 30, 14" --kernel=83 --dnn_dropout="0.0558, 0.6653" --stride=6 --epochs=50000
