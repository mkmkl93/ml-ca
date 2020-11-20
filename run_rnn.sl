#!/bin/bash
# Example slurm running script.
#SBATCH -A GR79-29
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=60000

cd ~/nasze-ca/models
python rnn_regressor.py --epochs=50000 --rnn_filter=512
