#!/bin/bash
# Example slurm running script.
#SBATCH -A GR79-29
#SBATCH -p gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1


cd ~/nasze-ca/models
python3 dnn_optuna.py
