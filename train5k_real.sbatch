#!/bin/bash
#SBATCH --job-name="Training 50k real labels"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=2GB
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
srun python train.py --trainingsetsize=30000 epochs=800
srun python train.py --trainingsetsize=40000 epochs=800
srun python train.py --trainingsetsize=50000 epochs=800

