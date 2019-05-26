#!/bin/bash
#SBATCH --job-name="Training 1-5k random labels"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=2GB
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
srun python train.py --trainingsetsize=1000 --randomlabels=True
srun python train.py --trainingsetsize=2000 --randomlabels=True
srun python train.py --trainingsetsize=3000 --randomlabels=True
srun python train.py --trainingsetsize=4000 --randomlabels=True
srun python train.py --trainingsetsize=5000 --randomlabels=True

