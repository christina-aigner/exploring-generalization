#!/bin/bash
#SBATCH --job-name="exgen_random_1k"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=2GB
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/usr/stud/aignerc/development/slurm/logs/slurm-%j.out
#SBATCH --error=/usr/stud/aignerc/development/slurm/logs/slurm-%j.out
srun python train.py --trainingsetsize=1000 --randomlabels=True

