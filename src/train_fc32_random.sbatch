#!/bin/bash
#SBATCH --job-name="exgen_fc_32_random"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=2GB
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/usr/stud/aignerc/development/slurm/logs/slurm-%j.out
#SBATCH --error=/usr/stud/aignerc/development/slurm/logs/slurm-%j.out
srun python train.py --network=fc --epochs=1000 --trainingsetsize=10000 --numhidden=32 --randomlabels=True

