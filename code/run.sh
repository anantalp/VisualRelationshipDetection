#!/bin/bash
#SBATCH --nodes=1 # Get one node
#SBATCH --cpus-per-task=1 # Two cores per task
#SBATCH --ntasks=1 # But only one task
#SBATCH --gres=gpu:1 # And one GPU
#SBATCH --gres-flags=enforce-binding # Insist on good CPU/GPU alignment
#SBATCH --time=9-00:00:00
#SBATCH --job-name=vrd_run # Name the job so I can see it in squeue
#SBATCH --mail-type=BEGIN,END,FAIL # Send me email for various states
#SBATCH --mail-user anantalp@knights.ucf.edu # Use this address
#SBATCH --output=vrd-%J.txt

# Load modules
module load cuda/cuda-10.1

python main.py