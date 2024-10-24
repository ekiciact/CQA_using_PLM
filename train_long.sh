#!/bin/bash
#SBATCH --account=intro_vsc36028
#SBATCH --cluster genius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=10gb
#SBATCH -p gpu_v100
#SBATCH --gpus-per-node=1
#SBATCH --time=72:00:00
#SBATCH --job-name=train


/usr/bin/nvidia-smi
/data/leuven/360/vsc36028/minicinda3/envs/semtab1/bin/python /user/leuven/360/vsc36028/semtabr2-main/train_multi.py