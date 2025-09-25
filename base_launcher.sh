#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpuv

cd $SLURM_SUBMIT_DIR

# Load environment
source ~/.bashrc
conda activate metataco

python pretrain_gym.py save_train_video=true  use_wandb=true wandb_tag=apt_fixed agent=icm_apt random_init=false 
# python pretrain_gym.py save_train_video=true  use_wandb=true wandb_tag=proto_fixed agent=proto random_init=false 
# python pretrain_gym.py save_train_video=true  use_wandb=true wandb_tag=rnd_fixed agent=rnd random_init=false 