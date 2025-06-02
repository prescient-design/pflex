#!/usr/bin/env bash

#BSUB -J pretrain
#BSUB -eo pretrain-%J.err
#BSUB -oo pretrain-%J.out
#BSUB -gpu num=1
#BSUB -R "rusage[mem=20GB]"  # GB  / core (256 GB total avail on sHPC)
#BSUB -n2 -R "span[hosts=1]"  # n cores on same node


# eval "$(conda shell.bash hook)"
# ml awscli
# ml Anaconda3/2022.10
source ~/.bashrc
conda deactivate
conda deactivate
conda activate structure
echo "pip          = $(which pip)"
echo "pip          = $(which python)"

nvidia-smi

export WANDB_DISABLE_CODE=true
export WANDB_INSECURE_DISABLE_SSL=true

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr , ' ')
echo "GPUS=${GPUS}"

cd $HOME/stage/pmhc_interface

python philia/train.py -m +sweep=pretrain
