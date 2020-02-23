#!/usr/bin/env bash
#SBATCH -J C1D_SX_Firebusters
#SBATCH -o results/o_stageX_conv1d.txt
#SBATCH -e results/e_stageX_conv1d.txt
#SBATCH -p gpgpu-1 --gres=gpu:1 --mem=200G
#SBATCH -t 10080
#SBATCH -s
#SBATCH -n 32
python master_stageX_conv1d.py
