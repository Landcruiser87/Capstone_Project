#!/usr/bin/env bash
#SBATCH -J CL2D_SX_Firebusters
#SBATCH -o results/o_stageX_cl2d.txt
#SBATCH -e results/e_stageX_cl2d.txt
#SBATCH -p gpgpu-1 --gres=gpu:1 --mem=200G
#SBATCH -t 10080
#SBATCH -s
#SBATCH -n 32
python master_stageX_convlstm2d.py
