#!/usr/bin/env bash
#SBATCH -J cl2d_S2_Firebusters
#SBATCH -o results/o_stage2_cl2d.txt
#SBATCH -e results/e_stage2_cl2d.txt
#SBATCH -p gpgpu-1 --gres=gpu:1 --mem=100G
#SBATCH -t 10080
#SBATCH -s
#SBATCH -n 32
python master_stage2_convlstm2d.py
