#!/usr/bin/env bash
#SBATCH -J C1D_S3_Firebusters
#SBATCH -o results/o_stage4_conv1d.txt
#SBATCH -e results/e_stage4_conv1d.txt
#SBATCH -p gpgpu-1 --gres=gpu:1 --mem=100G
#SBATCH -t 10080
#SBATCH -s
#SBATCH -n 32
python master_stage4_conv1d.py
