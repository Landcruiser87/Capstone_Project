#!/usr/bin/env bash
#SBATCH -J Dense_S5_Firebusters
#SBATCH -o results/o_stage5_dense.txt
#SBATCH -e results/e_stage5_dense.txt
#SBATCH -p gpgpu-1 --gres=gpu:1 --mem=200G
#SBATCH -t 10080
#SBATCH -s
#SBATCH -n 32
python master_stage5_dense.py
