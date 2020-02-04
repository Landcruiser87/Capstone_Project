#!/usr/bin/env bash
#SBATCH -J BLSTM_S3_Firebusters
#SBATCH -o results/o_stage3_blstm.txt
#SBATCH -e results/e_stage3_blstm.txt
#SBATCH -p gpgpu-1 --gres=gpu:1 --mem=100G
#SBATCH -t 10080
#SBATCH -s
#SBATCH -n 32
python master_stage3_bilstm.py
