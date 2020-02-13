#!/usr/bin/env bash
#SBATCH -J BGRU_S3_Firebusters
#SBATCH -o results/o_stage4_bigru.txt
#SBATCH -e results/e_stage4_bigru.txt
#SBATCH -p gpgpu-1 --gres=gpu:1 --mem=100G
#SBATCH -t 10080
#SBATCH -s
#SBATCH -n 32
python master_stage4_bigru.py
