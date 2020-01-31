#!/usr/bin/env bash
#SBATCH -J Firebusters
#SBATCH -o o_stage2_gru.txt
#SBATCH -e e_stage2_gru.txt
#SBATCH -p gpgpu-1 --gres=gpu:1 --mem=250G
#SBATCH -t 720
#SBATCH --exclusive
python master_stage2_gru.py
