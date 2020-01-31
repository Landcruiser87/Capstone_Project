#!/usr/bin/env bash
#SBATCH -J Firebusters
#SBATCH -o output-%j.txt
#SBATCH -e error-%j.txt
#SBATCH -p gpgpu-1 --gres=gpu:1 --mem=250G
#SBATCH -t 5
#SBATCH --exclusive
python mane_test_01.py
