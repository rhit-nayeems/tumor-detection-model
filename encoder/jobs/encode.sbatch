#!/bin/bash

#SBATCH --job-name=autopet-encode

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8

#SBATCH --mem=64G

#SBATCH --time=06:00:00

#SBATCH --array=0-19

#SBATCH --output=%x_%A_%a.out

set -eo pipefail

source /home/steelevw/miniconda3/etc/profile.d/conda.sh

conda activate autopet

export AP=/home/steelevw/18/autopet

export nnUNet_raw=$AP/nnUNet_raw

export nnUNet_preprocessed=$AP/nnUNet_preprocessed

export nnUNet_results=$AP/nnUNet_results

export nnUNet_compile=f

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

python $AP/extract_features.py --chunk $SLURM_ARRAY_TASK_ID --num-chunks 20 --out $AP/features
