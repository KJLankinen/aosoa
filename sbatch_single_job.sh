#!/bin/bash

#SBATCH --job-name=cubble
#SBATCH --account=project_2002078
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:v100:1

module load gcc/10.4.0 cuda/12.1.1 gcc/11.2.0

## Build
srun cp -r $HOME/Code/cubble/include/aosoa /scratch/project_2002078/$USER/
srun cd /scratch/project_2002078/$USER/aosoa
srun scripts/build.sh

## Run
srun build/samples/cuda_samples/cuda_samples
