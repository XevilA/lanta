#!/bin/bash
#SBATCH -p gpu                      # Partition
#SBATCH -N 1                        # Number of nodes
#SBATCH --ntasks-per-node=1         # Tasks per node
#SBATCH --cpus-per-task=8           # CPUs per task
#SBATCH --gpus-per-node=1
