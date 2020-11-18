#!/bin/bash
#SBATCH 

source /home/home2/ct214/virtual_envs/ml/bin/activate
srun -u python 360to2d.py