#!/bin/bash
#SBATCH  --gres=gpu:2 --constraint=[v100|p100] -p compsci-gpu
set path=/usr/local/cuda/bin $path 
source /home/home2/ct214/virtual_envs/ml/bin/activate 
cd ./networks/correlation_package
rm -rf *_cuda.egg-info build dist __pycache__
python3 setup.py install --user

cd ../resample2d_package
rm -rf *_cuda.egg-info build dist __pycache__
python3 setup.py install --user

cd ../channelnorm_package
rm -rf *_cuda.egg-info build dist __pycache__
python3 setup.py install --user

cd ..
