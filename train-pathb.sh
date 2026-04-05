#!/bin/bash
# PATH B with GPU Support - TensorFlow 2.12.1 + CUDA 11.2
# Sets up CUDA paths and runs training

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

conda run -n bob python /DATA/anikde/Aurindum/DCTeam/DC_VIT/dc-aug-3april-pathb.py "$@"
