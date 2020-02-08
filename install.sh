#!/bin/bash

# Install necessary python modules with conda
conda_exe=$(which conda)
length_code=${#conda_exe}
if [ $length_code -lt 0 ]; then
  conda create -n onionnet python=3.6
fi

conda activate onionnet

conda install -c rdkit rdkit
conda install -c omnia mdtraj
conda install -c openbabel openbabel
conda install numpy pandas scipy
conda install tensorflow

