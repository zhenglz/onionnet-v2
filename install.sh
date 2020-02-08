#!/bin/bash

#Check and install conda
conda_exe=$(which conda)
length_code=${#conda_exe}

if [ $length_code > 0 ]; then
  echo "Anaconda or Miniconda found in: $conda_exe"
else
  echo "No anaconda or miniconda, installing miniconda3 now ..."
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/miniconda.sh
  bash $HOME/miniconda.sh -b -p $HOME/miniconda3
  echo ""
  # add path to bashrc
  echo "Add Miniconda3 to bashrc PATH variable. "
  echo 'export PATH="$PATH:$HOME/miniconda3/bin"' >> $HOME/.bashrc
  rm $HOME/miniconda.sh

  # get current dir
  current_dir=$(echo $PWD)
  source $HOME/.bashrc
  cd $current_dir
fi

#Create a python environment
conda create -n onionnet python=3.6 -y
source activate onionnet
#conda activate onionnet
#Install necessary python modules with conda
conda install -c rdkit rdkit -y
conda install -c omnia mdtraj -y
conda install -c openbabel openbabel -y
conda install biopandas -c conda-forge -y
conda install numpy pandas scipy -y
conda install tensorflow-gpu -y

#Finish installation
echo "Installation completed. To use the package, please enable the environment by: "
echo "conda activate onionnet"
