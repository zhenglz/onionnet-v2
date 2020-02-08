#!/bin/bash

output=$1

for d in ./* 
do

  if [ -d $d ]; then

    for pdbqt in $d/${d}_vinaout_*.pdb
    do
      echo "${d}/${d}_protein.pdb $pdbqt" >> $output
    done
  fi
done
