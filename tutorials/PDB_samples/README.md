# Generate features for PDB or docked protein-ligand complex. 

For each complex file, just place the files' pathes into the input 
file ("receptor.pdb ligand.mol2"), one file per line.

## 1. Make an input file for featurization


    10gs/10gs_protein.pdb 10gs/10gs_ligand.mol2
    1a30/1a30_protein.pdb 1a30/1a30_ligand.mol2
    1bcu/1bcu_protein.pdb 1bcu/1bcu_ligand.mol2
    1e66/1e66_protein.pdb 1e66/1e66_ligand.mol2
    1f8b/1f8b_protein.pdb 1f8b/1f8b_ligand.mol2

## 2. Generate features using generate_features.py
Example commands:
    
    python generate_features.py -h
    python generate_features.py -inp input_PDB_samples.dat -out features.csv
    # use 5 cpu cores
    python generate_features.py -inp input_PDB_samples.dat -out features.csv -nt 5

## 3. Make the prediction

    python predict_pKa.py -h
    python predict_pKa.py -fn docking_complexes_features.csv -model ../../models/OnionNet_HFree.model \
    -scaler ../../models/StandardScaler.model -out predicted_pka_values.csv

Note: The larger the pka value is, the stronger it binds to a receptor.
