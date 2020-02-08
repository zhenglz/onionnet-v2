#!/usr/bin/env python

import numpy as np
import pandas as pd
import mdtraj as mt
import itertools
import multiprocessing
import sys, os
from biopandas.mol2 import PandasMol2
import argparse
from argparse import RawDescriptionHelpFormatter
from rdkit import Chem
import subprocess as sp


# Define required atomic element type
elements_ligand = ["H", "C", "CAR", "O", "N",
                   "S", "P", "DU", "Br", "Cl", "F"]
elements_protein = ["H", "C", "O", "N", "S", "DU"]


class Molecule(object):
    """Small molecule parser object with Rdkit package.

    Parameters
    ----------
    in_format : str, default = 'smile'
        Input information (file) format.
        Options: smile, pdb, sdf, mol2, mol

    Attributes
    ----------
    molecule_ : rdkit.Chem.Molecule object
    mol_file : str
        The input file name or Smile string
    converter_ : dict, dict of rdkit.Chem.MolFrom** methods
        The file loading method dictionary. The keys are:
        pdb, sdf, mol2, mol, smile


    """

    def __init__(self, in_format="smile"):

        self.format = in_format
        self.molecule_ = None
        self.mol_file = None
        self.converter_ = None
        self.mol_converter()

    def mol_converter(self):
        """The converter methods are stored in a dictionary.

        Returns
        -------
        self : return an instance of itself

        """
        self.converter_ = {
            "pdb": Chem.MolFromPDBFile,
            "mol2": Chem.MolFromMol2File,
            "mol": Chem.MolFromMolFile,
            "smile": Chem.MolFromSmiles,
            "sdf": Chem.MolFromMolBlock,
            "pdbqt": self.babel_converter,
        }

        return self

    def babel_converter(self, mol_file, output):
        if os.path.exists(mol_file):
            try:
                cmd = 'obabel %s -O %s > /dev/null' % (mol_file, output)
                job = sp.Popen(cmd, shell=True)
                job.communicate()

                self.molecule_ = self.converter_['pdb']()
                return self.molecule_
            except:
                return None

    def load_molecule(self, mol_file):
        """Load a molecule to have a rdkit.Chem.Molecule object

        Parameters
        ----------
        mol_file : str
            The input file name or SMILE string

        Returns
        -------
        molecule : rdkit.Chem.Molecule object
            The molecule object

        """

        self.mol_file = mol_file
        if not os.path.exists(self.mol_file):
            print("Molecule file not exists. ")
            return None

        if self.format not in ["mol2", "mol", "pdb", "sdf", "pdbqt"]:
            print("File format is not correct. ")
            return None
        else:
            try:
                self.molecule_ = self.converter_[self.format](self.mol_file)
            except RuntimeError:
                return None

            return self.molecule_

class ProteinParser(object):
    """Featurization of Protein-Ligand Complex based on
    onion-shape distance counts of atom-types.

    Parameters
    ----------
    pdb_fn : str
        The input pdb file name. The file must be in PDB format.

    Attributes
    ----------
    pdb : mdtraj.Trajectory
        The mdtraj.trajectory object containing the pdb.
    receptor_indices : np.ndarray
        The receptor (protein) atom indices in mdtraj.Trajectory
    rec_ele : np.ndarray
        The element types of each of the atoms in the receptor
    pdb_parsed_ : bool
        Whether the pdb file has been parsed.
    distance_computed : bool
        Whether the distances between atoms in receptor and ligand has been computed.

    Examples
    --------
    >>> pdb = ProteinParser("input.pdb")
    >>> pdb.parsePDB('protein and chainid 0')
    >>> pdb.coordinates_
    >>> print(pdb.rec_ele)

    """

    def __init__(self, pdb_fn):
        self.pdb = mt.load_pdb(pdb_fn)

        self.receptor_indices = np.array([])
        self.rec_ele = np.array([])

        self.pdb_parsed_ = False
        self.coordinates_ = None

    def get_coordinates(self):
        """
        Get the coordinates in the pdb file given the receptor indices.

        Returns
        -------
        self : an instance of itself

        """
        self.coordinates_ = self.pdb.xyz[0][self.receptor_indices]

        return self

    def parsePDB(self, rec_sele="protein"):
        """
        Parse the pdb file and get the detail information of the protein.

        Parameters
        ----------
        rec_sele : str,
            The string for protein selection. Please refer to the following link.

        References
        ----------
        Mdtraj atom selection language: http://mdtraj.org/development/atom_selection.html

        Returns
        -------

        """
        top = self.pdb.topology
        # obtain the atom indices of the protein
        self.receptor_indices = top.select(rec_sele)
        _table, _bond = top.to_dataframe()

        # fetch the element type of each one of the protein atom
        self.rec_ele = _table['element'][self.receptor_indices].values
        # fetch the coordinates of each one of the protein atom
        self.get_coordinates()

        self.pdb_parsed_ = True

        return self


class LigandParser(object):
    """Parse the ligand with biopanda to obtain coordinates and elements.

    Parameters
    ----------
    ligand_fn : str,
        The input ligand file name.

    Methods
    -------

    Attributes
    ----------
    lig : a biopandas mol2 read object
    lig_data : a panda data object holding the atom information
    coordinates : np.ndarray, shape = [ N, 3]
        The coordinates of the atoms in the ligand, N is the number of atoms.

    """

    def __init__(self, ligand_fn):
        self.lig_file = ligand_fn
        self.lig = None
        self.lig_data = None

        self.lig_ele = None
        self.coordinates_ = None
        self.mol2_parsed_ = False

    def _format_convert(self, input, output):
        mol = Molecule(in_format=input.split(".")[-1])
        mol.babel_converter(input, output)
        return self

    def get_element(self):
        ele = list(self.lig_data["atom_type"].values)
        self.lig_ele = list(map(get_ligand_elementtype, ele))
        return self

    def get_coordinates(self):
        """
        Get the coordinates in the pdb file given the ligand indices.

        Returns
        -------
        self : an instance of itself

        """
        self.coordinates_ = self.lig_data[['x', 'y', 'z']].values
        return self

    def parseMol2(self):
        if not self.mol2_parsed_:
            if self.lig_file.split(".")[-1] != "mol2":
                out_file = self.lig_file + ".mol2"
                self._format_convert(self.lig_file, out_file)
                self.lig_file = out_file

            if os.path.exists(self.lig_file):
                self.lig = PandasMol2().read_mol2(self.lig_file)
            else:
                return None
            self.lig_data = self.lig.df
            self.get_element()
            self.get_coordinates()
            self.mol2_parsed_ = True

        return self


def get_protein_elementtype(e):
    if e in elements_protein:
        return e
    else:
        return "DU"


def get_ligand_elementtype(e):
    if e == "C.ar":
        return "CAR"
    elif e.split(".")[0] in elements_ligand:
        return e.split(".")[0]
    else:
        return "DU"


def atomic_distance(dat):
    return np.sqrt(np.sum(np.square(dat[0] - dat[1])))


def distance_pairs(coord_pro, coord_lig):
    pairs = list(itertools.product(coord_pro, coord_lig))
    distances = map(atomic_distance, pairs)

    return list(distances)


def distance2counts(megadata):
    d = np.array(megadata[0])
    c = megadata[1]

    return np.sum((np.array(d) <= c) * 1.0)


def generate_features(args):
    pro_fn, lig_fn, n_cutoffs = args
    print("INFO: Processing %s and %s ..." % (pro_fn, lig_fn))

    pro = ProteinParser(pro_fn)
    pro.parsePDB()
    protein_data = pd.DataFrame([])
    protein_data["element"] = pro.rec_ele
    # print(pro.rec_ele)
    for i, d in enumerate(['x', 'y', 'z']):
        # coordinates by mdtraj in unit nanometer
        protein_data[d] = pro.coordinates_[:, i]

    lig = LigandParser(lig_fn)
    lig.parseMol2()
    ligand_data = pd.DataFrame()
    ligand_data['element'] = lig.lig_ele
    for i, d in enumerate(['x', 'y', 'z']):
        # the coordinates in ligand are in angstrom
        ligand_data[d] = lig.coordinates_[:, i] * 0.1

    onionnet_counts = pd.DataFrame()

    for el in elements_ligand:
        for ep in elements_protein:
            protein_xyz = protein_data[protein_data['element'] == ep][['x', 'y', 'z']].values
            ligand_xyz = ligand_data[ligand_data['element'] == el][['x', 'y', 'z']].values

            counts = np.zeros(len(n_cutoffs))

            if protein_xyz.shape[0] and ligand_xyz.shape[0]:
                distances = distance_pairs(protein_xyz, ligand_xyz)

                for i, c in enumerate(n_cutoffs):
                    single_count = distance2counts((distances, c))
                    if i > 0:
                        single_count = single_count - np.sum(counts[i - 1])
                    counts[i] = single_count

            feature_id = "%s_%s" % (el, ep)
            onionnet_counts[feature_id] = counts

    return list(onionnet_counts.values.ravel()), pro_fn+"_"+lig_fn


def main():
    d = """
       Predicting protein-ligand binding affinities (pKa) with OnionNet model.
       
       Citation: Zheng L, Fan J, Mu Y. arXiv preprint arXiv:1906.02418, 2019.
       Author: Liangzhen Zheng (zhenglz@outlook.com)
       This script is used to generate inter-molecular element-type specific
       contact features. Installation instructions should be refered to
       https://github.com/zhenglz/onionnet-v2
       
       Examples:
       Show help information
       python generate_features.py -h
       Run the script
       python generate_features.py -inp input_samples.dat -out features_samples.csv
       # tutorial example
       cd tuttorials/PDB_samples
       python ../../generate_features.py -inp input_PDB_testing.dat -out 
       PDB_testing_features.csv
    """
    parser = argparse.ArgumentParser(description=d,
                                     formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-inp", type=str, default="input.dat",
                        help="Input. The input file containg the file path of each \n"
                             "of the protein-ligand complexes files (in pdb format.)\n"
                             "There should be only 1 column, each row or line containing\n"
                             "the input file path, relative or absolute path.")
    parser.add_argument("-out", type=str, default="output.csv",
                        help="Output. Default is output.csv \n"
                             "The output file name containing the features, each sample\n"
                             "per row. ")
    parser.add_argument("-nt", type=int, default=1,
                        help="Input, optional. Default is 1. "
                             "Use how many of cpu cores.")
    parser.add_argument("-upbound", type=float, default=3.1,
                        help="Input, optional. Default is 3.1 nm. "
                             "The largest distance cutoff.")
    parser.add_argument("-lowbound", type=float, default=0.1,
                        help="Input, optional. Default is 0.1 nm. "
                             "The lowest distance cutoff.")
    parser.add_argument("-nbins", type=int, default=61,
                        help="Input, optional. Default is 61. "
                             "The number of distance cutoffs.")

    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    num_threads = args.nt

    n_cutoffs = np.linspace(args.lowbound,
                            args.upbound, args.nbins)

    with open(args.inp) as lines:
        codes = [x.split() for x in lines if
                 ("#" not in x and len(x.split()))]

    input_arguments = []
    for c in codes:
        r_fn, l_fn = c[0], c[1]
        input_arguments.append((r_fn, l_fn, n_cutoffs))

    if num_threads <= 1:
        dat = []
        labels = []
        for item in input_arguments:
            ocontacts, fn = generate_features(item)
            dat.append(ocontacts)
            labels.append(fn)
    else:
        pool = multiprocessing.Pool(num_threads)
        results = pool.map(generate_features, input_arguments)

        pool.close()
        pool.join()
        dat = [x[0] for x in results]
        labels = [x[1] for x in results]

    elecombin =list(itertools.product(elements_ligand, elements_protein)) \
               * n_cutoffs.shape[0]
    elecombin = ["_".join(x) for x in elecombin]
    features = [elecombin[i]+"_"+str(i+1) for i in
                range(len(elecombin) * n_cutoffs.shape[0])]

    features = pd.DataFrame(dat, index=labels, columns=features)
    features.to_csv(args.out, header=True, index=True)

    print("INFO: Feature extraction completed.")
