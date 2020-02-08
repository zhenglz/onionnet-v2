#!/usr/bin/env python

import numpy as np
import pandas as pd
import mdtraj as mt
import itertools
import sys, os, math
import uuid
import subprocess as sp
import re

try:
    from mpi4py import MPI
except ImportError:
    print("MPI4PY not loaded. ")

import argparse
from argparse import RawDescriptionHelpFormatter
from rdkit import Chem


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

    def babel_converter(self, mol_file):
        if os.path.exists(mol_file):
            try:
                templ_file = str(hex(uuid.uuid4())) + ".pdb"
                cmd = 'obabel %s -O %s > /dev/null' % (mol_file, templ_file)
                job = sp.Popen(cmd, shell=True)
                job.communicate()

                self.molecule_ = self.converter_['pdb']()
                os.remove(templ_file)
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


class ParseMolecule(Molecule):

    def __init__(self, molfile, input_format="smile", addH=False):
        super().__init__(input_format)

        if addH:
            self.molecule_ = Chem.AddHs(self.molecule_)

        # load mol file, or a simile string
        self.load_molecule(mol_file=molfile)
        self.coordinates_ = np.array([])

    def get_xyz(self):
        """
        Get the coordinates of a ligand

        Returns
        -------
        xyz : np.ndarray, shape = [M, 3]
            The xyz coordinates of all ligand atoms. M is the number of
            ligand atoms.
        """

        pos = self.molecule_.GetConformer()
        self.coordinates_ = pos.GetPositions()

        return self.coordinates_


class ParseProtein(object):

    def __init__(self, pdb_fn):

        pdb = mt.load_pdb(pdb_fn)

        self.pdb = pdb.atom_slice(pdb.topology.select("protein"))
        self.top = self.pdb.topology
        self.seq = ""

        self.n_residues = None

    def get_seq(self):
        """Generate the residue sequence from the PDB file of the receptor.

        Returns
        -------
        self: an instance of itself
        """
        self.seq = self.top.to_fasta()
        self.n_residues = len(self.seq)

        return self.seq

    def contact_calpha(self, cutoff=0.5):
        """Compute the Capha contact map of the protein itself.

        Parameters
        ----------
        cutoff : float, default = 0.5 nm
            The distance cutoff for contacts

        Returns
        -------
        cmap : np.ndarray, shape = [N, N]
            The alphaC contact map, N is the number of residues

        """

        # define pairs
        c_alpha_indices = self.top.select("name CA")
        print("Number of Calpha atoms ", c_alpha_indices.shape)
        pairs_ = list(itertools.product(c_alpha_indices, c_alpha_indices))

        distance_matrix_ = mt.compute_distances(self.pdb, atom_pairs=pairs_)[0]
        distance_matrix_ = distance_matrix_.reshape((-1, c_alpha_indices.shape[0]))

        cmap = (distance_matrix_ <= cutoff)*1.0

        return cmap

    def cal_distances(self, point_pair):

        return np.sqrt(np.sum(np.square(point_pair[0] - point_pair[1])))

    def contacts_nbyn(self, cutoff, crds_p, crds_l, nbyn=True):
        """
        Calculate the normalized contact number between two sets of points

        Parameters
        ----------
        cutoff : float,
            Distance cutoff
        crds_p : np.ndarray, shape = [N, 3]
            The coordinates of protein atoms
        crds_l : np.ndarray, shape = [N, 3]
            The coordinates of ligand atoms
        nbyn : bool, default is True
            Do normalization of the contact number

        Returns
        -------
        counts : float
            The atom number normalized counts

        """

        # if not self.distance_calculated_:
        pairs = itertools.product(crds_l, crds_p)
        counts = np.sum((np.array(list(map(self.cal_distances, pairs))) <= cutoff) * 1.0)

        if nbyn:
            return counts / (crds_p.shape[0] * crds_l.shape[0])
        else:
            return counts

    def distances_all_pairs(self, lig_xyz, cutoff, verbose=True):
        # looping over all pairs
        d = np.zeros(self.n_residues)

        for i, p in enumerate(range(self.n_residues)):
            if i % 100 == 0 and verbose:
                print("Progress of residue-ligand contacts: ", i)

            atom_indices = self.top.select("(resid %d) and (symbol != H)" % i)

            pro_xyz = self.pdb.xyz[0][atom_indices]
            d[i] = self.contacts_nbyn(cutoff, pro_xyz, lig_xyz, nbyn=True)

        return d


def distance_padding(dist, max_pairs_=1000, padding_with=0.0):
    """

    Parameters
    ----------
    dist: np.array, shape = [N, ]
        The input data array
    max_pairs_: int, default = 1000
        The maximium number of features in the array
    padding_with: float, default=0.0
        The value to pad to the array

    Returns
    -------
    d: np.array, shape = [N, ]
        The returned array after padding
    """

    if dist.shape[0] < max_pairs_:
        left_size = math.floor((max_pairs_ - dist.shape[0])/2)
        right_size = max_pairs_ - left_size - dist.shape[0]
        if left_size > 0:
            d = np.concatenate((np.repeat(padding_with, left_size), dist))
        else:
            d = dist

        d = np.concatenate((d, np.repeat(padding_with, right_size)))

    elif dist.shape == max_pairs_:
        d = dist
    else:
        d = dist[:max_pairs_]
        print("Warning: number of features higher than %d" % max_pairs_)

    return d


def hydrophobicity():
    '''http://assets.geneious.com/manual/8.0/GeneiousManualsu41.html'''
    hydrophobic = {
        'PHE': 1.0,
        'LEU': 0.943,
        'ILE': 0.943,
        'TYR': 0.880,
        'TRP': 0.878,
        'VAL': 0.825,
        'MET': 0.738,
        'PRO': 0.711,
        'CYS': 0.680,
        'ALA': 0.616,
        'GLY': 0.501,
        'THR': 0.450,
        'SER': 0.359,
        'LYS': 0.283,
        'GLN': 0.251,
        'ASN': 0.236,
        'HIS': 0.165,
        'GLU': 0.043,
        'ASP': 0.028,
        'ARG': 0.0,
        'UNK': 0.501,
    }

    return hydrophobic


def polarizability():
    """https://www.researchgate.net/publication/220043303_Polarizabilities_of_amino_acid_residues/figures"""
    polar = {
        'PHE': 121.43,
        'LEU': 91.6,
        'ILE': 91.21,
        'TYR': 126.19,
        'TRP': 153.06,
        'VAL': 76.09,
        'MET': 102.31,
        'PRO': 73.47,
        'CYS': 74.99,
        'ALA': 50.16,
        'GLY': 36.66,
        'THR': 66.46,
        'SER': 53.82,
        'LYS': 101.73,
        'GLN': 88.79,
        'ASN': 73.15,
        'HIS': 99.35,
        'GLU': 84.67,
        'ASP': 69.09,
        'ARG': 114.81,
        'UNK': 36.66,
    }

    return polar


def stringcoding():
    """Sequence from http://www.bligbi.com/amino-acid-table_242763/epic-amino-acid-table-l99-
    on-nice-home-designing-ideas-with-amino-acid-table/"""

    sequence = {
        'PHE': 18,
        'LEU': 16,
        'ILE': 15,
        'TYR': 19,
        'TRP': 20,
        'VAL': 14,
        'MET': 17,
        'PRO': 12,
        'CYS': 10,
        'ALA': 13,
        'GLY': 11,
        'THR': 7,
        'SER': 6,
        'LYS': 3,
        'GLN': 8,
        'ASN': 8,
        'HIS': 2,
        'GLU': 4,
        'ASP': 5,
        'ARG': 1,
        'UNK': 11,
    }

    return sequence


def residue_string2code(seq, method=stringcoding):
    mapper = method()
    return [mapper[x] if x in mapper.keys()
            else mapper['UNK']
            for x in seq]


def generate_contact_features(protein_fn, ligand_fn,
                              ncutoffs, verbose=True):
    """
    Generate features based on protein sequences.

    Parameters
    ----------
    protein_fn
    ligand_fn
    ncutoffs
    verbose

    Returns
    -------
    features : np.ndarray, shape = [M * N, ]
        The output features.

    """

    protein = ParseProtein(protein_fn)
    ligand  = ParseMolecule(ligand_fn, input_format=ligand_fn.split(".")[-1])
    xyz_lig = ligand.get_xyz()

    seq = protein.seq
    if verbose: print("Length of residues ", len(seq))

    if verbose: print("START alpha-C contact map")
    r = np.array([])
    for c in np.linspace(0.6, 1.6, 3):
        cmap = protein.contact_calpha(cutoff=c).sum(axis=0)
        cmap = distance_padding(cmap)

        r = np.concatenate((r, cmap))
        #if verbose: print(cmap)
    if verbose:print("COMPLETE contactmap")

    scaling_factor = [20, 153, 1.]
    for i, m in enumerate([stringcoding, polarizability, hydrophobicity]):
        coding = np.array(residue_string2code(seq, m)) / scaling_factor[i]

        if verbose:
            print("START sequence to coding")

        mapper = m()
        coding = distance_padding(coding, padding_with=0)

        if verbose:
            print(coding)

        r = np.concatenate((r, coding))

    if verbose:
        print("COMPLETE sequence to coding")
        print("SHAPE of result: ", r.shape)

    for c in ncutoffs:
        if verbose:
            print("START residue based atom contact nbyn, cutoff=", c)

        counts = protein.distances_all_pairs(xyz_lig, c, verbose)

        d = distance_padding(counts)

        r = np.concatenate((r, d))

    if verbose:
        print("SHAPE of result: ", r.shape)

    return r.ravel()

