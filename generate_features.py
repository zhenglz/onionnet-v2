#!/usr/bin/env python

import numpy as np
import pandas as pd
import mdtraj as mt
import itertools
import multiprocessing
import sys
import os

from biopandas.mol2 import PandasMol2


class ProteinParser(object):
    """Featurization of Protein-Ligand Complex based on
    onion-shape distance counts of atom-types.

    Parameters
    ----------
    pdb_fn : str
        The input pdb file name.
    lig_code : str
        The ligand residue name in the input pdb file.

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

    """

    def __init__(self, pdb_fn):
        self.pdb = mt.load(pdb_fn)

        self.receptor_indices = np.array([])
        self.rec_ele = np.array([])

        self.pdb_parsed_ = False
        self.coordinates = None

    def get_coordinates(self):
        self.coordinates = self.pdb.xyz[0][self.receptor_indices]

        return self

    def parsePDB(self, rec_sele="protein"):
        top = self.pdb.topology

        self.receptor_indices = top.select(rec_sele)

        table, bond = top.to_dataframe()

        # fetch the element type of each one of the protein atom
        self.rec_ele = table['element'][self.receptor_indices].values
        # fetch the coordinates of each one of the protein atom
        self.get_coordinates()

        self.pdb_parsed_ = True

        return self


class LigandParser(object):
    """

    Parameters
    ----------

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
        self.lig = PandasMol2().read_mol2(ligand_fn)
        #   print(self.lig.df.head())
        self.lig_data = self.lig.df

        self.lig_ele = None
        self.coordinates = None
        self.mol2_parsed_ = False

    def get_element(self):
        ele = list(self.lig_data["atom_type"].values)
        #   print(ele)

        self.lig_ele = list(map(get_ligand_elementtype, ele))

        #    print(self.lig_ele)
        return self

    def get_coordinates(self):
        self.coordinates = self.lig_data[['x', 'y', 'z']].values
        #        print(self.coordinates)
        return self

    def parseMol2(self):
        if not self.mol2_parsed_:
            self.get_element()
            self.get_coordinates()

            self.mol2_parsed_ = True

        return self


def get_protein_elementtype(e):
    all_elements = ["H", "C", "O", "N", "S", "DU"]
    if e in all_elements:
        return e
    else:
        return "DU"


def get_ligand_elementtype(e):
    all_elements = ["H", "C", "CAR", "Br", "Cl", "P", "F", "O", "N", "S", "DU"]
    # print(e, e.split(".")[0])
    if e == "C.ar":
        return "CAR"
    elif e.split(".")[0] in all_elements:
        return e.split(".")[0]
    else:
        return "DU"


def atomic_distance(dat):
    return np.sqrt(np.sum(np.square(dat[0] - dat[1])))


def distance_pairs(coord_pro, coord_lig):
    pairs = list(itertools.product(coord_pro, coord_lig))
    #   print(pairs[:10])
    distances = map(atomic_distance, pairs)

    return list(distances)


def distance2counts(megadata):
    d = np.array(megadata[0])
    c = megadata[1]

    return np.sum((np.array(d) <= c) * 1.0)


def generate_features(args):
    pro_fn, lig_fn, n_cutoffs = args

    pro = ProteinParser(pro_fn)
    pro.parsePDB()
    protein_data = pd.DataFrame([])
    protein_data["element"] = pro.rec_ele
    # print(pro.rec_ele)
    for i, d in enumerate(['x', 'y', 'z']):
        # coordinates by mdtraj in unit nanometer
        protein_data[d] = pro.coordinates[:, i] * 10.0

    lig = LigandParser(lig_fn)
    lig.parseMol2()
    ligand_data = pd.DataFrame()
    ligand_data['element'] = lig.lig_ele
    for i, d in enumerate(['x', 'y', 'z']):
        ligand_data[d] = lig.coordinates[:, i]

    # print("LIGAND COORD GENERATE")
    elements_ligand = ["H", "C", "CAR", "O", "N", "S", "P", "DU", "Br", "Cl", "F"]
    elements_protein = ["H", "C", "O", "N", "S", "DU"]

    onionnet_counts = pd.DataFrame()

    for el in elements_ligand:
        for ep in elements_protein:
            protein_xyz = protein_data[protein_data['element'] == ep][['x', 'y', 'z']].values
            ligand_xyz = ligand_data[ligand_data['element'] == el][['x', 'y', 'z']].values

            #       print(ligand_xyz[:10], protein_xyz[:10])
            # distances = distance_pairs(protein_xyz, ligand_xyz)
            # print(distances[:10])
            counts = np.zeros(len(n_cutoffs))

            #       print(el, ep, "GET ELE TYPE SPEC DISTANCES")
            if len(protein_xyz) and len(ligand_xyz):
                # print(protein_xyz.shape, ligand_xyz.shape)
                distances = distance_pairs(protein_xyz, ligand_xyz)

                # print(sorted(distances)[:10], sorted(distances)[-10:])
                for i, c in enumerate(n_cutoffs):
                    single_count = distance2counts((distances, c))
                    if i > 0:
                        single_count = single_count - counts[i - 1]
                    counts[i] = single_count

            feature_id = "%s_%s" % (el, ep)
            onionnet_counts[feature_id] = counts

    return list(onionnet_counts.values.ravel()), pro_fn


def main():
    inp = sys.argv[1]
    out = sys.argv[2]

    data_root = './general_set'

    num_threads = 1

    n_cutoffs = np.linspace(0.1, 3.1, 61)

    with open(inp)  as lines:
        codes = [x for x in lines if ("#" not in x and len(x.split()))]

    input_arguments = []
    for c in codes:
        r_fn = os.path.join(data_root, "%s/%s_protein.pdb" % (c, c))
        l_fn = os.path.join(data_root, "%s/%s_ligand.mol2" % (c, c))

        input_arguments.append((r_fn, l_fn, n_cutoffs))

    if num_threads <= 1:
        dat = []
        labels = []
        for item in input_arguments:
            r = generate_features(item)
            dat.append(r[0])
            labels.append(r[1])
    else:
        pool = multiprocessing.Pool(num_threads)
        results = pool.map(generate_features, input_arguments)

        pool.close()
        pool.join()

        dat = [x[0] for x in results]
        labels = [x[1] for x in results]

    features = pd.DataFrame(dat, index=labels)

    features.to_csv(out, header=False, index=True)

    print("Completed.")
