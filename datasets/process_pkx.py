
import sys
import numpy as np
import pandas as pd

if __name__ == "__main__":

    with open('INDEX_refined_data.2018') as lines:
        index = [x.split() for x in lines if "#" not in x]

    index_df = pd.DataFrame(index,
                            columns=['pdbid', 'reso', 'time',
                                     'pKx', 'Kx', 'sep',
                                     'pdf', 'lig'])
    index_df.index = index_df['pdbid']

    with open('input_refine_set.dat') as lines:
        inp = [x.split() + [x.split()[0].split("/")[0]] for x in lines]

    inp_df = pd.DataFrame(inp, columns=['pro', 'lig', 'pdbid'])
    inp_df.index = inp_df['pdbid']
    index_df = index_df.reindex(inp_df.index.values)

    # get test set v2013
    v2013 = pd.read_csv('test_v2013_195_PDB_infor.csv',
                        header=0, index_col=0)['PDBID'].values
    v2016 = pd.read_csv('test_v2016_287_PDB_infor.csv',
                        header=0, index_col=0)['PDBID'].values

    is_v2016, is_v2013 = [], []
    for c in index_df.index.values:
        if c in v2013:
            is_v2013.append(1)
        else:
            is_v2013.append(0)

        if c in v2016:
            is_v2016.append(1)
        else:
            is_v2016.append(0)

    index_df['is_v2013'] = is_v2013
    index_df['is_v2016'] = is_v2016

    index_df.to_csv('pkx_ordered.csv', header=True, index=False)
