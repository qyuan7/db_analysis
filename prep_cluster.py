#!/usr/bin/env python
from data import *
from rdkit import Chem, DataStructs

"""
Calculate the pairwise molecular similarity between the D-A molecules for future clustering
"""


def get_similarity(mols, compounds, fps_morgan):
    """
    Calculate the pairwise molecular similarity
    Args:
        mols: list of mol files for the compounds
        compounds: list of compound unique ids
        fps_morgan: list of fingerprints for the compounds

    Returns: lines containing the 'source','target','similarity' information

    """
    total_sim = ''
    for i in range(len(mols)):
        ref_fp = fps_morgan[i]
        for j in range(i+1,len(mols)):
            morgan2_sim = DataStructs.DiceSimilarity(ref_fp, fps_morgan[j])
            sims = str(compounds[i])+','+str(compounds[j].rstrip())+','+str(morgan2_sim)+'\n'
            total_sim += sims
    return total_sim

def main():
    gresult = connect_db('solar.db', 'KS_gap')
    smiles, compounds, gaps = get_data(gresult)
    mols = get_mols(smiles)
    fps_morgan, failed_mols = get_fingerprints(mols)
    refine_compounds(compounds, mols, gaps, failed_mols)
    total_sim = get_similarity(mols, compounds,fps_morgan)
    output = open('DA_similarity.csv', 'w')
    output.write('reference,compound,morgan2\n')
    output.write(total_sim)
    output.close()


if __name__ == '__main__':
    main()
