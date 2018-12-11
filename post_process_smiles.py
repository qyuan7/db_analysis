import pandas as pd
import numpy as np
from copy import deepcopy
from rdkit import Chem
import matplotlib.pyplot as plot


def get_smi_list_overlap(large, small):
    """

    Args:
        large: list containing the SMILE structures for transfer training
        small: list containing the SMILE structures for transfer sampling

    Returns: num of repeat SMILES, num of unique SMILES in transfer sampling, list of unique SMILES

    """
    def can_smile(smi_list):
        can_list = []
        for item in smi_list:
            if Chem.MolFromSmiles(item) is not None:
                can_item = Chem.MolToSmiles(Chem.MolFromSmiles(item))
                can_list.append(can_item)
        return can_list
    large_can, small_can = can_smile(large), can_smile(small)
    small_copy = deepcopy(small_can)
    overlap = set(small_can).intersection(large_can)
    for item in overlap:
        small_copy.remove(item)
    return len(overlap), len(small_copy), small_copy


total_over, total_num = [], []
total_unique_list = []
ori_df = pd.read_csv('./sampled_da_info/refined_smii.csv',header=None)
ori_list = ori_df[0].tolist()
frames = []
for i in [1024, 2048, 4096, 8192, 16384, 32768]:
    gen_df = pd.read_csv('./sampled_da_info/sampled_da'+str(i)+'_smi.csv', header=None)
    gen_list = gen_df[0].tolist()
    over, num, smi_list = get_smi_list_overlap(ori_list, gen_list)
    smi_df = pd.Series(data=smi_list, name='SMILES').to_frame()
    smi_df.loc[:,'Group'] = i
    frames.append(smi_df)
    total_over.append(over)
    total_num.append(num)
    total_unique_list.extend([smi_list])

unique_df = pd.concat(frames)
unique_df.to_csv('unique_sampled_smiles.csv', index=False)
