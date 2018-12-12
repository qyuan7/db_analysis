import pandas as pd
import numpy as np
from copy import deepcopy
from rdkit import Chem
from data import *
from sklearn.externals import joblib


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


def predict_property(model_file, fps):
    """
    Function to predict the properties of generated molecules
    Args:
        model_file: File containing pre-trained ML model for prediction
        fps: list of molecular fingerprints

    Returns: list of predicted valued

    """
    model = joblib.load(model_file)
    return model.predict(fps)


def main():
    #total_over, total_num = [], []
    #total_unique_list = []
    ori_df = pd.read_csv('./sampled_da_info/refined_smii.csv',header=None)
    ori_list = ori_df[0].tolist()
    frames = []
    gen_mols = []
    gen_fps = []
    for i in [1024, 2048, 4096, 8192, 16384, 32768]:
        gen_df = pd.read_csv('./sampled_da_info/sampled_da'+str(i)+'_smi.csv', header=None)
        gen_list = gen_df[0].tolist()
        over, num, smi_list = get_smi_list_overlap(ori_list, gen_list)
        smi_mols = get_mols(smi_list)
        smi_fps, failed_mols = get_fingerprints(smi_mols)
        for idx in sorted(failed_mols, reverse=True):
            del smi_list[idx]
        smi_df = pd.Series(data=smi_list, name='SMILES').to_frame()
        smi_df.loc[:,'Group'] = i
        frames.append(smi_df)
    #    total_over.append(over)
    #    total_num.append(num)
    #    total_unique_list.extend([smi_list])

    unique_df = pd.concat(frames)
    print(len(unique_df))
    gen_smi = unique_df['SMILES'].tolist()
    #for idx in sorted(failed_mols, reverse=True):
    #    del gen_smi[idx]
    #    del gen_mols[idx]
    gen_mols = get_mols(gen_smi)
    gen_fps, _ = get_fingerprints(gen_mols)
    print(len(gen_fps))
    unique_df['Gaps'] = predict_property('gbdt_regessor_gap.joblib', gen_fps)
    unique_df['Dips'] = predict_property('gbdt_regessor_dip.joblib', gen_fps)
    #gaps_series = pd.Series(data=gen_gaps)
    #dip_series = pd.Series(data=gen_dips)

    #unique_df.loc[:, 'Gap'] = gaps_series
    #unique_df.loc[:, 'Dip'] = dip_series
    unique_df.to_csv('unique_sampled_smiles_corr.csv', index=False)


if __name__ == '__main__':
    main()





