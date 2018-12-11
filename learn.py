#!/usr/bin/env python
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
import numpy as np
from data import *
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import os.path
import torch
#import warning
#warnings.filterwarnings("ignore", category=DeprecationWarning) 


def get_fp_from_id(all_ids, all_fps, id_list):
    """
    Get list of fingerprints for molecules of which the unique ids are in id_list
    Args:
        all_ids: list, ids for all the compounds
        all_fps: list containing the fingerprints of all compounds (except unreasonable mols)
        id_list: list, ids for compounds that are needed in the train/test sets

    Returns: lists of fingerprints for the target molecules

    """
    fp_list = [all_fps[all_ids.index(item)] for item in id_list]
    return fp_list


def grid_GBR(X, Y, params, n_fold):
    """
    GridsearchCV for the regression DT
    Args:
        X: List, Train features (Xtrain)
        Y: List, Lablels (Ytrain)
        params: dictionary of parameters to tune on
        n_fold: number of folds for cross validation

    Returns: Best estimator and best params for the GridsearchCV

    """
    gdb_regressor = GradientBoostingRegressor()
    clf = GridSearchCV(gdb_regressor, params, scoring='neg_mean_squared_error', cv=n_fold, n_jobs=16)
    clf.fit(X, Y)
    return clf.best_estimator_, clf.best_params_


def main():
    qresult = connect_db('solar.db', 'dip')
    smiles, compounds, gaps = get_data(qresult)
    mols = get_mols(smiles)
    fps_morgan, failed_mols = get_fingerprints(mols)
    refine_compounds(compounds, mols, gaps, failed_mols)
    compound_array = np.array(compounds)
    gaps_array = np.array(gaps)
    train_id, test_id, y_train, y_test = train_test_split(compound_array, gaps_array, test_size=0.20, random_state=0)
    train_fps = get_fp_from_id(compounds, fps_morgan, train_id)
    test_fps = get_fp_from_id(compounds, fps_morgan, test_id)
    params = {'learning_rate': [0.01, 0.03, 0.05,0.1], 'n_estimators': [100,300, 500, 700], 'max_depth': [3, 4, 5]}
    gbr_regressor, gbr_cv_params = grid_GBR(train_fps, y_train,params,4)
    y_pred_train = gbr_regressor.predict(train_fps)
    y_pred_test = gbr_regressor.predict(test_fps)
    train_err = mean_squared_error(y_train,y_pred_train)
    test_err = mean_squared_error(y_test, y_pred_test)
    #print('MSE on training set is {}\nMSE on test set is {}'.format(train_err,test_err))
    train_db = pd.DataFrame()
    train_db['id'] = pd.Series(train_id)
    train_db['dip_exp'] = pd.Series(y_train)
    train_db['dip_gdbt'] = pd.Series(y_pred_train)
    test_db = pd.DataFrame()
    test_db['id'] = pd.Series(test_id)
    test_db['dip_exp'] = pd.Series(y_test)
    test_db['dip_gdbt'] = pd.Series(y_pred_test)
    frames = [train_db, test_db]
    result_db = pd.concat(frames)
    save_path = '/work/qyuan/db_analysis'
    file_name = 'gbdt_dip2.csv'
    full_name = os.path.join(save_path, file_name)
    result_db.to_csv(full_name, index=False)
    model = 'gbdt_regessor_dip2.joblib'
    model_name = os.path.join(save_path, model)
    joblib.dump(gbr_regressor, model_name)
    #output = open(full_name, 'w')
    #output.write('training error: {}'.format(train_err))
    #output.write('test error: {}'.format(test_err))
    #output.close()

if __name__ == '__main__':
    main()
