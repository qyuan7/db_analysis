#!/usr/bin/env python
#noinspection PyUnresolvedReferences
import pandas as pd
import numpy as np
from data import *
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def get_fp_from_id(all_ids,all_fps,id_list):
    """
    Get list of fingerprints for molecules of which the unique ids are in id_list
    Args:
        all_ids: list, ids for all the compounds
        all_fps: list containing the fingerprints of all compounds (except unreasonable mols)
        id_list: list, ids for compounds that are needed in the train/test sets

    Returns: lists of fingerprints for the target molecules

    """
    fp_list = []
    for item in id_list:
        idx = all_ids.index(item)
        fp_list.append(all_fps[idx])
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
    clf = GridSearchCV(gdb_regressor, params, scoring = 'neg_mean_squared_error', cv=n_fold)
    clf.fit(X, Y)
    return clf.best_estimator_, clf.best_params_


def main():
    qresult = connect_db('solar.db','KS_gap')
    smiles, compounds,gaps = get_data(qresult)
    mols = get_mols(smiles)
    fps_morgan, failed_mols = get_fingerprints(mols)
    refine_compounds(compounds,mols,gaps,failed_mols)
    compound_array = np.array(compounds)
    gaps_array = np.array(gaps)
    train_id, test_id, y_train, y_test = train_test_split(compound_array,gaps_array,test_size=0.20,random_state=0)
    train_fps = get_fp_from_id(compounds,fps_morgan,train_id)
    test_fps = get_fp_from_id(compounds,fps_morgan,test_id)
    params = {'learning_rate': [0.01, 0.03, 0.05,0.1], 'n_estimators': [100,300, 500, 700], 'max_depth': [3, 4, 5]}
    gbr_regressor, gbr_cv_params = grid_GBR(train_fps, y_train,params,4)
    y_pred_train = gbr_regressor.predict(train_fps)
    y_pred_test = gbr_regressor.predict(test_fps)
    train_err = mean_squared_error(y_train,y_pred_train)
    test_err = mean_squared_error(y_test, y_pred_test)
    print('MSE on training set is {}\nMSE on test set is {}'.format(train_err,test_err))


if __name__ == '__main__':
    main()
