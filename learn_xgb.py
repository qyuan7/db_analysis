#!/usr/bin/env python
#noinspection PyUnresolvedReferences
import pandas as pd
import numpy as np
from data import *
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict as cvp
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold

def get_fp_from_id(all_ids,all_fps,id_list):
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


def modelfit(alg, trainfeatures,trainlabels, useTrainCV=True, cv_folds=4,early_stopping_rounds=30):
    """
    GridsearchCV for the regression DT
    Args:
        X: List, Train features (Xtrain)
        Y: List, Lablels (Ytrain)
        params: dictionary of parameters to tune on
        n_fold: number of folds for cross validation

    Returns: Best estimator and best params for the GridsearchCV

    """

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(trainfeatures, label=trainlabels)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_xgb_params()['n_estimators'], nfold=cv_folds,
                          metrics='rmse',early_stopping_rounds=early_stopping_rounds,verbose_eval=True)
        alg.set_params(n_estimators=cvresult.shape[0])
    alg.fit(trainfeatures, trainlabels, eval_metric='rmse')
    y_pred_train = alg.predict(trainfeatures)
    print('Model report\n')
    print('RMSE error {:.4f}'.format(np.sqrt(mean_squared_error(trainlabels, y_pred_train ))))



def main():
    qresult = connect_db('solar.db','dip')
    smiles, compounds,gaps = get_data(qresult)
    mols = get_mols(smiles)
    fps_morgan, failed_mols = get_fingerprints(mols)
    refine_compounds(compounds,mols,gaps,failed_mols)
    compound_array = np.array(compounds)
    gaps_array = np.array(gaps)
    train_id, test_id, y_train, y_test = train_test_split(compound_array,gaps_array,test_size=0.20,random_state=0)
    train_fps = get_fp_from_id(compounds,fps_morgan,train_id)
    test_fps = get_fp_from_id(compounds,fps_morgan,test_id)
    xgb1 =XGBRegressor(n_estimators=2000,learning_rate=0.03,max_depth=7,
                       colsample_bytree=0.6, nthread=8,scale_pos_weight=1,gamma=0, random_state=0,
                       subsample=0.6,min_child_weight=3,early_stopping_rounds=10,reg_alpha=1)
    modelfit(xgb1, train_fps, y_train)
    #xgb1 = joblib.load('gbdt_dip_xgb.joblib')
    #joblib.dump(xgb1, 'gbdt_dip_xgb2.joblib')
    y_pred_cv = cvp(xgb1, train_fps, y_train, cv=4, n_jobs=8)
    y_train_pred = xgb1.predict(train_fps)
    y_pred_test = xgb1.predict(test_fps)
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    train_df['id'] = pd.Series(train_id)
    train_df['dip_exp'] = pd.Series(y_train)
    train_df['dip_cv'] = pd.Series(y_pred_cv)
    train_df['dip_gbdt'] = pd.Series(y_train_pred)
    train_df['Group'] = 'Train'
    test_df['id'] = pd.Series(test_id)
    test_df['dip_exp'] = pd.Series(y_test)
    test_df['dip_cv'] = pd.Series(y_pred_test)
    test_df['dip_gbdt'] = pd.Series(y_pred_test)
    test_df['Group'] = 'Test'
    result_df = pd.concat([train_df, test_df])

    result_df.to_csv('dip_xgb_train_test.csv')
    test_err = mean_squared_error(y_pred_test, y_test)
    print('Test error: {:4f}'.format(np.sqrt(test_err)))
    #param_test1 = {'max_depth':[3,4,5]}
    #gsearch1 = GridSearchCV(estimator=xgb1,param_grid=param_test1, scoring='neg_mean_squared_error', n_jobs=8,
    #                         return_train_score=True, cv=4)
    #gsearch1.fit(train_fps, y_train)
    #cv_df = pd.DataFrame.from_dict(gsearch1.cv_results_)
    #cv_df.to_csv('cv_scores_reg.csv')
    #print(cv_df)
    #print('Best params {}\n Best score {}'.format(gsearch1.best_params_, gsearch1.best_score_))


if __name__ == '__main__':
    main()