import sys
import os
sys.path.append('src/')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import model_selection
import models
import plotting 
import delong
import argparse
import pickle
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from models import RandomForest, LGBM, LinearModel

logger = logging.getLogger("qtof")

def fit_processor(X_train, output_dir):
    processor = utils.CustomPreprocessor(transforms = ['quantile'])
    processor.fit(X_train)
    
    with open(os.path.join(output_dir, 'processor.pkl'),'wb') as f:
        pickle.dump(processor, f)
    
    return processor

def train_all_models(norm_X_train, y_train, output_dir):
    model_output_dir = os.path.join(output_dir, 'models')
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    
    logger.info("")
    logger.info("Training Models:")
    # random forest
    logger.info("    Training Random Forest model...")
    rf_model = RandomForest('RF',  {})
    best_params = rf_model.random_search(4, norm_X_train, y_train)
    rf_model = RandomForest('RF', best_params)
    rf_model.run_cv(norm_X_train, y_train, 4)
    with open(os.path.join(model_output_dir, 'rf_model.pkl'), 'wb') as f:
        pickle.dump(rf_model, f)

    # LGBM
    logger.info("    Training LGBM model...")
    lgbm_model = LGBM('LGBM', {'n_jobs': -1})
    best_params = lgbm_model.random_search(4, norm_X_train, y_train)
    lgbm_model = LGBM('LGBM', best_params)
    lgbm_model.run_cv(norm_X_train, y_train, 4)
    with open(os.path.join(model_output_dir, 'lgbm_model.pkl'), 'wb') as f:
        pickle.dump(lgbm_model, f)

    # lasso
    logger.info("    Training LASSO model...")
    lasso = LinearModel('lasso', {'penalty' : 'l1', 'solver' : 'liblinear'})
    best_params = lasso.grid_search(4, norm_X_train, y_train, visualize=False)
    lasso = LinearModel('lasso', dict({'penalty' : 'l1', 'solver' : 'liblinear'}, **best_params))
    lasso.run_cv(norm_X_train, y_train, 4)
    with open(os.path.join(model_output_dir, 'lasso.pkl'), 'wb') as f:
        pickle.dump(lasso, f)

    # logistic
    logger.info("    Training logistic model...")
    logistic = LinearModel('logistic', {'solver' : 'lbfgs'})
    best_params = logistic.grid_search(4, norm_X_train, y_train, visualize=True)
    logistic = LinearModel('logistic', dict({'solver' : 'lbfgs'}, **best_params))
    logistic.run_cv(norm_X_train, y_train, 4)
    with open(os.path.join(model_output_dir, 'logistic.pkl'), 'wb') as f:
        pickle.dump(logistic, f)
    
    return rf_model, lgbm_model, lasso, logistic
