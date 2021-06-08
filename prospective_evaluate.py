#!/usr/bin/env python
# Train and test
import sys
import os
sys.path.append('src/')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import model_selection
from models import LGBM
import plotting 
import delong
import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
import logging
import pickle

# get LGBM model checkpoint from retrospective training
with open('prospective_processor.pkl', 'rb') as f:
	processor = pickle.load(f)

# get preprocessor fitted on retrospective data
with open('prospective_model_output.pkl', 'rb') as f:
	output = pickle.load(f)

def load_and_evaluate_prospective_data(prospective_file):
	prospective_data = pd.read_csv(prospective_file, index_col=0)
	X_prospective_test, y_prospective_test = utils.df_to_array(prospective_data)
	prospective_norm_X_test = processor.transform(X_prospective_test)

	lgbm_model = output['LGBM']['model']
	y_pred = lgbm_model.avgd_folds_decision_function(prospective_norm_X_test)
	print('Prospective Dataset Results')
	y_test = y_prospective_test
	y_pred_prob = y_pred
	fpr, tpr, auc = lgbm_model.get_prediction_stats(y_test, y_pred_prob)
	d = delong.compute_stats(.95, y_pred_prob, y_test)
	print(d)

print('Evaluating on Validation Cohort Lab 1...')
load_and_evaluate_prospective_data('data/prospective_influenza_dataset/validation_cohort_lab_1.csv')
print('====================================')
print('Evaluating on Validation Cohort Lab 2...')
load_and_evaluate_prospective_data('data/prospective_influenza_dataset/validation_cohort_lab_2.csv')

