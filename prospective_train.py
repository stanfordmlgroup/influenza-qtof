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


def prep_restrospective_data():
	# read data
	all_data = utils.load_unformatted_data('data/expanded_flu_dataset/Version3_Unfiltered/20191105 resp data no filter revised 1 2020.csv', 'data/expanded_flu_dataset/Version3_Unfiltered/Sample_key_csv revised 1 2020.csv')
	all_data = utils.map_label_column(all_data)
	all_data, flu_mapping, subtype_mapping = utils.encode_labels(all_data)

	# drop zero cols
	all_data = all_data.loc[:, all_data.any()]
	col_names = all_data.columns[:-2]

	# from results of QTOF paper
	# determined by SHAP as the most important metabolites
	# for LGBM classification
	top_features = [
		'0.81_84.0447m/z', 
		'0.81_130.0507m/z', 
		'10.34_106.0865m/z',
		'4.73_422.1307m/z', 
		'9.34_349.0774n', 
		'10.87_249.1085m/z',
		'9.28_956.3750n', 
		'10.89_352.2131n', 
		'7.88_86.0965m/z',
		'1.78_63.0440m/z', 
		'10.85_214.1306m/z', 
		'8.36_144.0935n',
		'10.23_227.0793m/z', 
		'8.65_211.1376m/z', 
		'1.30_230.0961m/z',
		'10.33_178.1441m/z', 
		'11.61_102.1268m/z', 
		'2.11_232.1182m/z',
		'7.00_634.7114m/z', 
		'3.21_201.0740m/z'
	]

	# Get just top columns from dataset
	all_data = all_data.loc[:, top_features + ['flu', 'subtype']]

	# train test split
	train, test = utils.train_test_split(all_data, test_size=50)

	return train, test

def save_retrospective_models():
	train, test = prep_restrospective_data()
	X_train, y_train = utils.df_to_array(train)
	X_test, y_test = utils.df_to_array(test)

	processor = utils.CustomPreprocessor(transforms = ['quantile'])
	processor.fit(X_train)
	norm_X_train = processor.transform(X_train)
	norm_X_test = processor.transform(X_test)

	with open("prospective_processor.pkl", 'wb') as f:
		pickle.dump(processor, f)

	# 4-fold cross-validation tuning + training
	lgbm_model = LGBM('LGBM', {'n_jobs': -1})
	best_params = lgbm_model.random_search(4, norm_X_train, y_train)
	lgbm_model = LGBM('LGBM', best_params)
	lgbm_model.run_cv(norm_X_train, y_train, 4)

	output = {}
	y_train_prob = lgbm_model.avgd_folds_decision_function(norm_X_train)
	y_pred_prob = lgbm_model.avgd_folds_decision_function(norm_X_test)
	best_threshold = utils.get_optimal_threshold(y_train, y_train_prob)
	output[lgbm_model.model_name] = {'model': lgbm_model, 'best_threshold': best_threshold,'y_pred_prob': y_pred_prob, 'y_test': y_test}

	with open('prospective_model_output.pkl', 'wb') as f:
		pickle.dump(output, f)

if __name__ == "__main__":
	save_retrospective_models()
	print('=============')
	print('LGBM Model Checkpoint successfully saved.')


