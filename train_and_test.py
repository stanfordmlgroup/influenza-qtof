#!/usr/bin/env python
# Train and test
import sys
import os
sys.path.append('src/')
import train
import test
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import model_selection
import models
import plotting 
import delong
import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
import logging
import pickle

def config_logger(output_dir):
    logger = logging.getLogger("qtof")
    logger.setLevel(logging.DEBUG)
    # create handlers
    fh = logging.FileHandler(os.path.join(output_dir, 'train_test_log.txt'))
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', default='results', help='Output dir for results')
    parser.add_argument('-d', '--data', default='', help='Path to CSV file of data, see README for format.')
    parser.add_argument('-l', '--labels', default= '', help='Path to CSV file of labels, see README for format.')
    parser.add_argument('-t', '--test_points', default=50, help="Number of samples to set aside for training.")
    return parser.parse_args()
    

def create_output_dirs(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def main():
    args = _parse_args()
    create_output_dirs(args.output_dir)
    logger = config_logger(args.output_dir)
    
    # preprocess and split into train and test
    logger.info("Loading and preprocessing data...")
    all_data = utils.load_unformatted_data(args.data, args.labels)
    all_data = utils.map_label_column(all_data)
    all_data, flu_mapping, subtype_mapping = utils.encode_labels(all_data)

    # drop zero cols
    all_data = all_data.loc[:, all_data.any()]
    col_names = all_data.columns[:-2]
    
    # train test split
    train_data, test_data = utils.train_test_split(all_data, test_size=args.test_points)
    
    # print some stats on data
    logger.info("Total samples: {}".format(len(all_data)))
    logger.info("    Train samples: {}".format(len(train_data)))
    logger.info(train_data.flu.value_counts())
    logger.info("    Test samples: {}".format(len(test_data)))
    logger.info(test_data.flu.value_counts())
    
    # preprocess
    X_train, y_train = utils.df_to_array(train_data)
    X_test, y_test = utils.df_to_array(test_data)
    
    # preprocess
    processor = train.fit_processor(X_train, args.output_dir)
    norm_X_train = processor.transform(X_train)
    norm_X_test = processor.transform(X_test)
    
    # Train all models
    rf_model, lgbm_model, lasso, logistic = train.train_all_models(norm_X_train, y_train, args.output_dir)
    
    # Predict
    all_models = [rf_model, lgbm_model, lasso, logistic]
    output = test.predict_all_models(norm_X_train, y_train, norm_X_test, y_test, all_models)
    
    # Print and save plots
    test.print_operating_points(output)
    test.plot_all_predictions(output, args.output_dir)
    test.compute_auc(output)
    
if __name__ == "__main__":
    main()