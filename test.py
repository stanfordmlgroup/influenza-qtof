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
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.metrics import confusion_matrix


logger = logging.getLogger("qtof")

def predict_all_models(norm_X_train, y_train, norm_X_test, y_test, all_models):
    output = {}
    for model in all_models:
        y_train_prob = model.avgd_folds_decision_function(norm_X_train)
        y_pred_prob = model.avgd_folds_decision_function(norm_X_test)

        # get best threshold
        best_threshold = utils.get_optimal_threshold(y_train, y_train_prob)    

        output[model.model_name] = {'model': model, 'best_threshold': best_threshold,'y_pred_prob': y_pred_prob, 'y_test': y_test}
    return output
        
def get_optimal_operating_point(y_true, y_pred):
    thresholds = []
    sens_plus_spec = []
    for threshold in np.arange(0, 1, .01):
        preds = [1 if p > threshold else 0 for p in y_pred]
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        thresholds.append(threshold)
        sens_plus_spec.append(sensitivity + specificity)
    max_idx = np.argmax(sens_plus_spec)
    return thresholds[max_idx]

def get_high_sensitivity_operating_point(y_true, y_pred, target=.9):
    thresholds = []
    sensitivities = []
    for threshold in np.arange(0, 1, .01):
        preds = [1 if p > threshold else 0 for p in y_pred]
        tn, fp, fn ,tp = confusion_matrix(y_true, preds).ravel()
        sensitivity = tp / (tp + fn)
        thresholds.append(threshold)
        sensitivities.append(sensitivity)
        
    for idx, s in enumerate(sensitivities):
        if s <= target:
            return thresholds[idx-1]
        
def print_operating_points(output):
    for model_name, data in output.items():
        model = data['model']

        y_fold_preds = model.cv_storage.y_pred
        y_validations = model.cv_storage.y_test  

        optimal_thresholds = []
        high_sens_thresholds = []
        for y_pred, y_val in zip(y_fold_preds, y_validations):
            optimal_thresholds.append(get_optimal_operating_point(y_val, y_pred))
            high_sens_thresholds.append(get_high_sensitivity_operating_point(y_val, y_pred))

        logger.info(model_name)
        logger.info("High Sensitivity Thresholds:")
        logger.info(high_sens_thresholds)
        logger.info("Optimal Thresholds:")
        logger.info(optimal_thresholds)
        logger.info("")
        output[model_name]['high_sensitivity_threshold'] = np.mean(high_sens_thresholds)
        output[model_name]['optimal_thresholds'] = np.mean(optimal_thresholds)
        
def plot_all_predictions(output, output_dir):
    plt.figure(figsize=(6,6))

    for model_name, data in output.items():
        model = data['model']
        y_test = data['y_test']
        y_pred_prob = data['y_pred_prob']
        fpr, tpr, a = model.get_prediction_stats(y_test, y_pred_prob)
        plt.plot(fpr, tpr, lw=2, label=model_name + ' ROC curve - (area = %0.2f)' % a)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Linear ROC - Test Set')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'test_roc.png'))
    
def compute_auc(output):
    for model_name, data in output.items():
        logger.info(model_name)
        y_test = data['y_test']
        y_pred_prob = data['y_pred_prob']
        model = data['model']
        fpr, tpr, auc = model.get_prediction_stats(y_test, y_pred_prob)
        logger.info(delong.compute_stats(.95, y_pred_prob, y_test))
